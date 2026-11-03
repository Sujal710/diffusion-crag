# CRAG — Corrective RAG with a self-check loop

Groq (LLMs) · LangGraph (pipeline) · Pinecone (vector DB) · Tavily (web search) · FastAPI (REST) · MCP (tools for Claude Desktop / Cursor)

## How it works

```
question
  └─► retrieve (Pinecone) ─► evaluate docs (Groq scores each 0–1)
          ├─ CORRECT   (any doc ≥ 0.7)  → refine internal docs
          ├─ INCORRECT (all docs < 0.3) → rewrite query → web search → refine
          └─ AMBIGUOUS (in between)     → refine internal + web search, combined
                         ↓
                     generate answer (Groq)
                         ↓
                  CHECK ANSWER (Groq judge)
          ├─ correct, or MAX_ATTEMPTS reached → return
          └─ wrong → reframe question → back to retrieve (whole pipeline again)
```

"Refine" = decompose docs into small strips → LLM keeps only useful strips → join them.
The reframed question is used for retrieval; the final answer always targets the user's original question.

## Files

| File | What it does |
|---|---|
| `crag_app/config.py` | All settings from `.env` |
| `crag_app/llm.py` | Groq models + every prompt (evaluator, strip filter, query rewriter, generator, checker, reframer) |
| `crag_app/vectorstore.py` | Pinecone serverless index (auto-created on first run), ingest, search, stats (FastEmbed embeddings, run locally) |
| `crag_app/graph.py` | The LangGraph state machine (CRAG + check loop) |
| `crag_app/api.py` | FastAPI: `/ingest`, `/ask`, `/stats`, `/health` |
| `crag_app/mcp_server.py` | MCP server: tools `ask_crag`, `web_search`, `add_documents` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
```

Keys you need:
- **Groq**: console.groq.com → API Keys
- **Pinecone**: app.pinecone.io → API Keys → create a key. You don't need to create the index yourself; the app creates `crag-docs` (384 dims, cosine, serverless aws/us-east-1) on first use. The free Starter plan only allows aws/us-east-1.
- **Tavily**: tavily.com (free tier) — used for the web-search fallback

Groq's model list changes often. Defaults are `openai/gpt-oss-120b` (answers) and `openai/gpt-oss-20b` (grading). If one is retired, set `GROQ_GENERATOR_MODEL` / `GROQ_GRADER_MODEL` in `.env` from console.groq.com/docs/models.

## Run the API

```bash
uvicorn crag_app.api:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

```bash
# 1. add knowledge
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"texts": ["Astana is the capital of Kazakhstan. It was renamed from Nur-Sultan in 2022."]}'

# 2. check vectors landed in Pinecone
curl localhost:8000/stats

# 3. ask
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of Kazakhstan?"}'
```

Response:

```json
{
  "question": "...",
  "answer": "...",
  "verified": true,          // did the checker accept the final answer?
  "attempts": 1,             // how many times the pipeline ran
  "final_question_used": "...",
  "trace": [ { "attempt": 1, "verdict": "CORRECT", "doc_scores": [...],
               "web_sources": [...], "answer": "...", "answer_ok": true, "feedback": "..." } ]
}
```

## Run as an MCP server

The server uses HTTP + SSE (no stdio): `python -m crag_app.mcp_server` → `http://127.0.0.1:8001/sse`

Point your MCP client at that URL (e.g. `{"mcpServers": {"crag": {"url": "http://127.0.0.1:8001/sse"}}}`).

Works with MCP Python SDK v2 and v1.

## Tuning (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `PINECONE_NAMESPACE` | default | Keeps different datasets separate inside one index |
| `MAX_ATTEMPTS` | 3 | Max full pipeline runs before returning the best effort |
| `UPPER_THRESHOLD` | 0.7 | Score needed for CORRECT |
| `LOWER_THRESHOLD` | 0.3 | Below this for all docs → INCORRECT |
| `TOP_K` | 5 | Docs pulled from Pinecone |
| `WEB_RESULTS` | 5 | Web results pulled from Tavily |

Cost note: each attempt makes roughly `TOP_K + 4` Groq calls (one per doc score, plus filter, generate, check, and a reframe if it loops). Groq's free tier has tight per-minute limits, so lower `TOP_K` or `MAX_ATTEMPTS` if you hit 429 errors.

Note: if you ever change `EMBEDDING_MODEL` to one with a different dimension, use a new `PINECONE_INDEX` name, because a Pinecone index's dimension is fixed when it's created.
