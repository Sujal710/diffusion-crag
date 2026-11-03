"""
CRAG pipeline as a LangGraph state machine.

  retrieve -> evaluate_documents --CORRECT-----> refine_internal ----------------------> generate
                                 --INCORRECT---> web_search -> refine_external ---------> generate
                                 --AMBIGUOUS---> refine_internal + web_search + refine -> generate

  generate -> check_answer --correct / out of attempts--> END
                           --wrong--> reframe_question -> retrieve   (whole pipeline again)
"""
import asyncio
import json
import os
import re
from typing import List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession
from mcp.client.sse import sse_client

from . import llm, vectorstore
from .config import get_settings

Verdict = Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]


class CRAGState(TypedDict, total=False):
    namespace: Optional[str]      # Pinecone namespace to search (None = all of them)
    original_question: str        # what the user asked (never changes)
    question: str                 # current (possibly reframed) question used for retrieval
    attempt: int                  # 1, 2, 3 ...
    documents: List[Document]     # docs from Pinecone
    scores: List[float]           # evaluator score per doc
    verdict: Verdict
    internal_knowledge: str       # refined strips from Pinecone docs
    external_knowledge: str       # refined strips from web
    web_sources: List[str]
    knowledge: str                # what the generator actually sees
    answer: str
    answer_ok: bool
    feedback: str
    trace: List[dict]             # log of every attempt, returned to the caller


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _decompose(text: str, sentences_per_strip: int = 2) -> List[str]:
    """Split text into small 'knowledge strips' of a couple of sentences each."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [
        " ".join(sentences[i:i + sentences_per_strip])
        for i in range(0, len(sentences), sentences_per_strip)
    ]


async def _refine(question: str, texts: List[str]) -> str:
    """Decompose -> filter -> recompose."""
    strips: List[str] = []
    for t in texts:
        strips.extend(_decompose(t))
    if not strips:
        return ""

    strips = strips[:60]  # keep the filter prompt a sane size
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(strips))
    try:
        selection = await llm.strip_filter_chain().ainvoke(
            {"question": question, "strips": numbered}
        )
        kept = [strips[i] for i in selection.keep if 0 <= i < len(strips)]
    except Exception:
        kept = strips  # if the filter fails, fall back to unfiltered text
    return "\n".join(kept)


MCP_SSE_URL = os.getenv("MCP_SSE_URL", "http://127.0.0.1:8001/sse")


async def _mcp_web_search(query: str, max_results: int) -> List[dict]:
    """Web search is done ONLY by the MCP server (crag_app.mcp_server); call it over SSE."""
    async with sse_client(MCP_SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("web_search", {"query": query, "max_results": max_results})
    text = "".join(getattr(c, "text", "") for c in res.content)
    if getattr(res, "is_error", getattr(res, "isError", False)):  # v2 / v1 field name
        raise RuntimeError(f"MCP web_search failed: {text}")
    return json.loads(text) if text else []


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
async def retrieve(state: CRAGState) -> CRAGState:
    s = get_settings()
    docs = await asyncio.to_thread(
        vectorstore.search, state["question"], s.top_k, state.get("namespace"))
    return {"documents": docs}


async def evaluate_documents(state: CRAGState) -> CRAGState:
    s = get_settings()
    docs = state.get("documents", [])
    chain = llm.relevance_chain()

    async def score(doc: Document) -> float:
        try:
            r = await chain.ainvoke({"question": state["question"], "document": doc.page_content})
            return max(0.0, min(1.0, r.score))
        except Exception:
            return 0.0

    scores = list(await asyncio.gather(*(score(d) for d in docs)))

    if scores and max(scores) >= s.upper_threshold:
        verdict: Verdict = "CORRECT"
    elif not scores or max(scores) < s.lower_threshold:
        verdict = "INCORRECT"
    else:
        verdict = "AMBIGUOUS"

    # keep only docs that weren't scored as junk
    kept = [d for d, sc in zip(docs, scores) if sc >= s.lower_threshold]
    return {"scores": scores, "verdict": verdict, "documents": kept}


async def refine_internal(state: CRAGState) -> CRAGState:
    texts = [d.page_content for d in state.get("documents", [])]
    return {"internal_knowledge": await _refine(state["question"], texts)}


async def web_search(state: CRAGState) -> CRAGState:
    s = get_settings()
    rewritten = await llm.web_query_chain().ainvoke({"question": state["question"]})
    hits = await _mcp_web_search(rewritten.query, s.web_results)
    texts = [h.get("content", "") for h in hits]
    return {
        "external_knowledge": await _refine(state["question"], texts),
        "web_sources": [h.get("url", "") for h in hits],
    }


async def refine_and_search(state: CRAGState) -> CRAGState:
    """AMBIGUOUS path: refine internal docs AND search the web, in parallel."""
    internal, external = await asyncio.gather(refine_internal(state), web_search(state))
    return {**internal, **external}


async def generate(state: CRAGState) -> CRAGState:
    verdict = state["verdict"]
    parts = []
    if verdict in ("CORRECT", "AMBIGUOUS") and state.get("internal_knowledge"):
        parts.append("From knowledge base:\n" + state["internal_knowledge"])
    if verdict in ("INCORRECT", "AMBIGUOUS") and state.get("external_knowledge"):
        parts.append("From web search:\n" + state["external_knowledge"])
    knowledge = "\n\n".join(parts) or "(no relevant knowledge found)"

    # Answer the user's ORIGINAL question; the reframed one is only for retrieval.
    msg = await llm.generate_chain().ainvoke(
        {"question": state["original_question"], "knowledge": knowledge}
    )
    return {"knowledge": knowledge, "answer": msg.content}


async def check_answer(state: CRAGState) -> CRAGState:
    try:
        r = await llm.check_chain().ainvoke({
            "question": state["original_question"],
            "knowledge": state["knowledge"],
            "answer": state["answer"],
        })
        ok, feedback = r.is_correct, r.feedback
    except Exception as e:
        ok, feedback = True, f"checker failed ({e}); accepting answer"

    trace = list(state.get("trace", []))
    trace.append({
        "attempt": state["attempt"],
        "question_used": state["question"],
        "verdict": state["verdict"],
        "doc_scores": state.get("scores", []),
        "web_sources": state.get("web_sources", []),
        "answer": state["answer"],
        "answer_ok": ok,
        "feedback": feedback,
    })
    return {"answer_ok": ok, "feedback": feedback, "trace": trace}


async def reframe_question(state: CRAGState) -> CRAGState:
    r = await llm.reframe_chain().ainvoke({
        "original_question": state["original_question"],
        "question": state["question"],
        "feedback": state["feedback"],
    })
    # reset per-attempt fields so the next run starts clean
    return {
        "question": r.question,
        "attempt": state["attempt"] + 1,
        "documents": [], "scores": [],
        "internal_knowledge": "", "external_knowledge": "", "web_sources": [],
    }


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------
def route_by_verdict(state: CRAGState) -> str:
    return {
        "CORRECT": "refine_internal",
        "INCORRECT": "web_search",
        "AMBIGUOUS": "refine_and_search",
    }[state["verdict"]]


def route_after_check(state: CRAGState) -> str:
    if state["answer_ok"] or state["attempt"] >= get_settings().max_attempts:
        return END
    return "reframe_question"


# ---------------------------------------------------------------------------
# build graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("evaluate_documents", evaluate_documents)
    g.add_node("refine_internal", refine_internal)
    g.add_node("web_search", web_search)
    g.add_node("refine_and_search", refine_and_search)
    g.add_node("generate", generate)
    g.add_node("check_answer", check_answer)
    g.add_node("reframe_question", reframe_question)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "evaluate_documents")
    g.add_conditional_edges("evaluate_documents", route_by_verdict,
                            ["refine_internal", "web_search", "refine_and_search"])
    g.add_edge("refine_internal", "generate")
    g.add_edge("web_search", "generate")
    g.add_edge("refine_and_search", "generate")
    g.add_edge("generate", "check_answer")
    g.add_conditional_edges("check_answer", route_after_check, ["reframe_question", END])
    g.add_edge("reframe_question", "retrieve")  # loop: run the whole pipeline again
    return g.compile()


crag_graph = build_graph()


async def run_crag(question: str, namespace: Optional[str] = None) -> dict:
    s = get_settings()
    final = await crag_graph.ainvoke(
        {"original_question": question, "question": question, "attempt": 1, "trace": [],
         "namespace": namespace},
        # each attempt is ~8 steps; give headroom for all attempts
        config={"recursion_limit": 10 * s.max_attempts + 10},
    )
    return {
        "question": question,
        "answer": final["answer"],
        "verified": final["answer_ok"],
        "attempts": final["attempt"],
        "final_question_used": final["question"],
        "trace": final["trace"],
    }
