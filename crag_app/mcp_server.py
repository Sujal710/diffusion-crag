"""
MCP server exposing the CRAG pipeline as tools, so Claude Desktop, Cursor or any
MCP client can call it.

  Transport: HTTP + SSE only (no stdio).
  Run:       python -m crag_app.mcp_server   ->   http://127.0.0.1:8001/sse

Works with the MCP Python SDK v2 (MCPServer) and v1 (FastMCP).
"""
import asyncio
import json
from typing import List

from tavily import AsyncTavilyClient

from . import vectorstore
from .config import get_settings

HTTP_PORT = 8001

try:  # MCP SDK v2
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("crag")
    _V2 = True
except ImportError:  # MCP SDK v1
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("crag", port=HTTP_PORT)
    _V2 = False


@mcp.tool()
async def ask_crag(question: str) -> str:
    """Answer a question with Corrective RAG (Pinecone + web fallback + self-check loop).
    Returns JSON with answer, verified flag, attempts and a per-attempt trace."""
    from .graph import run_crag  # lazy: avoid import cycle with graph

    print(f"[mcp] ask_crag called: {question!r}", flush=True)
    result = await run_crag(question)
    print(f"[mcp] ask_crag done: verified={result['verified']} attempts={result['attempts']}", flush=True)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def web_search(query: str, max_results: int = 0) -> str:
    """Search the web (Tavily). Returns JSON list of {title, url, content}.
    This is the ONLY place in the project that queries the web."""
    s = get_settings()
    n = max_results or s.web_results
    print(f"[mcp] web_search called: query={query!r} max_results={n}", flush=True)
    client = AsyncTavilyClient(api_key=s.tavily_api_key)
    result = await client.search(query, max_results=n)
    hits = [
        {"title": h.get("title", ""), "url": h.get("url", ""), "content": h.get("content", "")}
        for h in result.get("results", [])
    ]
    print(f"[mcp] web_search returned {len(hits)} results", flush=True)
    for h in hits:
        print(f"[mcp]   - {h['url']}", flush=True)
    return json.dumps(hits, indent=2)


@mcp.tool()
async def add_documents(texts: List[str]) -> str:
    """Add raw text documents to the CRAG knowledge base (Pinecone)."""
    print(f"[mcp] add_documents called: {len(texts)} texts", flush=True)
    n = await asyncio.to_thread(vectorstore.add_texts, texts)
    print(f"[mcp] add_documents stored {n} chunks", flush=True)
    return f"Stored {n} chunks."


if __name__ == "__main__":
    print(f"[mcp] starting SSE server on http://127.0.0.1:{HTTP_PORT}/sse", flush=True)
    if _V2:
        mcp.run("sse", port=HTTP_PORT)
    else:
        mcp.run(transport="sse")
