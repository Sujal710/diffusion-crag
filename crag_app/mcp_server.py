"""
MCP server exposing the CRAG pipeline as tools, so Claude Desktop, Cursor or any
MCP client can call it.

  stdio (Claude Desktop / Cursor):  python -m crag_app.mcp_server
  HTTP:                             python -m crag_app.mcp_server --http   (http://127.0.0.1:8001/mcp)

Works with the MCP Python SDK v2 (MCPServer) and v1 (FastMCP).
"""
import asyncio
import json
import sys
from typing import List

from . import vectorstore
from .graph import run_crag

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
    return json.dumps(await run_crag(question), indent=2, default=str)


@mcp.tool()
async def add_documents(texts: List[str]) -> str:
    """Add raw text documents to the CRAG knowledge base (Pinecone)."""
    n = await asyncio.to_thread(vectorstore.add_texts, texts)
    return f"Stored {n} chunks."


if __name__ == "__main__":
    if "--http" in sys.argv:
        if _V2:
            mcp.run("streamable-http", port=HTTP_PORT)
        else:
            mcp.run(transport="streamable-http")
    else:
        mcp.run("stdio")
