"""DocuMind MCP server: exposes DocuMind's document search as MCP tools.

Any MCP host (the free local `mcp_chat.py` + Ollama client in this repo, Gemini CLI,
Claude Desktop, VS Code, the MCP Inspector) can launch this file over stdio, discover
the tools and call them.

Design choices
    * Read-only: the tools only search and list; nothing can add, change or delete
      documents, so a misbehaving model can't damage the index.
    * Retrieval only: the server returns cited passages and the HOST's model writes the
      answer, so no LLM has to run inside the server.
    * stdout is the protocol channel in stdio mode, so all logging goes to stderr.

Run
    python ingest.py                 # index the PDFs in docs/ first
    python mcp_server.py             # normally started by an MCP host, not by hand
"""

from __future__ import annotations

import logging
import sys

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:  # SDK 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from rag_core import DEFAULT_TOP_K, NoDocumentsError, RAGEngine

MAX_K = 10

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[documind-mcp] %(message)s")
log = logging.getLogger("documind-mcp")

mcp = MCPServer(
    "documind",
    instructions=(
        "Search the user's own PDF documents indexed by DocuMind. Call search_documents "
        "before answering questions about those documents and cite file name and page."
    ),
)

_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    """Load the embeddings and FAISS index once, on the first tool call."""
    global _engine
    if _engine is None:
        log.info("loading embeddings and FAISS index ...")
        _engine = RAGEngine()
    return _engine


def set_engine(engine: RAGEngine) -> None:
    """Inject an engine (used by the tests)."""
    global _engine
    _engine = engine


@mcp.tool()
def search_documents(query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search the user's indexed PDF documents.

    Returns the most relevant passages, best first, each with the file name, page
    number, a cosine similarity score (higher is more relevant) and the passage text.
    Answer only from these passages and cite them as (file, page). If none of them
    is relevant, say the documents don't contain the answer.

    Args:
        query: What to look for, in natural language.
        k: How many passages to return (1-10, default 4).
    """
    query = (query or "").strip()
    if not query:
        return [{"error": "query must not be empty"}]
    k = max(1, min(int(k), MAX_K))
    try:
        hits = get_engine().retrieve(query, k)
    except NoDocumentsError:
        return [{"error": "No documents are indexed yet. Upload or ingest a PDF first."}]
    log.info("search_documents(%r, k=%d) -> %d hits", query, k, len(hits))
    return [
        {
            "rank": i,
            "file": doc.metadata.get("source"),
            "page": doc.metadata.get("page", 0) + 1,
            "score": score,
            "text": doc.page_content,
        }
        for i, (doc, score) in enumerate(hits, start=1)
    ]


@mcp.tool()
def list_documents() -> list[dict]:
    """List the PDF files currently indexed and how many chunks each one has."""
    return get_engine().documents()


if __name__ == "__main__":
    log.info("starting (stdio)")
    mcp.run()  # stdio transport by default
