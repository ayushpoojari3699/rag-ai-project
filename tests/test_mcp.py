"""Tests for the DocuMind MCP server and the free Ollama MCP client (no Ollama needed).

The server is started as a real subprocess and spoken to over stdio, exactly as an MCP
host would. The Ollama model is replaced by a scripted fake.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # "tests" may clash with an installed package

from test_rag import HANDBOOK, TRAVEL  # noqa: E402


@pytest.fixture()
def indexed(tmp_path, monkeypatch):
    """Index two small PDFs into a temporary FAISS store and point DocuMind at it."""
    monkeypatch.setenv("DOCUMIND_DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("DOCUMIND_INDEX_DIR", str(tmp_path / "vector_store"))
    import rag_core
    importlib.reload(rag_core)
    engine = rag_core.RAGEngine(llm=object())
    for name, data in [("handbook.pdf", HANDBOOK), ("travel.pdf", TRAVEL)]:
        pdf = tmp_path / name
        pdf.write_bytes(data)
        engine.add_pdf(pdf)
    return engine


def server_params():
    from mcp import StdioServerParameters
    return StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")],
                                 cwd=str(ROOT), env=dict(os.environ))


def run(coro):
    return asyncio.run(coro)


def test_search_tool_called_directly(indexed):
    import mcp_server
    importlib.reload(mcp_server)
    mcp_server.set_engine(indexed)
    hits = mcp_server.search_documents("resignation notice period", k=2)
    assert (hits[0]["file"], hits[0]["page"]) == ("handbook.pdf", 3)
    assert "60 days" in hits[0]["text"]
    assert len(hits) == 2
    assert len(mcp_server.search_documents("hotel", k=99)) <= mcp_server.MAX_K  # k is capped
    assert "error" in mcp_server.search_documents("   ")[0]
    assert {d["file"] for d in mcp_server.list_documents()} == {"handbook.pdf", "travel.pdf"}


def test_server_over_stdio(indexed):
    """Discover the tools and call them through the MCP protocol."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp_chat import tool_result_text

    async def go():
        async with stdio_client(server_params()) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            assert names == {"search_documents", "list_documents"}
            res = await s.call_tool("search_documents",
                                    {"query": "hotel reimbursement per night", "k": 1})
            return json.loads(tool_result_text(res))

    data = run(go())
    hits = data["result"] if isinstance(data, dict) else data
    assert (hits[0]["file"], hits[0]["page"]) == ("travel.pdf", 2)


def test_ollama_client_runs_the_tool_loop(indexed, monkeypatch):
    """A scripted 'model' asks for a search, then answers from the tool result."""
    import httpx
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    import mcp_chat

    seen = []

    async def fake_chat(http, messages, tools):
        seen.append([t["function"]["name"] for t in tools])
        if messages[-1]["role"] == "user":
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "search_documents",
                                                 "arguments": {"query": "notice period"}}}]}
        tool_output = messages[-1]["content"]
        assert messages[-1]["role"] == "tool" and "60 days" in tool_output
        return {"role": "assistant", "content": "60 days (handbook.pdf, page 3)."}

    monkeypatch.setattr(mcp_chat, "ollama_chat", fake_chat)

    async def go():
        async with stdio_client(server_params()) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            tools = mcp_chat.mcp_tools_to_ollama((await s.list_tools()).tools)
            history = [{"role": "system", "content": mcp_chat.SYSTEM_PROMPT}]
            async with httpx.AsyncClient() as http:
                return await mcp_chat.answer(s, http, tools, history,
                                             "What is the notice period?", verbose=False)

    assert run(go()) == "60 days (handbook.pdf, page 3)."
    assert set(seen[0]) == {"search_documents", "list_documents"}
