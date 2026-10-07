"""A free, fully local MCP client: chat with your documents through Ollama + MCP.

    you --> Ollama model (tool calling) --> MCP client --stdio--> mcp_server.py --> FAISS

The script starts `mcp_server.py` as a subprocess, discovers its tools with
`tools/list`, hands them to a local Ollama model, and runs every tool call the model
makes through MCP until the model gives a final answer. No API keys, no cost.

Setup (once)
    ollama pull llama3.1:8b          # needs a tool-calling model; llama3 (v1) has no tools
    pip install -r requirements.txt  # includes mcp and httpx
    python ingest.py                 # index the PDFs in docs/

Use
    python mcp_chat.py "What is the notice period?"     # one question
    python mcp_chat.py                                  # interactive chat ("exit" to quit)

Environment
    DOCUMIND_AGENT_MODEL   Ollama model with tool support (default llama3.1:8b;
                           qwen2.5:7b and qwen3:8b also work well)
    OLLAMA_BASE_URL        default http://localhost:11434
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
MODEL = os.getenv("DOCUMIND_AGENT_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = (
    "You are DocuMind, an assistant that answers questions about the user's PDF documents. "
    "For any question about the documents, call search_documents first; call list_documents "
    "if the user asks which files exist. Answer ONLY from the returned passages and cite each "
    "fact as (file, page). If the passages don't contain the answer, say: "
    "\"I couldn't find this in the uploaded documents.\" Keep answers short."
)


# --------------------------------------------------------------------------- helpers
def _attr(obj, *names, default=None):
    """Read a field that is snake_case in MCP SDK 2.x and camelCase in 1.x."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def mcp_tools_to_ollama(tools) -> list[dict]:
    """Convert MCP tool definitions into Ollama's function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": _attr(t, "input_schema", "inputSchema",
                                    default={"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def tool_result_text(result) -> str:
    """Flatten an MCP CallToolResult into text the model can read."""
    structured = _attr(result, "structured_content", "structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    parts = [getattr(c, "text", "") for c in (result.content or [])]
    text = "\n".join(p for p in parts if p)
    if _attr(result, "is_error", "isError", default=False):
        return f"TOOL ERROR: {text}"
    return text


async def ollama_chat(http: httpx.AsyncClient, messages: list[dict], tools: list[dict]) -> dict:
    resp = await http.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": MODEL, "messages": messages, "tools": tools,
              "stream": False, "options": {"temperature": 0}},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["message"]


# --------------------------------------------------------------------------- agent loop
async def answer(session: ClientSession, http: httpx.AsyncClient, tools: list[dict],
                 history: list[dict], question: str, verbose: bool = True) -> str:
    history.append({"role": "user", "content": question})
    for _ in range(MAX_TOOL_ROUNDS):
        msg = await ollama_chat(http, history, tools)
        history.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", "").strip()
        for call in calls:
            fn = call["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # some models send JSON as a string
                args = json.loads(args or "{}")
            if verbose:
                print(f"  -> tool call: {fn['name']}({json.dumps(args, ensure_ascii=False)})",
                      file=sys.stderr)
            result = await session.call_tool(fn["name"], args)
            history.append({"role": "tool", "tool_name": fn["name"],
                            "content": tool_result_text(result)})
    return "Stopped after too many tool calls without a final answer."


async def safe_answer(session, http, tools, history, question) -> str:
    """answer() with readable errors instead of a traceback."""
    try:
        return await answer(session, http, tools, history, question)
    except httpx.ConnectError:
        return f"Can't reach Ollama at {OLLAMA_URL}. Start it (open the Ollama app or run `ollama serve`)."
    except (httpx.HTTPError, RuntimeError) as exc:
        return f"Error: {exc}"


async def main(question: str | None = None) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(HERE / "mcp_server.py")],
        cwd=str(HERE),
        env=dict(os.environ),  # our own local server: pass DOCUMIND_*, HF cache, venv paths
    )
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
        tools = mcp_tools_to_ollama(listed.tools)
        print(f"Connected to DocuMind MCP server. Tools: {[t['function']['name'] for t in tools]}",
              file=sys.stderr)
        print(f"Model: {MODEL} via Ollama (free, local)\n", file=sys.stderr)

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        async with httpx.AsyncClient(timeout=300) as http:
            if question:
                print(await safe_answer(session, http, tools, history, question))
                return
            while True:
                try:
                    q = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() in {"exit", "quit", ""}:
                    break
                print("documind>", await safe_answer(session, http, tools, history, q), "\n")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or None))
