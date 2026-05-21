import json
import sys
from pathlib import Path
from typing import Generator

import ollama

sys.path.insert(0, str(Path(__file__).parent))
from tools import graph_search, grep, search_messages

MODEL = "llama3.2"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_messages",
            "description": (
                "Semantically search the user's message history (iMessage, WhatsApp, Telegram). "
                "Use for questions about past conversations, people, plans, events, "
                "or anything the user might have discussed in messages. "
                "Results are merged and ranked across all indexed sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language)",
                    },
                    "after": {
                        "type": "string",
                        "description": "Only include messages after this date. ISO (2024-01-15) or relative ('last 7 days')",
                    },
                    "before": {
                        "type": "string",
                        "description": "Only include messages before this date.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": (
                "Search message history (iMessage, WhatsApp, Telegram) using a knowledge graph "
                "with Personalized PageRank. Best for multi-hop queries — finding connections "
                "between people, places, and events across all indexed sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query containing names, places, or topics to seed the graph search",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for exact text patterns in local files using grep. "
                "Use for literal keyword searches — specific names, phrases, or identifiers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default: repo root)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful personal assistant that can search the user's message history.

Indexed sources: iMessage, WhatsApp, Telegram (any others that have been ingested).

You have three search tools:
- search_messages: semantic/vector search across all message sources (good for topics, themes, events)
- graph_search: knowledge-graph search using Personalized PageRank (good for connections between people, places, recurring topics)
- grep: exact text search over local files (good for specific names, exact phrases, keywords)

Results include a [source] tag (e.g. [iMessage] or [WhatsApp Chat with Alice]) so you can attribute answers correctly.
Always use at least one tool before answering a question that requires looking up information.
Include source and timestamps in your answer when relevant. Be concise and direct."""


def _run_tool(name: str, args: dict) -> str:
    clean = {k: v for k, v in args.items() if v is not None}
    if name == "search_messages":
        return search_messages(**clean)
    if name == "graph_search":
        return graph_search(**clean)
    if name == "grep":
        return grep(**clean)
    return f"Unknown tool: {name}"


def ensure_model() -> None:
    try:
        ollama.show(MODEL)
        print(f"Model '{MODEL}' is ready.")
    except Exception:
        print(f"Pulling model '{MODEL}'... (this may take a few minutes)")
        ollama.pull(MODEL)
        print(f"Model '{MODEL}' ready.")


def agent_stream(
    query: str, after: str = None, before: str = None
) -> Generator[dict, None, None]:
    """
    ReAct agent loop. Yields SSE-style event dicts:
      {"type": "tool_call",   "tool": str, "args": dict}
      {"type": "tool_result", "tool": str, "result": str}
      {"type": "token",       "content": str}
      {"type": "error",       "content": str}
    """
    user_content = query
    if after or before:
        parts = []
        if after:
            parts.append(f"after: {after}")
        if before:
            parts.append(f"before: {before}")
        user_content += f"\n\n[Date filter: {', '.join(parts)}]"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Tool-calling loop (non-streaming so we can parse tool calls cleanly)
    for _ in range(5):
        try:
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        except Exception as e:
            yield {"type": "error", "content": f"LLM error: {e}"}
            return

        msg = response["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            yield {"type": "tool_call", "tool": name, "args": args}

            result = _run_tool(name, args)

            yield {"type": "tool_result", "tool": name, "result": result[:600]}

            messages.append({"role": "tool", "content": result, "name": name})

    # Stream final answer
    try:
        stream = ollama.chat(model=MODEL, messages=messages, stream=True)
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield {"type": "token", "content": token}
    except Exception as e:
        yield {"type": "error", "content": f"Streaming error: {e}"}
