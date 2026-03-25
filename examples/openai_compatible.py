#!/usr/bin/env python3
"""
Titan Memory + llama-server / Local OpenAI-Compatible Endpoint
===============================================================

Works with any local server that exposes /v1/chat/completions:
  - llama-server (llama.cpp)
  - LocalAI
  - vLLM (local mode)
  - LM Studio
  - koboldcpp
  - text-generation-webui (with openai extension)

Start your model:
    llama-server -m ./qwen2.5-3b-instruct-q4_k_m.gguf -c 4096 --port 8080

Then run this script:
    pip install openai titan-memory
    python openai_compatible.py
"""

import json
import sys
from openai import OpenAI
from titan import Titan, TitanConfig


# ── Configuration ────────────────────────────────────────────────────────

# Point at your local llama-server / vLLM / LocalAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)
MODEL = "qwen2.5-3b"  # Model name from your server

titan = Titan(TitanConfig(data_dir="./agent_memory"))


# ── Memory tool definitions ──────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "memory_store",
        "description": "Store information in long-term memory",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "What to remember"}
        }, "required": ["text"]}
    }},
    {"type": "function", "function": {
        "name": "memory_recall",
        "description": "Search long-term memory",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to search for"}
        }, "required": ["query"]}
    }}
]


def execute_tool(tool_call) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    if name == "memory_store":
        r = titan.ingest(args["text"], origin="user")
        return f"Stored: {r['claims']} claims, {r['entities']} entities"
    elif name == "memory_recall":
        return titan.get_context_string(args["query"])
    return f"Unknown tool: {name}"


# ── Chat function ────────────────────────────────────────────────────────

def chat(user_message: str, history: list) -> str:
    # Inject memory context into system prompt
    context = titan.get_context_string(user_message)
    system = {"role": "system", "content": f"You are a helpful assistant.\n\n{context}"}

    history.append({"role": "user", "content": user_message})

    # Try with tools first, fall back to plain chat if server doesn't support them
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[system] + history, tools=TOOLS, tool_choice="auto",
        )
    except Exception:
        response = client.chat.completions.create(
            model=MODEL, messages=[system] + history,
        )

    msg = response.choices[0].message

    # Handle tool calls
    if msg.tool_calls:
        history.append(msg.model_dump())
        for tc in msg.tool_calls:
            result = execute_tool(tc)
            history.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        # Get final response after tool execution
        response = client.chat.completions.create(
            model=MODEL, messages=[system] + history,
        )
        msg = response.choices[0].message

    reply = msg.content or ""
    history.append({"role": "assistant", "content": reply})

    # Store conversation
    titan.ingest(f"User: {user_message}", origin="user")

    return reply


# ── Interactive loop ─────────────────────────────────────────────────────

def main():
    print(f"Titan Memory + {MODEL} (type 'quit' to exit, 'stats' for memory stats)")
    print("-" * 60)
    history = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            print(titan.get_stats())
            continue

        reply = chat(user_input, history)
        print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    main()
