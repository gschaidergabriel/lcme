#!/usr/bin/env python3
"""
Titan Memory + OpenAI-Compatible API Integration
==================================================

Generic pattern for any OpenAI-compatible endpoint: vLLM, LocalAI,
LiteLLM, Groq, Together, or OpenAI itself.

Uses the OpenAI Python SDK with a custom base_url. Tool calls are
returned in response.choices[0].message.tool_calls. You execute tools
locally and send results back as tool-role messages.

Memory is injected in two ways:
  (A) System prompt augmentation: [MEMORY] block injected before each turn
  (B) Tool-based: model calls memory_store / memory_recall

Requirements:
    pip install openai titan-memory

Environment:
    For local servers (vLLM, LocalAI): no API key needed.
    For cloud APIs: set the appropriate API key env var.
"""

import json
from openai import OpenAI
from titan import Titan, TitanConfig


# ── Initialize ────────────────────────────────────────────────────────────────

titan = Titan(TitanConfig(data_dir="./oai_memory"))

# Point at any OpenAI-compatible endpoint.
# Uncomment the one you need:

# vLLM / LocalAI (local):
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local",  # Many local servers ignore this
)
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Whatever is served

# OpenAI (cloud):
# client = OpenAI()  # Uses OPENAI_API_KEY
# MODEL = "gpt-4o"

# Groq (cloud):
# client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
# MODEL = "llama-3.1-70b-versatile"

# Together (cloud):
# client = OpenAI(base_url="https://api.together.xyz/v1", api_key=TOGETHER_KEY)
# MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


# ── Define memory tools ──────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": "Store information in long-term memory for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "What to remember. Be specific and self-contained."
                    },
                    "origin": {
                        "type": "string",
                        "enum": ["user", "inference", "observation"],
                        "description": "Source type: 'user' for stated facts, 'inference' for deductions."
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_recall",
            "description": "Search long-term memory for relevant stored information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for. Use natural language."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ── Tool executor ────────────────────────────────────────────────────────────

def execute_tool_call(tool_call) -> str:
    """Execute a tool call and return the result as a string."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "memory_store":
        result = titan.ingest(args["text"], origin=args.get("origin", "user"))
        return f"Stored: {result['claims']} claims, {result['entities']} entities"

    elif name == "memory_recall":
        results = titan.retrieve(args["query"], limit=args.get("limit", 5))
        if not results:
            return "No memories found."
        return "\n".join(
            f"[conf={r.get('confidence', '?')}] {r.get('text', r.get('content', ''))}"
            for r in results
        )

    return f"Unknown tool: {name}"


# ── Chat function ────────────────────────────────────────────────────────────

def chat(user_message: str, messages: list = None) -> str:
    """
    Send a message and handle the full tool-call loop.

    Args:
        user_message: The user's input.
        messages: Conversation history (mutated in place). Pass the same list
                  across calls to maintain context.

    Returns:
        The assistant's final text response.
    """
    if messages is None:
        messages = []

    # (A) Passive memory: inject context into system prompt
    memory_context = titan.get_context_string(user_message)
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant with long-term memory.\n"
            "Use memory_store to save important information.\n"
            "Use memory_recall to search for things you should know.\n\n"
            f"{memory_context}"
        )
    }

    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[system] + messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # If no tool calls, we're done
        if not msg.tool_calls:
            # Store the user's turn in memory
            titan.ingest(f"User: {user_message}", origin="user")
            return msg.content

        # Process tool calls
        for tc in msg.tool_calls:
            result = execute_tool_call(tc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # Loop continues -- model processes results and may call more tools


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    conversation = []

    print("Titan Memory + OpenAI-Compatible API")
    print("Type 'quit' to exit, 'stats' for memory stats.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            stats = titan.get_stats()
            print(f"  [Titan] Nodes: {stats['nodes']}, Edges: {stats['edges']}, "
                  f"Vectors: {stats['vectors']}")
            continue

        reply = chat(user_input, conversation)
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
