#!/usr/bin/env python3
"""
Titan Memory + Claude API (Anthropic SDK) Integration
=====================================================

Full working example: Claude with Titan memory via tool_use.

The agent loop sends messages + tool definitions to client.messages.create().
Claude responds with text blocks and/or tool_use blocks. You execute tools
locally, then send tool_result blocks back. Loop until Claude responds with
stop_reason="end_turn" (no more tool calls).

Memory injection happens in two places:
  (A) System prompt augmentation -- inject [MEMORY] block before each turn
  (B) Tool-based -- Claude calls memory_store/memory_recall/memory_forget

Requirements:
    pip install anthropic titan-memory

Environment:
    ANTHROPIC_API_KEY must be set.
"""

import anthropic
from titan import Titan, TitanConfig


# ── Initialize ────────────────────────────────────────────────────────────────

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
memory = Titan(TitanConfig(data_dir="./claude_memory"))


# ── Define memory tools ──────────────────────────────────────────────────────

TITAN_TOOLS = [
    {
        "name": "memory_store",
        "description": (
            "Store important information in long-term memory. Use this "
            "when the user shares facts about themselves, preferences, "
            "project details, decisions, or anything worth remembering "
            "across conversations. Include full context, not fragments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The information to remember. Be specific and self-contained."
                },
                "origin": {
                    "type": "string",
                    "enum": ["user", "inference", "observation"],
                    "description": "Source: 'user' for stated facts, 'inference' for deductions, 'observation' for context."
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "memory_recall",
        "description": (
            "Search long-term memory for relevant information. Use this "
            "when you need to remember something the user told you before, "
            "check if you know something, or find context for the current "
            "conversation. Returns timestamped memory entries with confidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for. Be descriptive - use natural language, not keywords."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_forget",
        "description": (
            "Remove a specific memory entry. Use when the user explicitly "
            "asks you to forget something, or when information is confirmed "
            "to be wrong. Requires the node_id from a previous recall."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The ID of the memory node to forget."
                }
            },
            "required": ["node_id"]
        }
    }
]


# ── Tool executor ────────────────────────────────────────────────────────────

def execute_tool(name: str, input_data: dict) -> str:
    """Execute a Titan tool and return the result as a string."""
    if name == "memory_store":
        result = memory.ingest(
            input_data["text"],
            origin=input_data.get("origin", "user")
        )
        return (
            f"Stored. Event {result['event_id']}: "
            f"{result['claims']} claims, entities: {result['entities']}"
        )

    elif name == "memory_recall":
        results = memory.retrieve(
            input_data["query"],
            limit=input_data.get("limit", 5)
        )
        if not results:
            return "No relevant memories found."
        lines = []
        for r in results:
            lines.append(
                f"[{r.get('timestamp', '?')}] (conf={r.get('confidence', '?')}, "
                f"id={r.get('node_id', '?')}) {r.get('text', r.get('content', ''))}"
            )
        return "\n".join(lines)

    elif name == "memory_forget":
        ok = memory.forget(input_data["node_id"])
        return f"{'Forgotten' if ok else 'Failed (protected or not found)'}."

    return f"Unknown tool: {name}"


# ── Agent loop ───────────────────────────────────────────────────────────────

def chat(user_message: str, conversation: list = None) -> str:
    """
    Send a message and handle the full tool-use loop.
    Returns Claude's final text response.
    """
    if conversation is None:
        conversation = []

    # (A) Passive memory: inject context into system prompt
    memory_context = memory.get_context_string(user_message)
    system_prompt = (
        "You are a helpful assistant with long-term memory.\n"
        "You have access to tools for storing and recalling information.\n"
        "Proactively use memory_store when the user shares important facts.\n"
        "Use memory_recall when you need to check what you know.\n\n"
        f"{memory_context}"
    )

    conversation.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TITAN_TOOLS,
            messages=conversation
        )

        # Append the full assistant response to conversation history
        # (must include tool_use blocks for context continuity)
        conversation.append({
            "role": "assistant",
            "content": response.content  # List of TextBlock / ToolUseBlock
        })

        # If Claude is done (no tool calls), extract text and return
        if response.stop_reason == "end_turn":
            text_parts = [
                block.text for block in response.content
                if block.type == "text"
            ]
            return "\n".join(text_parts)

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_str = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # MUST match the tool_use block id
                    "content": result_str
                })

        # Send tool results back to Claude
        conversation.append({
            "role": "user",
            "content": tool_results
        })
        # Loop continues -- Claude will process results and maybe call more tools


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    conversation = []

    # First message: Claude should store this in memory
    print("User: My name is Alice and I'm working on a robotics project in Berlin.")
    reply = chat(
        "My name is Alice and I'm working on a robotics project in Berlin.",
        conversation
    )
    print(f"Claude: {reply}\n")

    # Second message: Claude should recall from memory
    print("User: What project am I working on?")
    reply = chat("What project am I working on?", conversation)
    print(f"Claude: {reply}\n")

    # Third message: test forgetting
    print("User: What do you remember about me?")
    reply = chat("What do you remember about me?", conversation)
    print(f"Claude: {reply}\n")


if __name__ == "__main__":
    main()
