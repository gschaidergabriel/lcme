#!/usr/bin/env python3
"""
LCME Memory + Ollama Integration
==================================

Ollama with LCME memory via system prompt augmentation.

This is the simplest and most reliable pattern. Before each LLM call,
relevant memories are retrieved and injected into the system message.
The model sees a [STORE: ...] directive pattern to signal when it wants
to save new information.

Works with any Ollama model -- no tool-calling support required.

Requirements:
    pip install ollama lcme

Environment:
    Ollama must be running locally (default: http://localhost:11434).
    Pull a model first: ollama pull llama3.1:8b
"""

import re
import ollama
from lcme import LCME, LCMEConfig


# ── Initialize ────────────────────────────────────────────────────────────────

lcme = LCME(LCMEConfig(data_dir="./ollama_memory"))


# ── Chat with system prompt augmentation ─────────────────────────────────────

def chat_with_memory(
    user_message: str,
    messages: list = None,
    model: str = "llama3.1:8b"
) -> str:
    """
    Send a message to Ollama with LCME memory context injected.

    Memory is injected as a [MEMORY] block in the system prompt.
    The model can request storage by including [STORE: <text>] in its response.

    Args:
        user_message: The user's input.
        messages: Conversation history (mutated in place). Pass the same list
                  across calls to maintain conversation context.
        model: Ollama model name. Any model works (no tool support needed).

    Returns:
        The assistant's response text.
    """
    if messages is None:
        messages = []

    # Retrieve relevant context from LCME
    memory_block = lcme.get_context_string(user_message)

    system_msg = (
        "You are a helpful assistant with long-term memory.\n\n"
        f"{memory_block}\n\n"
        "If the user tells you something worth remembering, include "
        "'[STORE: <what to remember>]' in your response. You can include "
        "multiple [STORE: ...] directives if there are several facts."
    )

    # Build full message list: system + history + new user message
    full_messages = [{"role": "system", "content": system_msg}]
    full_messages.extend(messages)
    full_messages.append({"role": "user", "content": user_message})

    # Call Ollama
    response = ollama.chat(model=model, messages=full_messages)
    reply = response["message"]["content"]

    # Parse [STORE: ...] directives from response and ingest them
    stores = re.findall(r'\[STORE:\s*(.+?)\]', reply)
    for text in stores:
        result = lcme.ingest(text, origin="inference")
        print(f"  [LCME] Stored: {text[:60]}... ({result['claims']} claims)")

    # Always store the user's message
    lcme.ingest(f"User said: {user_message}", origin="user")

    # Update conversation history
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": reply})

    return reply


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    conversation = []

    print("LCME Memory + Ollama Chat")
    print("Type 'quit' to exit, 'stats' for memory stats, 'consolidate' to train.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            stats = lcme.get_stats()
            print(f"  [LCME] Nodes: {stats['nodes']}, Edges: {stats['edges']}, "
                  f"Vectors: {stats['vectors']}")
            continue
        if user_input.lower() == "consolidate":
            result = lcme.consolidate()
            print(f"  [LCME] Consolidation: {result}")
            continue

        reply = chat_with_memory(user_input, conversation)
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
