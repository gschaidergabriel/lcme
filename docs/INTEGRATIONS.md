# Titan Memory -- Integration Guide

How to add Titan to your local AI agent. All examples use **3B-8B models running on consumer hardware** via Ollama, llama.cpp, or compatible local inference servers.

## Table of Contents

- [Universal Pattern](#universal-pattern)
- [Ollama + Python](#ollama--python)
- [llama.cpp / llama-server (OpenAI-compatible)](#llama-cpp--llama-server)
- [LangChain + Local Model](#langchain--local-model)
- [LlamaIndex + Local Model](#llamaindex--local-model)
- [Raw Python (No Framework)](#raw-python-no-framework)

---

## Universal Pattern

There are two ways to wire Titan into any agent. Both work with any model, any framework.

### Pattern A: System Prompt Injection (Recommended for Small Models)

Before each LLM call, retrieve relevant memories and prepend them to the system message. The model sees the context without needing tool-calling support. This is the most reliable approach for 3B-8B models where tool calling can be unreliable.

```python
from titan import Titan, TitanConfig

memory = Titan(TitanConfig(data_dir="./memory"))

# Before each LLM call:
context = memory.get_context_string(user_message)
system_prompt = f"You are a helpful assistant.\n\n{context}"
# Pass system_prompt + user_message to your model
```

After the model responds, ingest both sides:

```python
memory.ingest(f"User: {user_message}", origin="user")
memory.ingest(f"Assistant: {response}", origin="observation")
```

### Pattern B: Tool-Based (If Your Model Supports It)

Give the model `memory_store` / `memory_recall` tools. Models like Qwen2.5-7B, Llama-3.1-8B, and Mistral-7B support tool calling, but reliability varies. Test with your specific model before relying on this pattern.

```python
tools = [
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

def execute_tool(name, args):
    if name == "memory_store":
        result = memory.ingest(args["text"], origin="user")
        return f"Stored: {result['claims']} claims"
    elif name == "memory_recall":
        return memory.get_context_string(args["query"])
```

> [!TIP]
> For 3B models, Pattern A (system prompt injection) is more reliable than Pattern B (tool calling). Small models often hallucinate tool calls or format them incorrectly.

---

## Ollama + Python

The most common local LLM setup. Works with any model Ollama supports.

**Install:**
```bash
pip install ollama titan-memory
```

**Models tested:** `qwen2.5:3b`, `llama3.1:8b`, `phi3:3.8b`, `gemma2:2b`, `mistral:7b`

### System Prompt Injection (Recommended)

Full example: [`examples/ollama_chat.py`](../examples/ollama_chat.py)

```python
import re
import ollama
from titan import Titan, TitanConfig

memory = Titan(TitanConfig(data_dir="./agent_memory"))

def chat(user_message: str, history: list = None) -> str:
    if history is None:
        history = []

    # Retrieve relevant memories for this query
    memory_block = memory.get_context_string(user_message)

    system = (
        "You are a helpful assistant with long-term memory.\n\n"
        f"{memory_block}\n\n"
        "If the user tells you something worth remembering, include "
        "[STORE: <what to remember>] in your response."
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = ollama.chat(model="qwen2.5:3b", messages=messages)
    reply = response["message"]["content"]

    # Parse [STORE: ...] directives
    for text in re.findall(r'\[STORE:\s*(.+?)\]', reply):
        memory.ingest(text, origin="inference")

    # Store the conversation turn
    memory.ingest(f"User: {user_message}", origin="user")

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    return reply

# Usage
history = []
print(chat("My name is Alex and I work at a startup in Berlin.", history))
print(chat("What do you know about me?", history))
```

### Tool-Based (Qwen2.5, Llama3.1, Mistral)

Only use this if your model reliably handles tool calling.

```python
import ollama
from titan import Titan, TitanConfig

memory = Titan(TitanConfig(data_dir="./agent_memory"))

tools = [
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

def chat(user_message: str) -> str:
    context = memory.get_context_string(user_message)
    messages = [
        {"role": "system", "content": f"You have long-term memory.\n\n{context}"},
        {"role": "user", "content": user_message},
    ]

    while True:
        response = ollama.chat(model="qwen2.5:7b", messages=messages, tools=tools)
        msg = response["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            memory.ingest(f"User: {user_message}", origin="user")
            return msg["content"]

        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            if name == "memory_store":
                r = memory.ingest(args["text"], origin="user")
                messages.append({"role": "tool", "content": f"Stored: {r['claims']} claims"})
            elif name == "memory_recall":
                messages.append({"role": "tool", "content": memory.get_context_string(args["query"])})
```

**Notes:**
- Context window varies by model (4K-32K). Memory context eats into it. Keep `max_context_length` in TitanConfig appropriate for your model.
- `ollama.chat()` is synchronous. Use `ollama.AsyncClient` for async.
- Tool calling reliability: Qwen2.5 > Llama3.1 > Mistral > Phi-3 > Gemma.

---

## llama.cpp / llama-server

If you run models directly via llama-server (llama.cpp's built-in HTTP server), use the OpenAI-compatible API.

**Install:**
```bash
pip install openai titan-memory
```

**Start your model:**
```bash
llama-server -m ./models/qwen2.5-3b-instruct-q4_k_m.gguf -c 4096 --port 8080
```

Full example: [`examples/openai_compatible.py`](../examples/openai_compatible.py)

```python
import json
from openai import OpenAI
from titan import Titan, TitanConfig

memory = Titan(TitanConfig(data_dir="./agent_memory"))

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

def chat(user_message: str, history: list = None) -> str:
    if history is None:
        history = []

    context = memory.get_context_string(user_message)
    system = {"role": "system", "content": f"You are a helpful assistant.\n\n{context}"}

    history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="qwen2.5-3b",  # Model name from llama-server
        messages=[system] + history,
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})

    # Store conversation
    memory.ingest(f"User: {user_message}", origin="user")

    return reply
```

**Also works with:** LocalAI, vLLM (local mode), LM Studio, koboldcpp, text-generation-webui -- anything that serves an OpenAI-compatible `/v1/chat/completions` endpoint.

---

## LangChain + Local Model

LangChain with a local model via Ollama or llama.cpp.

**Install:**
```bash
pip install langchain langchain-community titan-memory
```

### Custom Memory Class

```python
from typing import Any, Dict, List
from langchain_core.memory import BaseMemory
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from titan import Titan, TitanConfig

class TitanMemory(BaseMemory):
    """LangChain memory backed by Titan."""

    titan: Any = None
    memory_key: str = "titan_context"
    input_key: str = "input"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, data_dir: str = "./lc_memory", **kwargs):
        super().__init__(**kwargs)
        self.titan = Titan(TitanConfig(data_dir=data_dir))

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        query = inputs.get(self.input_key, "")
        context = self.titan.get_context_string(query) if query else ""
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get("response", outputs.get("output", ""))
        if user_input:
            self.titan.ingest(f"User: {user_input}", origin="user")
        if ai_output:
            self.titan.ingest(f"Assistant: {ai_output}", origin="observation")

    def clear(self) -> None:
        self.titan.run_maintenance()


# Usage with local Ollama model
memory = TitanMemory(data_dir="./lc_memory")
llm = Ollama(model="qwen2.5:3b")

template = """You are a helpful assistant with long-term memory.

{titan_context}
Current conversation:
Human: {input}
AI:"""

prompt = PromptTemplate(input_variables=["input", "titan_context"], template=template)
chain = ConversationChain(llm=llm, memory=memory, prompt=prompt)

result = chain.invoke({"input": "My name is Alex, I work on robotics."})
print(result["response"])
result = chain.invoke({"input": "What do you know about me?"})
print(result["response"])
```

**Also works with:** `langchain_community.llms.LlamaCpp` for direct llama.cpp integration, or any `langchain_community.chat_models` that wraps a local endpoint.

---

## LlamaIndex + Local Model

```bash
pip install llama-index llama-index-llms-ollama titan-memory
```

```python
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from titan import Titan, TitanConfig

titan = Titan(TitanConfig(data_dir="./li_memory"))

def memory_store(text: str) -> str:
    """Store information in long-term memory."""
    result = titan.ingest(text, origin="user")
    return f"Stored: {result['claims']} claims"

def memory_recall(query: str) -> str:
    """Search long-term memory for relevant information."""
    return titan.get_context_string(query)

tools = [
    FunctionTool.from_defaults(fn=memory_store),
    FunctionTool.from_defaults(fn=memory_recall),
]

llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

response = agent.chat("Remember that I prefer Python for backend work.")
print(response)
response = agent.chat("What programming language do I like?")
print(response)
```

---

## Raw Python (No Framework)

The simplest integration. No framework, no dependencies beyond `requests`.

```python
import requests
from titan import Titan, TitanConfig

memory = Titan(TitanConfig(data_dir="./memory"))

LLAMA_URL = "http://localhost:8080/v1/chat/completions"  # llama-server

def chat(user_message: str, history: list = None) -> str:
    if history is None:
        history = []

    # Get relevant memories
    context = memory.get_context_string(user_message)

    messages = [
        {"role": "system", "content": f"You are a helpful assistant.\n\n{context}"},
        *history,
        {"role": "user", "content": user_message},
    ]

    response = requests.post(LLAMA_URL, json={
        "model": "qwen2.5-3b",
        "messages": messages,
    }).json()

    reply = response["choices"][0]["message"]["content"]

    # Store conversation
    memory.ingest(f"User: {user_message}", origin="user")

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    return reply

# Run
history = []
print(chat("I'm building a weather app in Rust.", history))
print(chat("What project am I working on?", history))
```

---

## Summary

| Integration | Model | Install | Best Pattern |
|-------------|-------|---------|-------------|
| **Ollama** | Any Ollama model | `pip install ollama` | System prompt injection |
| **llama-server** | Any GGUF | `pip install openai` | System prompt injection |
| **LangChain** | Ollama / LlamaCpp | `pip install langchain langchain-community` | TitanMemory class |
| **LlamaIndex** | Ollama | `pip install llama-index llama-index-llms-ollama` | FunctionTool |
| **Raw Python** | Any HTTP endpoint | `pip install requests` | System prompt injection |

All examples use models in the 3B-8B range. For 3B models, always prefer system prompt injection over tool calling — small models are unreliable at tool-call formatting.
