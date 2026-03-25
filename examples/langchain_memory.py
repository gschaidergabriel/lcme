#!/usr/bin/env python3
"""
Titan Memory + LangChain + Local Ollama Model
===============================================

Custom BaseMemory subclass that backs LangChain's memory with Titan.
Uses a local model via Ollama.

Requirements:
    pip install langchain langchain-community titan-memory
    # Ollama must be running: ollama pull qwen2.5:3b
"""

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


def main():
    memory = TitanMemory(data_dir="./lc_memory")
    llm = Ollama(model="qwen2.5:3b")

    template = """You are a helpful assistant with long-term memory.

{titan_context}

Current conversation:
Human: {input}
AI:"""

    prompt = PromptTemplate(input_variables=["input", "titan_context"], template=template)
    chain = ConversationChain(llm=llm, memory=memory, prompt=prompt)

    print("LangChain + Titan + Ollama (type 'quit' to exit)")
    print("-" * 50)
    while True:
        try:
            user_input = input("\nHuman: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() == "quit":
            break
        result = chain.invoke({"input": user_input})
        print(f"AI: {result['response']}")


if __name__ == "__main__":
    main()
