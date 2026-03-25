#!/usr/bin/env python3
"""
LCME Memory + LangChain + Local Ollama Model
===============================================

Custom BaseMemory subclass that backs LangChain's memory with LCME.
Uses a local model via Ollama.

Requirements:
    pip install langchain langchain-community lcme
    # Ollama must be running: ollama pull qwen2.5:3b
"""

from typing import Any, Dict, List
from langchain_core.memory import BaseMemory
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from lcme import LCME, LCMEConfig


class LCMEMemory(BaseMemory):
    """LangChain memory backed by LCME."""

    lcme: Any = None
    memory_key: str = "lcme_context"
    input_key: str = "input"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, data_dir: str = "./lc_memory", **kwargs):
        super().__init__(**kwargs)
        self.lcme = LCME(LCMEConfig(data_dir=data_dir))

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        query = inputs.get(self.input_key, "")
        context = self.lcme.get_context_string(query) if query else ""
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get("response", outputs.get("output", ""))
        if user_input:
            self.lcme.ingest(f"User: {user_input}", origin="user")
        if ai_output:
            self.lcme.ingest(f"Assistant: {ai_output}", origin="observation")

    def clear(self) -> None:
        self.lcme.run_maintenance()


def main():
    memory = LCMEMemory(data_dir="./lc_memory")
    llm = Ollama(model="qwen2.5:3b")

    template = """You are a helpful assistant with long-term memory.

{lcme_context}

Current conversation:
Human: {input}
AI:"""

    prompt = PromptTemplate(input_variables=["input", "lcme_context"], template=template)
    chain = ConversationChain(llm=llm, memory=memory, prompt=prompt)

    print("LangChain + LCME + Ollama (type 'quit' to exit)")
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
