#!/usr/bin/env python3
"""
Titan Memory + LangChain Integration
=====================================

Three integration patterns:

1. BaseMemory subclass (legacy ConversationChain)
   - load_memory_variables() retrieves context before each LLM call
   - save_context() stores the turn after each LLM call

2. Modern LCEL pipe
   - RunnableLambda injects memory, another stores the response

3. AgentExecutor with Titan tools
   - LLM calls memory_store/memory_recall as tools in a ReAct loop

Requirements:
    pip install langchain langchain-anthropic titan-memory

Environment:
    ANTHROPIC_API_KEY must be set.
"""

from typing import Any, Dict, List

from langchain_core.memory import BaseMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.agents import create_tool_calling_agent, AgentExecutor

from titan import Titan, TitanConfig


# ═════════════════════════════════════════════════════════════════════════════
# TitanMemory: BaseMemory subclass
# ═════════════════════════════════════════════════════════════════════════════

class TitanMemory(BaseMemory):
    """
    LangChain memory backed by Titan.

    Exposes a single memory variable (default: 'titan_context') that contains
    the formatted [MEMORY] block from Titan. The prompt template must include
    a variable with this name.
    """

    titan: Any = None  # Titan instance (Any to avoid pydantic validation issues)
    memory_key: str = "titan_context"
    input_key: str = "input"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, data_dir: str = "./lc_memory", **kwargs):
        super().__init__(**kwargs)
        self.titan = Titan(TitanConfig(data_dir=data_dir))

    @property
    def memory_variables(self) -> List[str]:
        """Variables this memory exposes to the prompt template."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Called BEFORE the LLM. Retrieves relevant context from Titan."""
        query = inputs.get(self.input_key, "")
        context = self.titan.get_context_string(query) if query else ""
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Called AFTER the LLM. Stores the conversation turn in Titan."""
        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get("response", outputs.get("output", ""))
        if user_input:
            self.titan.ingest(f"{self.human_prefix}: {user_input}", origin="user")
        if ai_output:
            self.titan.ingest(f"{self.ai_prefix}: {ai_output}", origin="observation")

    def clear(self) -> None:
        """Run maintenance (Titan doesn't do hard clears -- it decays and prunes)."""
        self.titan.run_maintenance()


# ═════════════════════════════════════════════════════════════════════════════
# Option A: Legacy ConversationChain
# ═════════════════════════════════════════════════════════════════════════════

def demo_conversation_chain():
    """BaseMemory subclass with ConversationChain."""
    print("=== Option A: ConversationChain ===\n")

    memory = TitanMemory(data_dir="./lc_memory")
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    # Prompt must include the memory_key variable ('titan_context')
    template = """You are a helpful assistant with long-term memory.

{titan_context}

Current conversation:
Human: {input}
AI:"""
    prompt = PromptTemplate(
        input_variables=["input", "titan_context"],
        template=template
    )
    chain = ConversationChain(llm=llm, memory=memory, prompt=prompt)

    result = chain.invoke({"input": "My dog's name is Luna and she's a golden retriever."})
    print(f"Response: {result['response']}\n")

    result = chain.invoke({"input": "What's my dog's name?"})
    print(f"Response: {result['response']}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Option B: Modern LCEL pipe
# ═════════════════════════════════════════════════════════════════════════════

def demo_lcel():
    """LCEL pipe with RunnableLambda for memory injection."""
    print("=== Option B: LCEL Pipe ===\n")

    titan = Titan(TitanConfig(data_dir="./lc_memory"))
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    def inject_memory(input_dict):
        query = input_dict["input"]
        context = titan.get_context_string(query)
        return {**input_dict, "memory": context}

    def store_and_return(response):
        titan.ingest(response.content, origin="observation")
        return response

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant.\n\n{memory}"),
        ("human", "{input}")
    ])

    chain = (
        RunnableLambda(inject_memory)
        | prompt
        | llm
        | RunnableLambda(store_and_return)
    )

    result = chain.invoke({"input": "My dog's name is Luna"})
    print(f"Response: {result.content}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Option C: AgentExecutor with Titan tools
# ═════════════════════════════════════════════════════════════════════════════

def demo_agent_executor():
    """AgentExecutor with Titan as memory_store/memory_recall tools."""
    print("=== Option C: AgentExecutor ===\n")

    titan = Titan(TitanConfig(data_dir="./lc_memory"))

    @tool
    def memory_store(text: str, origin: str = "user") -> str:
        """Store information in long-term memory."""
        result = titan.ingest(text, origin=origin)
        return f"Stored: {result['claims']} claims, {result['entities']} entities"

    @tool
    def memory_recall(query: str, limit: int = 5) -> str:
        """Search long-term memory for relevant information."""
        results = titan.retrieve(query, limit=limit)
        if not results:
            return "No memories found."
        return "\n".join(
            f"[{r.get('confidence', '?')}] {r.get('text', r.get('content', ''))}"
            for r in results
        )

    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with long-term memory tools."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [memory_store, memory_recall], prompt)
    executor = AgentExecutor(agent=agent, tools=[memory_store, memory_recall])

    result = executor.invoke({"input": "Remember: my server runs on port 8101"})
    print(f"Response: {result['output']}\n")

    result = executor.invoke({"input": "What port does my server run on?"})
    print(f"Response: {result['output']}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Run whichever demo you prefer:
    demo_conversation_chain()
    # demo_lcel()
    # demo_agent_executor()


if __name__ == "__main__":
    main()
