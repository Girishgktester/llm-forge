from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()


class GraphState(TypedDict):
    """Basic state for the LangGraph workflow."""
    messages: list
    result: str


def chat_node(state: GraphState) -> GraphState:
    """Call the LLM to generate a response."""
    llm = ChatOpenAI(model="gpt-6-luna")
    response = llm.invoke(state["messages"])
    return {**state, "result": response.content}


# Build the graph
workflow = StateGraph(GraphState)
workflow.add_node("chat", chat_node)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)

app = workflow.compile()
