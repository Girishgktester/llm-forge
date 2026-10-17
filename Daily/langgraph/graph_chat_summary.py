from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")

class TypeState(TypedDict):
    prompt: str
    response: str
    summary: str


def node1(state: TypeState) -> dict:
    user_input = state["prompt"]
    response = llm.invoke(f"what user is asking just reply in 5 words {user_input}")
    return {"response": response.content}


def node2(state: TypeState) -> dict:
    response_of_node1 = state["response"]
    summary = llm.invoke(f"can you summarize the output: {response_of_node1}")
    return {"summary": summary.content}


builder = StateGraph(TypeState)

builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

graph = builder.compile()

if __name__ == "__main__":
    print(graph.get_graph().draw_ascii())
    response = graph.invoke({"prompt": "what is the capital of USA"})
    print(response)