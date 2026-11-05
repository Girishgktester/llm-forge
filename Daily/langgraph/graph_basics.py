from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

llm = ChatOpenAI(model="gpt-6-luna")


def node_1(state: MessagesState):
    # Extract last user message content
    last_msg = state["messages"][-1].content

    response = llm.invoke(
        f"User said: {last_msg}. Respond with output, no explanation."
    )

    return {"messages": state["messages"] + [AIMessage(content=response.content)]}


def node_2(state: MessagesState):
    response = state["messages"][-1].content

    summary = llm.invoke(
        f"Summarize this output: {response}"
    )

    return {"messages": state["messages"] + [AIMessage(content=summary.content)]}


builder = StateGraph(MessagesState)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

graph = builder.compile()


if __name__ == "__main__":
    messages = [HumanMessage(content="President of India?")]
    result = graph.invoke({"messages": messages})
    print("\n=== Conversation History ===")
    for msg in result["messages"]:
        print(f"[{type(msg).__name__}] {msg.content}")
