from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

llm = ChatOpenAI(model="gpt-6-luna")

class State(TypedDict):
    prompt: str
    response: str
    summary: str

def node_1(state: State):
    user_input = state.get("prompt", "")

    response = llm.invoke(
        f"User said: {user_input}. Respond with output, no explanation."
    )
    return {"response": response.content or ""}

def routeAfterNode1(state: State):
    if state["response"].strip():
        return "has_result"
    else:
        return "no_result"

def node_2(state: State):
    response = state["response"]
    summary = llm.invoke(
        f"Summarize this output: {response}"
    )
    return {"summary": summary.content or ""}


builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
# builder.add_node("routeAfterNode1", routeAfterNode1)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", routeAfterNode1, {
    "has_result": "node_2",
    "no_result": END})
# builder.add_edge("node_1", "node_2")
# builder.add_edge("node_2", END)

graph = builder.compile()
print(graph.get_graph().draw_ascii())


response = graph.invoke({"prompt": "captitaL of France?"})

print("response:", response["response"])