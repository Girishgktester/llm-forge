from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


@tool
def sub_numbers(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b


tools = [add_numbers, sub_numbers]
model = ChatOpenAI(model="gpt-6-luna").bind_tools(tools)


def call_model(state: MessagesState) -> dict[str, list[AnyMessage]]:
    """Answer the user or request the addition tool for arithmetic."""
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. Answer questions clearly. "
            "When asked to add numbers, use the add_numbers tool. "
            "When asked to subtract numbers, use the sub_numbers tool."
        )
    )
    response = model.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "assistant")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [("user", "What is 2 + 3?")]}
    )
    print(result["messages"][-1].content)
