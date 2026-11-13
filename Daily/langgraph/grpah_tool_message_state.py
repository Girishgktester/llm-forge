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
system_message = SystemMessage(
    content=(
        "You are a helpful assistant that can perform addition and "
        "subtraction. Use the available tools for arithmetic."
    )
)


def assistant(state: MessagesState) -> dict[str, list[AnyMessage]]:
    """Answer the user or request an arithmetic tool."""
    response = model.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
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
