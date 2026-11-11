from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage 
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys_message = SystemMessage(content="You are an helpfull assistant that can perform basic arthrmeiotc oprions like addition substartion ")

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def sub_numbers(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b

llm = ChatOpenAI(model="gpt-6-luna").bind_tools([add_numbers, sub_numbers])

def assistant(state:MessagesState):
   return {"messages": llm.invoke([sys_message] + state["messages"])}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([add_numbers, sub_numbers]))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

graph = builder.compile()
    
response = graph.invoke({"messages": "what is 2 + 3?"})
print("Answer = ",response['messages'][-1].content)