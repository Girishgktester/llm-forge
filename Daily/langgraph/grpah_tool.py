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

response = model.invoke("What is 2 + 3?")  
print(response.content)