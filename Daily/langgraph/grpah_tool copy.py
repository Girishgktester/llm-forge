from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import AnyMessage, SystemMessage , HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
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


class State(TypedDict):
      question:str
      tool_name:str
      tool_args:dict
      tools_result:str
      answer:str
      
      
def assistant(state:State):
    response = llm.invoke([sys_message, HumanMessage(content=state["question"])])

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        
        return {
            "tool_name": tool_call["name"],
            "tool_args": tool_call["args"]
        }
    return {
        "answer":response.content
    }
    
def excute_tool(state:State):
   if state['tool_name'] == 'add_numbers':
       result = add_numbers.invoke(state['tool_args'])
       return {
        "tools_result":str(result),
        "answer":str(result),
    }
   if state['tool_name'] == 'sub_numbers':
       result = sub_numbers.invoke(state['tool_args'])
       return {
        "tools_result":str(result),
        "answer":str(result),
    }
       
       
builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", excute_tool)
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    lambda state: "tools" if state.get("tool_name") else END,
)
builder.add_edge("tools", END)

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
    
response = graph.invoke({"question": "what is 2 + 3?"})

print("Answer = ",response['answer'])