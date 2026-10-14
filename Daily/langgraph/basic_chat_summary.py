from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")

class TypeState(TypedDict):
        prompt:str
        response :str
        summary:str

def node1(state:TypeState)-> TypeState:
    user_input = state["prompt"]
    response = llm.invoke(f"what user is asking just replay in 5 workds {user_input}")
    return {"response": response}

def node2(state:TypeState)-> TypeState:
     response_of_node1 = state["response"]
     summary = llm.invoke(f"can you summerize the outupt P{response_of_node1}")
     return {"summary":summary}
 
buildernode = StateGraph(TypeState)

buildernode.add_node("node1", node1)
buildernode.add_node("node2", node2)
buildernode.add_edge(START, "node1")
buildernode.add_edge("node1", "node2")
buildernode.add_edge("node2", END)

grpah = buildernode.compile()

print(grpah.get_graph().draw_ascii())


response = grpah.invoke({"prompt": "what is the capital of USA"})

print(response)