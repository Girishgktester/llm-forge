from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt , Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")


class MyState(TypedDict):
    question: str
    answer: str
    decision: str

def approval(state: MyState):
    result = interrupt("Approve or reject")
    return {"decision": result}

def answer(state: MyState):
    response = llm.invoke(state["question"])  
    return {"answer": response.content}

def check_decision(state: MyState):
    if state["decision"].strip().casefold() == "approve":
        return "Approve"
    else:
        return "Reject"

builder = StateGraph(MyState)
builder.add_node("approval_node", approval)
builder.add_node("answer_node", answer)
builder.add_edge(START, "approval_node")
builder.add_conditional_edges(
    "approval_node", check_decision, 
    {"Approve": "answer_node",
     "Reject": END
    }
)
builder.add_edge("answer_node", END)

graph = builder.compile(checkpointer=MemorySaver())


config = {"configurable":{"thread_id":"1"}}

while True:
    data = input("Enter your query: ")
    if data == "exit":
        break
    
    response = graph.invoke({"question": data}, config)
    print(response)

    state = graph.get_state(config)

    if state.next:
        print(state)
        decision = input("Approve or reject: ")
        graph.invoke(Command(resume=decision.strip()), config)
        state = graph.get_state(config)
        print(state.values)
        if "answer" in state.values:
            print(state.values["answer"])
        else:
            print("Request rejected; no answer generated.")
    
    print()
    print(graph.get_graph().draw_ascii())
