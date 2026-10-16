from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")


class QAState(TypedDict):
    requirement: str
    test_case: str
    approved: bool


def generate_testcases(state: QAState):
    prompt = f""""
    Generate 2 manual testcases for the requirement
    keep it simple and crisp
    dont add anything extra
    cover possitive and negative testcases
    {state["requirement"]}
        
    Return the testcases in this format 
    TC Number , 
    \nACTUAL 
    \nEXPCTED 
    table fromat

    """
    response = llm.invoke(prompt)
    return {"test_case": response.content}


def human_approval(state: QAState):
    decision = interrupt("Enter user approval or reject status")
    return {"approved": decision == "approve"}

def decision_router(state: QAState):
    if state["approved"] == True:
        return "approved"
    else:
        return "Reject"


grpah = StateGraph(QAState)
grpah.add_node("generate_testcases", generate_testcases)
grpah.add_node("human_approval", human_approval)

grpah.add_edge(START, "generate_testcases")
grpah.add_edge("generate_testcases", "human_approval")

grpah.add_conditional_edges(
    "human_approval", 
    decision_router, 
    {
     "approved" : END,
     "Reject" :END
    }
)

finalGrpah = grpah.compile(checkpointer= MemorySaver())
print(finalGrpah.get_graph().draw_ascii())

config = {
    "configurable": {
        "thread_id" :1
    }
}

result = finalGrpah.invoke(
    {
        "requirement": "Login functionality",
        "test_case": "",
        "approved": False
    },
    config=config
)

print(result)

state = finalGrpah.get_state(config)


if state.next:
    userdecision = input("Approve or reject: ").strip().lower()

    result = finalGrpah.invoke(
    Command(resume=userdecision),
    config=config
    )
    state = finalGrpah.get_state(config)
    print(state.values)
else:
    print("No pending interrupt found.")

print(result)