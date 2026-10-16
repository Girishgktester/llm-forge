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
    approval: bool
    


def generate_testcases(state: QAState):
    prompt = f"""
    Generate 2 simple testcases fof the user requiremntr {state["requirement"]}
    """
    response = llm.invoke(prompt)
    return {"test_case": response.content}


def human_approval(state: QAState):
    decision = interrupt("Enter your status")
    return {"approval": decision == "approve"}


def decsion(state: QAState):
    if state["approval"] == True:
        return "approval"
    else:
        return "reject"


grpahbuilder = StateGraph(QAState)
grpahbuilder.add_node("generate_testcases", generate_testcases)
grpahbuilder.add_node("human_approval", human_approval)

grpahbuilder.add_edge(START, "generate_testcases")
grpahbuilder.add_edge("generate_testcases", "human_approval")

grpahbuilder.add_conditional_edges(
    "human_approval", decsion, {"approval": END, "reject": END}
)

finalgrpah = grpahbuilder.compile(checkpointer=MemorySaver())

config = {
    "configurable": 
    {
       "thread_id": 1
       }
    }

result = finalgrpah.invoke({
    "requirement": "Generate login testcases",
    "test_case": "",
    "approval" : False
    },config = config
    )
print(result)

state = finalgrpah.get_state(config)

if state.next:
    userdecision = input("Approve or reject: ").strip().lower()

    result = finalgrpah.invoke(
    Command(resume=userdecision),
    config=config
    )
    state = finalgrpah.get_state(config)
    print(state.values)
else:
    print("No pending interrupt found.")

print(result)



