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
    feedback: str
    decision: str


def generate_testcases(state: QAState):
    if state["feedback"]:
        prompt = f"""
        Origianl requiremnt : {state['requirement']}
        based on this below testcases are generated 
        testcases : {state['test_case']}
        and the user has given the feedback 
        you need to take those suggestion and give best suitable result
        feedback : {state["feedback"]}
        """
    else:
        prompt = f"""
        Generate 2 simple testcases for the user flow
        requirement : {state["requirement"]}
        """
    response = llm.invoke(prompt)
    return {"test_case": response.content}


def human_approval(state: QAState):
    userfeedback = interrupt("Enter approve/edit/reject and feedback")

    return {
        "decision": userfeedback["decision"],
        "feedback": userfeedback.get("feedback", ""),
        "approval": userfeedback["decision"] == "approve"
    }


def route(state: QAState):
    if state["decision"] == "approve":
        return "approved"
    elif state["decision"] == "edit":
        return "edit"
    else:
        return "rejected"


builder = StateGraph(QAState)
builder.add_node("generate_testcases", generate_testcases)
builder.add_node("human_approval", human_approval)

builder.add_edge(START, "generate_testcases")
builder.add_edge("generate_testcases", "human_approval")
builder.add_conditional_edges(
    "human_approval",
    route,
    {"approved": END, "rejected": END, "edit": "generate_testcases"},
)

finalgraph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": 
    {
        "thread_id": 1
    }
}

result = finalgraph.invoke({
         "requirement": "Logout",
         "feedback":"",
         "approval" : False,
         "decision": "",
         "test_case": ""
    } , config=config
)

print(result)

state = finalgraph.get_state(config)

while state.next:
    print(state.values)
    decision = input("Enter approve, edit, or reject: ").strip().lower()
    while decision not in {"approve", "edit", "reject"}:
        decision = input("Please enter approve, edit, or reject: ").strip().lower()

    feedback = ""
    if decision == "edit":
        feedback = input("Enter feedback for the edit: ")

    result = finalgraph.invoke(
        Command(resume={"decision": decision, "feedback": feedback}),
        config=config,
    )
    state = finalgraph.get_state(config)

print(result)