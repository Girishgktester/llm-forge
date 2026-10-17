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
    automation_code: str
    validation_passed: bool
    validation_errors: str

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


def automated_code_generation(state: QAState):
    prompt = f"""
    Generate the simple playwright automation code for the testcases
    "test_case" : {state['test_case']}
    
    """
    response = llm.invoke(prompt)
    return{"automation_code":response.content}
    
def validate_automation_code(state: QAState):
    code = state["automation_code"]
    errors = []

    if not code.strip():
        errors.append("Generated code is empty")

    if "test(" not in code and "test." not in code:
        errors.append("Missing Playwright test declaration")

    if "expect(" not in code:
        errors.append("Missing Playwright assertion")

    return {
        "validation_passed": len(errors) == 0,
        "validation_errors": "\n".join(errors)
    }
    

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
builder.add_node("automated_code_generation", automated_code_generation)
builder.add_node("validate_automation_code", validate_automation_code)
builder.add_node("human_approval", human_approval)

builder.add_edge(START, "generate_testcases")
builder.add_edge("generate_testcases", "human_approval")

builder.add_conditional_edges(
    "human_approval",
    route,
    {"approved": "automated_code_generation", "rejected": END, "edit": "generate_testcases"},
)
builder.add_edge("automated_code_generation", "validate_automation_code")

builder.add_edge("validate_automation_code", END)
graph = builder.compile()


def pretty_print_result(result):
    """Format the graph result into a readable report."""
    print("\n" + "=" * 60)
    print("         QA AUTOMATION REPORT")
    print("=" * 60)

    print(f"\n[Requirement]      {result['requirement']}")
    print(f"[Feedback Loop]     {'Yes' if result.get('feedback') else 'No'}")

    print("\n" + "-" * 60)
    print("  TEST CASES")
    print("-" * 60)
    print(result.get("test_case", "N/A"))

    if result.get("automation_code"):
        print("\n" + "-" * 60)
        print("  PLAYWRIGHT AUTOMATION CODE")
        print("-" * 60)
        print(result["automation_code"])

        print(f"\n[Validation Passed]  {'Yes' if result.get('validation_passed') else 'No'}")
        if result.get("validation_errors"):
            print(f"[Errors]             {result['validation_errors']}")
    else:
        print(f"\n[Automation Code]   Not generated (rejected)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "qa-auto-001"}}

    initial_state = {
        "requirement": "Logout",
        "feedback": "",
        "approval": False,
        "decision": "",
        "test_case": "",
        "automation_code": "",
        "validation_passed": False,
        "validation_errors": "",
    }

    result = graph.invoke(initial_state, config=config)

    # Get the snapshot to check for pending nodes
    snap = graph.get_state(config)

    while snap.next:
        print(snap.values)
        decision = input("Enter approve, edit, or reject: ").strip().lower()
        while decision not in {"approve", "edit", "reject"}:
            decision = input("Please enter approve, edit, or reject: ").strip().lower()

        feedback = ""
        if decision == "edit":
            feedback = input("Enter feedback for the edit: ")

        result = graph.invoke(
            Command(resume={"decision": decision, "feedback": feedback}),
            config=config,
        )
        snap = graph.get_state(config)

    pretty_print_result(result)