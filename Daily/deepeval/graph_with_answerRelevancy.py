from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt , Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")

class MyState(TypedDict):
    requirement: str
    testcases: str
    decision: str
    relevancy_score: float
    relevancy_reason: str
    relevancy_pass: bool

def generate_testcases(state: MyState):
    prompt = f"""
    Generate 5 manual test cases for this requirement:
    {state["requirement"]}

    Return positive and negative cases.
    Use this format for every test case:
    ID:
    Title:
    Steps:
    Expected Result:
    """
    response = llm.invoke(prompt)
    return {"testcases": response.content}

def check_decision(state: MyState):
    if state["decision"].strip().casefold() == "approve":
        return "Approve"
    else:
        return "Reject"

def evaluate_testcases(state: MyState):

    test_case = LLMTestCase(
        input=state["requirement"],
        actual_output=state["testcases"]
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-6-luna",
        include_reason=True
    )

    metric.measure(test_case)

    return {
        "relevancy_score": metric.score,
        "relevancy_reason": metric.reason,
        "relevancy_pass": metric.is_successful()
    }

def human_approval(state: MyState):
    result = interrupt("Approve or reject the generated test cases")
    return {"decision": result}

builder = StateGraph(MyState)
builder.add_node("generate_testcases", generate_testcases)
builder.add_node("evaluate_testcases", evaluate_testcases)
builder.add_node("human_approval", human_approval)

builder.add_edge(START, "generate_testcases")
builder.add_edge("generate_testcases", "evaluate_testcases")
builder.add_edge("evaluate_testcases", "human_approval")
builder.add_conditional_edges(
    "human_approval",
    check_decision,
    {
        "Approve": END,
        "Reject": END
    }
)


graph = builder.compile(checkpointer=MemorySaver())


config = {"configurable": {"thread_id": "1"}}

requirement = input("Enter the requirement: ").strip()

result = graph.invoke({
    "requirement": requirement,
    "testcases": "",
    "relevancy_score": 0.0,
    "relevancy_reason": "",
    "relevancy_pass": False,
    "decision": ""
}, config)

state = graph.get_state(config)

print("\n--- Generated Test Cases ---")
print(state.values.get("testcases"))

print("\n--- Relevancy Evaluation ---")
print("Relevancy Score:", state.values.get("relevancy_score"))
print("Reason:", state.values.get("relevancy_reason"))
print("Passed:", state.values.get("relevancy_pass"))

if state.next:
    decision = input("\nApprove or reject: ").strip()
    result = graph.invoke(Command(resume=decision), config)
    state = graph.get_state(config)

    if state.values["decision"].strip().casefold() == "approve":
        print("\nTest cases approved.")
    else:
        print("\nTest cases rejected.")
