import os
import glob
import re
import subprocess
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric
from deepeval import evaluate

load_dotenv()

llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-6-luna"))

class TestCase(BaseModel):
    test_case_id: str
    title: str
    test_data: Dict[str, str]
    steps: List[str]
    expected_result: str
    actual_result: str = ""

class TestCases(BaseModel):
    test_cases: List[TestCase]

class TestState(TypedDict, total=False):
    input_file: str
    user_story: str
    output_file: str
    app_url: str

    analysis: str
    test_cases: str
    relevancy_score: float | None
    relevancy_reason: str | None
    relevancy_pass: bool | None
    faithfulness_score: float | None
    faithfulness_reason: str | None
    faithfulness_pass: bool | None
    page_snapshot: str
    test_code: str
    run_results: str
    passed: bool
    attempts: int
    commit: bool
    push: bool
    approval: bool
    feedback: str
    decision: str

def docuemnt_loader(state: TestState):
    with open(state["input_file"], "r", encoding="utf-8") as file:
        document_text = file.read()
    return {"user_story": document_text}

def requirement_Analyzer(state: TestState):
    prompt = f"""
    Analyze the following user story or requirement document.
    Identify:
    - The main user goal and expected behavior
    - Missing or unclear acceptance criteria
    - Important boundary and edge cases
    Do not invent requirements. Clearly label any assumptions.
    Document:
    {state['user_story']}
    """
    response = llm.invoke(prompt)
    return {"analysis": response.content}

def testcase_generator(state: TestState):
    feedback = state.get("feedback", "").strip()
    if feedback:
        generation_instruction = f"""
        Regenerate the test cases based on this reviewer feedback:
        {feedback}
        Follow the requested changes, including adding scenarios when requested.
        """
    else:
        generation_instruction = """
        Generate exactly ONE test case for the highest-priority scenario.
        """

    prompt = f"""
    Generate manual test cases from the requirement analysis below.
    {generation_instruction}
    Do not invent requirements. Leave Actual Result blank for the tester.

    Return valid JSON in exactly this structure:
    {
      "test_cases": [
        {
          "test_case_id": "TC001",
          "title": "Verify successful login",
          "test_data": {
            "username": "valid_user",
            "password": "valid_password"
          },
          "steps": ["Navigate to login page", "Enter valid username"],
          "expected_result": "User should successfully log in",
          "actual_result": ""
        }
      ]
    }

    Requirement analysis:
    {state['analysis']}

    Reviewer feedback:
    {feedback}
    """
    response = llm.with_structured_output(TestCases).invoke(prompt)
    test_cases_json = response.model_dump_json(indent=2)
    attempts = state.get("attempts", 0) + 1
    print(f"\nGenerated test cases (attempt {attempts}/3):\n")
    print(test_cases_json)
    return {
        "test_cases": test_cases_json,
        "attempts": attempts,
    }

def validate_testcases(state: TestState):
    testcase = LLMTestCase(
        input=state["user_story"],
        actual_output=state["test_cases"],
        retrieval_context=[state["analysis"]],
    )

    evaluation_result = evaluate(
        test_cases=[testcase],
        metrics=[
            AnswerRelevancyMetric(
                threshold=0.7,
                model="gpt-6-luna",
                include_reason=True,
            ),
            FaithfulnessMetric(
                threshold=0.7,
                model="gpt-6-luna",
                include_reason=True,
            ),
        ],
    )

    metrics = evaluation_result.test_results[0].metrics_data
    relevancy_metric = metrics[0]
    faithfulness_metric = metrics[1]

    return {
        "relevancy_score": relevancy_metric.score,
        "relevancy_reason": relevancy_metric.reason,
        "relevancy_pass": relevancy_metric.success,
        "faithfulness_score": faithfulness_metric.score,
        "faithfulness_reason": faithfulness_metric.reason,
        "faithfulness_pass": faithfulness_metric.success,
    }

def human_review(state: TestState):
    userfeedback = interrupt("Enter approve/edit/reject and feedback")
    return {
            "decision": userfeedback["decision"],
            "feedback": userfeedback.get("feedback", ""),
            "approval": userfeedback["decision"] == "approve"
        }
    
    
def route(state: TestState):

    if state["decision"] == "approve":
        return "approved"

    if state["decision"] == "reject":
        return "rejected"

    if state["decision"] == "edit":
        if state.get("attempts", 0) >= 3:
            return "rejected"

        return "edit"


def save_testcases(state: TestState):
    output_file = state["output_file"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(state["test_cases"])
    print(f"Approved test cases saved to: {output_file}")
    return {"output_file": output_file}


builder = StateGraph(TestState)
builder.add_node("docuemnt_loader", docuemnt_loader)
builder.add_node("requirement_Analyzer", requirement_Analyzer)
builder.add_node("testcase_generator", testcase_generator)
builder.add_node("validate_testcases", validate_testcases)
builder.add_node("human_review", human_review)
builder.add_node("save_testcases", save_testcases)

builder.add_edge(START, "docuemnt_loader")
builder.add_edge("docuemnt_loader", "requirement_Analyzer")
builder.add_edge("requirement_Analyzer", "testcase_generator")
builder.add_edge("testcase_generator", "validate_testcases")
builder.add_conditional_edges(
    "human_review",
    route,
    {"approved": "save_testcases", "rejected": END, "edit": "testcase_generator"})

builder.add_edge("validate_testcases", "human_review")
builder.add_edge("save_testcases", END)


input_file = os.path.join(os.path.dirname(__file__), "testcases", "TC001.txt")
output_file = os.path.join(os.path.dirname(__file__), "output", "test_cases.json")

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "qa-auto-001"}}
result = graph.invoke(
    {"input_file": input_file, "output_file": output_file},
    config=config,
)

print(graph.get_graph().draw_ascii())

snap = graph.get_state(config)

while snap.next:
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




