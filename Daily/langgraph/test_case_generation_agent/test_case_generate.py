import os
import glob
import re
import subprocess

from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric , FaithfulnessMetric

load_dotenv()

llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-6-luna"))

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
    page_snapshot: str
    test_code: str
    run_results: str
    passed: bool
    attempts: int
    commit: bool
    push: bool

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
    print("Generate one manual test case based on the user requirements")
    prompt = f"""
    Generate exactly ONE manual test case from the requirement analysis below.
    Do not generate multiple test cases, alternatives, or additional test case IDs.
    Choose the single highest-priority scenario for the main user goal. Include
    boundary or negative behavior only if it fits within that one scenario.
    Do not invent requirements. Leave Actual Result blank for the tester
    to fill in after executing the test.

    Use this format:
    Test Case ID:
    Title:
    Test Data:
    Steps:
    Expected Result:
    Actual Result:

    Requirement analysis:
    {state['analysis']}
    """
    response = llm.invoke(prompt)
    return {"test_cases": response.content}

def validate_testcases(state: TestState):
    testcase = LLMTestCase(
        input = state['user_story'],
        actual_output = state['test_cases'],
        retrieval_context=[state['analysis']]
    )
    answerRelevancyMetric = AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-6-luna",
            verbose_mode=True,
            include_reason=True
    )
    
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-6-luna",
    )

    faithfulness.measure(testcase)
    faithfulness.measure(testcase)

    print("Relevancy:", faithfulness.score)
    print("Faithfulness:", faithfulness.score)
    
    answerRelevancyMetric.measure(testcase)
    return {
            "relevancy_score": answerRelevancyMetric.score,
            "relevancy_reason": answerRelevancyMetric.reason,
            "relevancy_pass": answerRelevancyMetric.is_successful()
        }
    
    
builder = StateGraph(TestState)
builder.add_node("docuemnt_loader", docuemnt_loader)
builder.add_node("requirement_Analyzer", requirement_Analyzer)
builder.add_node("testcase_generator", testcase_generator)
builder.add_node("validate_testcases", validate_testcases)

builder.add_edge(START, "docuemnt_loader")
builder.add_edge("docuemnt_loader", "requirement_Analyzer")
builder.add_edge("requirement_Analyzer", "testcase_generator")
builder.add_edge("testcase_generator", "validate_testcases")
builder.add_edge("validate_testcases", END)

graph = builder.compile()
print(graph.get_graph().draw_ascii())

input_file = os.path.join(os.path.dirname(__file__), "testcases", "TC001.txt")

response = graph.invoke({"input_file": input_file})
print(response["test_cases"])


# def human_review(state: TestState):
#     pass
