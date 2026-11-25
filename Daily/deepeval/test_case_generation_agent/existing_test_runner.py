import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from playwright.sync_api import sync_playwright
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import evaluate
from playwright.sync_api import sync_playwright
import subprocess
import glob

load_dotenv()


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
    automated_output_file: str
    app_url: str
    analysis: str
    test_cases: str
    manual_attempts: int
    automation_attempts: int
    feedback: str
    decision: str
    automation_decision: str
    automation_feedback: str

    relevancy_score: float | None
    relevancy_reason: str | None
    relevancy_pass: bool | None
    faithfulness_score: float | None
    faithfulness_reason: str | None
    faithfulness_pass: bool | None
    page_snapshot: str
    test_code: str
    run_result: str
    passed: bool


def docuemnt_loader(state: TestState):
    with open(state["input_file"], "r", encoding="utf-8") as file:
        document_text = file.read()
    return {"user_story": document_text}


def requirement_Analyzer(state: TestState):
    return {"analysis": state["user_story"]}


def testcase_generator(state: TestState):
    with open(state["output_file"], "r", encoding="utf-8") as file:
        test_cases_json = file.read()

    manual_attempts = state.get("manual_attempts", 0) + 1
    feedback = state.get("feedback", "").strip()
    print(f"\nRead static test cases (attempt {manual_attempts}/3):\n")
    print(test_cases_json)
    if feedback:
        print(f"Feedback received but not applied to static test cases: {feedback}")

    return {
        "test_cases": test_cases_json,
        "manual_attempts": manual_attempts,
    }


def validate_testcases(state: TestState):
    # testcase = LLMTestCase(
    #     input=state["user_story"],
    #     actual_output=state["test_cases"],
    #     retrieval_context=[state["analysis"]],
    # )

    # evaluation_result = evaluate(
    #     test_cases=[testcase],
    #     metrics=[
    #         AnswerRelevancyMetric(
    #             threshold=0.7,
    #             model="gpt-6-luna",
    #             include_reason=True,
    #         ),
    #         FaithfulnessMetric(
    #             threshold=0.7,
    #             model="gpt-6-luna",
    #             include_reason=True,
    #         ),
    #     ],
    # )

    # metrics = evaluation_result.test_results[0].metrics_data
    # relevancy_metric = metrics[0]
    # faithfulness_metric = metrics[1]

    # return {
    #     "relevancy_score": relevancy_metric.score,
    #     "relevancy_reason": relevancy_metric.reason,
    #     "relevancy_pass": relevancy_metric.success,
    #     "faithfulness_score": faithfulness_metric.score,
    #     "faithfulness_reason": faithfulness_metric.reason,
    #     "faithfulness_pass": faithfulness_metric.success,
    # }
    
    pass

def human_review(state: TestState):
    userfeedback = interrupt({
        "type": "manual_testcase_review",
        "message": "Review generated manual test cases",
        "test_cases": state["test_cases"]
    })

    return {
        "decision": userfeedback["decision"],
        "feedback": userfeedback.get("feedback", "")
    }

def route(state: TestState):
    if state["decision"] == "approve":
        return "approved"
    if state["decision"] == "reject":
        return "rejected"
    if state["decision"] == "edit":
        if state.get("manual_attempts", 0) >= 3:
            return "rejected"
        return "edit"
    



def save_testcases(state: TestState):
    output_file = state["output_file"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(state["test_cases"])
    print(f"Approved test cases saved to: {output_file}")
    return {"output_file": output_file}


def read_page(state: TestState):
    print(f"Opening {state['app_url']} to read the real page elements")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(state["app_url"])
        snapshot = (
            f"Page title: {page.title()}\n" + page.locator("body").aria_snapshot()
        )
        browser.close()
    return {"page_snapshot": snapshot}


def clean_code(text: str):
    for fence in ["```typescript", "```ts", "```"]:
        text = text.replace(fence, "")
    return text.strip()


def generate_automated_tests(state: TestState):
    with open(state["automated_output_file"], "r", encoding="utf-8") as file:
        test_code = file.read()

    print(f"Read existing Playwright test from {state['automated_output_file']}")
    return {
        "test_code": test_code,
        "automation_attempts": 1,
    }


def generate_autopmated_tests(state: TestState):
    feedback = state.get("automation_feedback", "").strip()
    if feedback:
        print("Regenerating automation using human feedback")
        with open(state["automated_output_file"], "r", encoding="utf-8") as file:
                test_code = file.read()
    else:
        print("Generating automation for the first time")
        with open(state["automated_output_file"], "r", encoding="utf-8") as file:
               test_code = file.read()
    return {
        "test_code": test_code,
        "automation_attempts": state.get("automation_attempts", 0) + 1,
        "automation_feedback": ""
    }


def save_test_automated_tests(state: TestState):
    output_file = state["automated_output_file"]
    print(f"Saving test to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(state["test_code"])
    return {}

def automation_review(state: TestState):

    userfeedback = interrupt({
        "type": "automation_code_review",
        "message": "Review generated Playwright code",
        "test_code": state["test_code"]
    })

    return {
        "automation_decision": userfeedback["decision"],
        "automation_feedback": userfeedback.get("feedback", "")
    }
    
def route_automation(state: TestState):
    decision = state.get("automation_decision") or state.get("decision")

    if decision == "approve":
        return "approved"

    if decision == "reject":
        return "rejected"

    if decision == "edit":
        if state.get("automation_attempts", 0) >= 3:
            return "rejected"
        return "edit"

def run_test(state: TestState):
    print(f"Running the generated test (attempt {state['automation_attempts']})...")
    npx_command = "npx.cmd" if os.name == "nt" else "npx"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    test_file = os.path.relpath(state["automated_output_file"], project_root).replace(os.sep, "/")
    result = subprocess.run(
        [
            npx_command,
            "playwright",
            "test",
            test_file,
            "--headed",
            "--reporter=list",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    passed = result.returncode == 0
    run_result = result.stdout + result.stderr
    print("Test PASSED" if passed else "Test FAILED")
    if not passed:
        print(f"Playwright output:\n{run_result}")

    test_results_dir = os.path.join(project_root, "test-results")
    for context_file in glob.glob(
        os.path.join(test_results_dir, "**", "error-context.md"),
        recursive=True,
    ):
        with open(context_file, encoding="utf-8") as file:
            run_result += "\n\nPage at the moment of failure:\n" + file.read()

    return {"run_result": run_result, "passed": passed}


def fix_test(state: TestState):
    with open(state["automated_output_file"], "r", encoding="utf-8") as file:
        test_code = file.read()

    automation_attempts = state["automation_attempts"] + 1
    print(
        f"Reloaded existing Playwright test after failure "
        f"(attempt {automation_attempts}/3)"
    )
    return {
        "test_code": test_code,
        "automation_attempts": automation_attempts,
    }


def route_test_result(state: TestState):
    if state["passed"] or state["automation_attempts"] >= 3:
        return "finished"
    return "fix"


builder = StateGraph(TestState)
builder.add_node("docuemnt_loader", docuemnt_loader)
builder.add_node("requirement_Analyzer", requirement_Analyzer)
builder.add_node("testcase_generator", testcase_generator)
builder.add_node("validate_testcases", validate_testcases)
builder.add_node("human_review", human_review)

builder.add_node("save_testcases", save_testcases)
builder.add_node("read_page", read_page)
builder.add_node("generate_automated_tests", generate_automated_tests)
builder.add_node("save_test_automated_tests", save_test_automated_tests)
builder.add_node("automation_review", automation_review)

builder.add_node("run_test", run_test)
builder.add_node("fix_test", fix_test)

builder.add_edge(START, "docuemnt_loader")
builder.add_edge("docuemnt_loader", "requirement_Analyzer")
builder.add_edge("requirement_Analyzer", "testcase_generator")
builder.add_edge("testcase_generator", "validate_testcases")

builder.add_conditional_edges(
    "human_review",
    route,
    {"approved": "save_testcases", "rejected": END, "edit": "testcase_generator"},
)

builder.add_edge("validate_testcases", "human_review")
builder.add_edge("save_testcases", "read_page")
builder.add_edge("read_page", "generate_automated_tests")
builder.add_edge("generate_automated_tests","automation_review"
)
builder.add_conditional_edges("automation_review",
                        route_automation,
                        {
                        "approved": "save_test_automated_tests", 
                        "rejected": END, 
                        "edit": "generate_automated_tests", 
                        })
builder.add_edge("save_test_automated_tests", "run_test")


builder.add_conditional_edges(
    "run_test",
    route_test_result,
    {
      "finished": END, 
     "fix": "fix_test"},
    )
builder.add_edge("fix_test", "save_test_automated_tests")


input_file = os.path.join(os.path.dirname(__file__), "testcases", "TC001.txt")
output_file = os.path.join(os.path.dirname(__file__), "output", "test_cases.json")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
automated_output_file = os.path.join(
    project_root, "tests", "generated_automated", "test_login.spec.ts"
)
app_url = "https://practicetestautomation.com/practice-test-login/"

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "qa-auto-001"}}
result = graph.invoke(
    {
        "input_file": input_file,
        "output_file": output_file,
        "automated_output_file": automated_output_file,
        "app_url": app_url,
    },
    config=config,
)

print(graph.get_graph().draw_ascii())

snap = graph.get_state(config)

while snap.next:
    interrupts = [
        intr
        for task in snap.tasks
        for intr in task.interrupts
    ]

    if not interrupts:
        break

    interrupt_data = interrupts[0].value
    print("interrupt_data[0]", interrupt_data)

    review_type = interrupt_data["type"]
    if review_type == "manual_testcase_review":
        print("\nGenerated Manual Test Cases:")
        print(interrupt_data["test_cases"])
    elif review_type == "automation_code_review":
        print("\nGenerated Automation Code:")
        print(interrupt_data["test_code"])

    decision = input("\nEnter approve, edit, or reject: ").strip().lower()
    while decision not in {"approve", "edit", "reject"}:
        decision = input("Enter approve, edit, or reject: ").strip().lower()

    feedback = ""
    if decision == "edit":
        feedback = input("Enter your feedback... ").strip()

    result = graph.invoke(
        Command(resume={
            "decision": decision,
            "feedback": feedback,
        }),
        config=config,
    )
    snap = graph.get_state(config)