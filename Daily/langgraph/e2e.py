
from dotenv import load_dotenv
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.test_case import SingleTurnParams as LLMTestCaseParams

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")

THRESHOLD = 0.75
MAX_RETRIES = 2


class TestState(TypedDict, total=False):
    feature: str
    description: str
    category: str

    manual_testcases: str
    ui_automation_code: str
    api_automation_code: str

    manual_feedback: str
    code_feedback: str

    manual_score: float
    manual_eval_reason: str
    code_score: float
    code_eval_reason: str

    manual_retries: int
    code_retries: int

    manual_approved: bool
    code_approved: bool


# -----------------------------
# 1. Analyze requirement
# -----------------------------
def analyze_requirement(state: TestState):
    prompt = f"""
    Classify the requirement as ui or api.

    ui: pages, buttons, forms, dashboards.
    api: backend endpoints, data handling, database operations.

    Feature: {state['feature']}
    Description: {state['description']}

    Return only ui or api.
    """
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if category not in ("ui", "api"):
        raise ValueError(f"Invalid category: {category}")

    return {"category": category}


# -----------------------------
# 2. Generate manual test cases
# -----------------------------
def generate_manual_testcases(state: TestState):
    prompt = f"""
    Generate 3 manual test cases for this requirement.

    Feature: {state['feature']}
    Description: {state['description']}
    Category: {state['category']}

    Evaluator feedback from previous attempt:
    {state.get('manual_feedback', 'None')}

    Address the feedback if provided.

    Format:
    TC1: Steps, Expected: Result
    TC2: Steps, Expected: Result
    TC3: Steps, Expected: Result
    """
    response = llm.invoke(prompt)

    return {
        "manual_testcases": response.content,
        "manual_retries": state.get("manual_retries", 0),
    }


# -----------------------------
# 3. DeepEval manual test validation
# -----------------------------
def validate_manual_testcases(state: TestState):
    metric = GEval(
        name="Manual Test Case Quality",
        criteria=(
            "Evaluate whether the generated manual test cases accurately "
            "cover the requirement. Check relevance, clear execution steps, "
            "testable expected results, and meaningful positive and negative "
            "coverage where applicable. Penalize invented requirements."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=THRESHOLD,
        model="gpt-4o-mini",
    )

    test_case = LLMTestCase(
        input=(
            f"Feature: {state['feature']}\n"
            f"Description: {state['description']}\n"
            f"Category: {state['category']}"
        ),
        actual_output=state["manual_testcases"],
    )

    metric.measure(test_case)
    score = float(metric.score or 0.0)
    reason = metric.reason or "No evaluation reason returned."

    retries = state.get("manual_retries", 0)

    if score < THRESHOLD and retries < MAX_RETRIES:
        return {
            "manual_score": score,
            "manual_eval_reason": reason,
            "manual_feedback": (
                f"DeepEval score: {score:.2f}. "
                f"Evaluator feedback: {reason}"
            ),
            "manual_retries": retries + 1,
        }

    return {
        "manual_score": score,
        "manual_eval_reason": reason,
    }


def route_manual_validation(state: TestState):
    if (
        state["manual_score"] < THRESHOLD
        and state.get("manual_retries", 0) < MAX_RETRIES
    ):
        return "testcases"

    return "approve_testcases"


# -----------------------------
# 4. Human approval: test cases
# -----------------------------
def approve_manual_testcases(state: TestState):
    decision = interrupt({
        "stage": "Manual test case approval",
        "testcases": state["manual_testcases"],
        "deepeval_score": state["manual_score"],
        "evaluator_feedback": state["manual_eval_reason"],
        "instruction": (
            "Resume with approved=True or approved=False "
            "and feedback='your comments'."
        ),
    })

    return {
        "manual_approved": decision["approved"],
        "manual_feedback": decision.get("feedback", ""),
    }


def route_manual_approval(state: TestState):
    if not state["manual_approved"]:
        return "testcases"

    return (
        "ui_automation"
        if state["category"] == "ui"
        else "api_automation"
    )


# -----------------------------
# 5. Generate automation code
# -----------------------------
def generate_ui_automation(state: TestState):
    prompt = f"""
    Generate Playwright TypeScript automation code.

    Feature: {state['feature']}
    Description: {state['description']}

    Approved manual test cases:
    {state['manual_testcases']}

    Evaluator and human feedback:
    {state.get('code_feedback', 'None')}

    Address the feedback. Include meaningful locators and assertions.
    Return code only.
    """
    response = llm.invoke(prompt)

    return {
        "ui_automation_code": response.content,
        "code_retries": state.get("code_retries", 0),
    }


def generate_api_automation(state: TestState):
    prompt = f"""
    Generate pytest API automation code.

    Feature: {state['feature']}
    Description: {state['description']}

    Approved manual test cases:
    {state['manual_testcases']}

    Evaluator and human feedback:
    {state.get('code_feedback', 'None')}

    Address the feedback. Include endpoint calls, status checks,
    and response validation. Return code only.
    """
    response = llm.invoke(prompt)

    return {
        "api_automation_code": response.content,
        "code_retries": state.get("code_retries", 0),
    }


# -----------------------------
# 6. DeepEval automation validation
# -----------------------------
def validate_automation_code(state: TestState):
    if state["category"] == "ui":
        code = state["ui_automation_code"]
        code_type = "Playwright TypeScript"
    else:
        code = state["api_automation_code"]
        code_type = "pytest API automation"

    metric = GEval(
        name="Automation Code Quality",
        criteria=(
            f"Evaluate the generated {code_type} code against the "
            "feature requirement and approved manual test cases. "
            "Check whether the code is relevant, contains meaningful "
            "assertions, validates expected behavior, and avoids "
            "unsupported assumptions. Penalize missing critical checks. "
            "Evaluate the code as text; do not claim it was executed."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=THRESHOLD,
        model="gpt-6-luna",
    )

    test_case = LLMTestCase(
        input=(
            f"Feature: {state['feature']}\n"
            f"Description: {state['description']}\n"
            f"Approved test cases: {state['manual_testcases']}\n"
            f"Code type: {code_type}"
        ),
        actual_output=code,
    )

    metric.measure(test_case)
    score = float(metric.score or 0.0)
    reason = metric.reason or "No evaluation reason returned."

    retries = state.get("code_retries", 0)

    if score < THRESHOLD and retries < MAX_RETRIES:
        return {
            "code_score": score,
            "code_eval_reason": reason,
            "code_feedback": (
                f"DeepEval score: {score:.2f}. "
                f"Evaluator feedback: {reason}"
            ),
            "code_retries": retries + 1,
        }

    return {
        "code_score": score,
        "code_eval_reason": reason,
    }


def route_code_validation(state: TestState):
    if (
        state["code_score"] < THRESHOLD
        and state.get("code_retries", 0) < MAX_RETRIES
    ):
        return (
            "ui_automation"
            if state["category"] == "ui"
            else "api_automation"
        )

    return (
        "approve_ui_code"
        if state["category"] == "ui"
        else "approve_api_code"
    )


# -----------------------------
# 7. Human approval: automation
# -----------------------------
def approve_ui_code(state: TestState):
    decision = interrupt({
        "stage": "Playwright code approval",
        "code": state["ui_automation_code"],
        "deepeval_score": state["code_score"],
        "evaluator_feedback": state["code_eval_reason"],
    })

    return {
        "code_approved": decision["approved"],
        "code_feedback": decision.get("feedback", ""),
    }


def approve_api_code(state: TestState):
    decision = interrupt({
        "stage": "pytest code approval",
        "code": state["api_automation_code"],
        "deepeval_score": state["code_score"],
        "evaluator_feedback": state["code_eval_reason"],
    })

    return {
        "code_approved": decision["approved"],
        "code_feedback": decision.get("feedback", ""),
    }


def route_ui_approval(state: TestState):
    return END if state["code_approved"] else "ui_automation"


def route_api_approval(state: TestState):
    return END if state["code_approved"] else "api_automation"


# -----------------------------
# 8. Build LangGraph
# -----------------------------
builder = StateGraph(TestState)

builder.add_node("analyze", analyze_requirement)

builder.add_node("testcases", generate_manual_testcases)
builder.add_node("validate_manual", validate_manual_testcases)
builder.add_node("approve_testcases", approve_manual_testcases)

builder.add_node("ui_automation", generate_ui_automation)
builder.add_node("api_automation", generate_api_automation)
builder.add_node("validate_code", validate_automation_code)

builder.add_node("approve_ui_code", approve_ui_code)
builder.add_node("approve_api_code", approve_api_code)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "testcases")
builder.add_edge("testcases", "validate_manual")

builder.add_conditional_edges(
    "validate_manual",
    route_manual_validation,
    {
        "testcases": "testcases",
        "approve_testcases": "approve_testcases",
    },
)

builder.add_conditional_edges(
    "approve_testcases",
    route_manual_approval,
    {
        "testcases": "testcases",
        "ui_automation": "ui_automation",
        "api_automation": "api_automation",
    },
)

builder.add_edge("ui_automation", "validate_code")
builder.add_edge("api_automation", "validate_code")

builder.add_conditional_edges(
    "validate_code",
    route_code_validation,
    {
        "ui_automation": "ui_automation",
        "api_automation": "api_automation",
        "approve_ui_code": "approve_ui_code",
        "approve_api_code": "approve_api_code",
    },
)

builder.add_conditional_edges(
    "approve_ui_code",
    route_ui_approval,
    {"ui_automation": "ui_automation", END: END},
)

builder.add_conditional_edges(
    "approve_api_code",
    route_api_approval,
    {"api_automation": "api_automation", END: END},
)

graph = builder.compile(checkpointer=MemorySaver())


# -----------------------------
# 9. Run and resume the graph
# -----------------------------

def pretty_print_result(result):
    """Format the final graph result into a readable report."""
    print("\n" + "=" * 60)
    print("           QA AUTOMATION REPORT")
    print("=" * 60)

    print(f"\n[Feature]      {result['feature']}")
    print(f"[Description]   {result['description']}")
    print(f"[Category]      {result['category'].upper()}")

    print("\n" + "-" * 60)
    print("  MANUAL TEST CASES")
    print("-" * 60)
    print(result.get("manual_testcases", "N/A"))

    print(f"\n[DeepEval Score]     {result.get('manual_score', 'N/A'):.2f}")
    print(f"[Evaluator Reason]   {result.get('manual_eval_reason', 'N/A')}")
    print(f"[Manual Retries]      {result.get('manual_retries', 0)}")
    print(f"[Human Approved]      {'Yes' if result.get('manual_approved') else 'No'}")

    if result.get("manual_approved"):
        if result["category"] == "ui":
            code = result.get("ui_automation_code", "N/A")
            code_type = "Playwright TypeScript"
        else:
            code = result.get("api_automation_code", "N/A")
            code_type = "pytest API"

        print(f"\n{'─' * 60}")
        print(f"  {code_type} AUTOMATION CODE")
        print(f"{'─' * 60}")
        if isinstance(code, str):
            print(f"```python\n{code}\n```")
        else:
            print(str(code))

        print(f"\n[DeepEval Score]    {result.get('code_score', 'N/A'):.2f}")
        print(f"[Evaluator Reason]  {result.get('code_eval_reason', 'N/A')}")
        print(f"[Code Retries]       {result.get('code_retries', 0)}")
        print(f"[Human Approved]     {'Yes' if result.get('code_approved') else 'No'}")

    print("\n" + "=" * 60)


def human_input_loop():
    """Handle interrupt prompts and return decision dict."""
    global result
    while "__interrupt__" in result:
        for item in result["__interrupt__"]:
            data = item.value

            print(f"\n--- {data['stage']} ---")

            if "testcases" in data:
                print("\nGenerated Test Cases:")
                print(data["testcases"])
            if "code" in data:
                print(f"\nGenerated Automation Code:")
                print(data["code"])

            if "deepeval_score" in data:
                print(f"\nDeepEval Score: {float(data['deepeval_score']):.2f}")
                print(f"Evaluator Feedback: {data['evaluator_feedback']}")

            approved = input("\nApprove? (y/n): ").strip().lower() == "y"

            feedback = ""
            if not approved:
                feedback = input("Enter revision feedback: ")

            decision = {"approved": approved, "feedback": feedback}

        result = graph.invoke(Command(resume=decision), config=config)



if __name__ == "__main__":
    # ----- UI Example -----
    config = {"configurable": {"thread_id": "qa-ui-001"}}

    initial_state = {
        "feature": "User Dashboard Profile Page",
        "description": (
            "A dashboard page where users can view their profile "
            "photo, username, email, and edit basic info."
        ),
        "manual_retries": 0,
        "code_retries": 0,
    }

    result = graph.invoke(initial_state, config=config)
    human_input_loop()

    pretty_print_result(result)
