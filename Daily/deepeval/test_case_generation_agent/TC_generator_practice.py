import json
import os
from pathlib import Path
from typing import Dict, List, TypedDict

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel


load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "gpt-6-luna")
llm = ChatOpenAI(model=MODEL_NAME)


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
    analysis: str
    test_cases: str
    attempts: int
    feedback: str
    decision: str
    relevancy_score: float
    relevancy_reason: str
    relevancy_pass: bool
    faithfulness_score: float
    faithfulness_reason: str
    faithfulness_pass: bool


def load_document(state: TestState) -> dict:
    document = Path(state["input_file"]).read_text(encoding="utf-8")
    return {"user_story": document}


def analyze_requirements(state: TestState) -> dict:
    prompt = f"""
Analyze the following user story or requirement document.
Identify the main user goal, expected behavior, acceptance criteria, and important edge cases.
Do not invent requirements. Clearly label assumptions.

Document:
{state["user_story"]}
"""
    response = llm.invoke(prompt)
    return {"analysis": response.content}


def generate_test_cases(state: TestState) -> dict:
    feedback = state.get("feedback", "").strip()
    instruction = "Generate exactly one test case for the highest-priority scenario."
    if feedback:
        instruction = f"Regenerate the test case using this reviewer feedback:\n{feedback}"

    prompt = f"""
Generate manual test cases from the requirement analysis below.
{instruction}
Do not invent requirements. Leave actual_result blank.
Return valid data matching the requested schema.

Requirement analysis:
{state["analysis"]}
"""
    response = llm.with_structured_output(TestCases).invoke(prompt)
    test_cases_json = response.model_dump_json(indent=2)
    attempts = state.get("attempts", 0) + 1
    print(f"\nGenerated test cases (attempt {attempts}/3):\n{test_cases_json}")
    return {"test_cases": test_cases_json, "attempts": attempts}


def validate_test_cases(state: TestState) -> dict:
    test_case = LLMTestCase(
        input=state["user_story"],
        actual_output=state["test_cases"],
        retrieval_context=[state["analysis"]],
    )
    evaluation_result = evaluate(
        test_cases=[test_case],
        metrics=[
            AnswerRelevancyMetric(threshold=0.7, model=MODEL_NAME, include_reason=True),
            FaithfulnessMetric(threshold=0.7, model=MODEL_NAME, include_reason=True),
        ],
    )
    metrics = evaluation_result.test_results[0].metrics_data
    relevancy = next(
        metric for metric in metrics if "relevancy" in metric.name.lower()
    )
    faithfulness = next(
        metric for metric in metrics if "faithfulness" in metric.name.lower()
    )
    return {
        "relevancy_score": relevancy.score,
        "relevancy_reason": relevancy.reason,
        "relevancy_pass": relevancy.success,
        "faithfulness_score": faithfulness.score,
        "faithfulness_reason": faithfulness.reason,
        "faithfulness_pass": faithfulness.success,
    }


def human_review(_: TestState) -> dict:
    response = interrupt("Enter approve, edit, or reject with optional feedback")
    decision = response["decision"].strip().lower()
    return {
        "decision": decision,
        "feedback": response.get("feedback", "").strip(),
    }


def route(state: TestState) -> str:
    decision = state["decision"]
    if decision == "approve":
        return "approved"
    if decision == "edit" and state.get("attempts", 0) < 3:
        return "edit"
    return "rejected"


def save_test_cases(state: TestState) -> dict:
    output_file = Path(state["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    json.loads(state["test_cases"])
    output_file.write_text(state["test_cases"], encoding="utf-8")
    print(f"Approved test cases saved to: {output_file}")
    return {"output_file": str(output_file)}


def build_graph():
    builder = StateGraph(TestState)
    builder.add_node("load_document", load_document)
    builder.add_node("analyze_requirements", analyze_requirements)
    builder.add_node("generate_test_cases", generate_test_cases)
    builder.add_node("validate_test_cases", validate_test_cases)
    builder.add_node("human_review", human_review)
    builder.add_node("save_test_cases", save_test_cases)
    builder.add_edge(START, "load_document")
    builder.add_edge("load_document", "analyze_requirements")
    builder.add_edge("analyze_requirements", "generate_test_cases")
    builder.add_edge("generate_test_cases", "validate_test_cases")
    builder.add_edge("validate_test_cases", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route,
        {"approved": "save_test_cases", "rejected": END, "edit": "generate_test_cases"},
    )
    builder.add_edge("save_test_cases", END)
    return builder.compile(checkpointer=MemorySaver())


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "testcases" / "TC001.txt"
    output_file = base_dir / "output" / "test_cases.json"
    graph = build_graph()
    config = {"configurable": {"thread_id": "qa-practice-001"}}
    graph.invoke(
        {"input_file": str(input_file), "output_file": str(output_file)},
        config=config,
    )
    print(graph.get_graph().draw_ascii())

    snapshot = graph.get_state(config)
    while snapshot.next:
        decision = input("Enter approve, edit, or reject: ").strip().lower()
        while decision not in {"approve", "edit", "reject"}:
            decision = input("Please enter approve, edit, or reject: ").strip().lower()
        feedback = input("Enter feedback for the edit: ").strip() if decision == "edit" else ""
        graph.invoke(
            Command(resume={"decision": decision, "feedback": feedback}),
            config=config,
        )
        snapshot = graph.get_state(config)


if __name__ == "__main__":
    main()
