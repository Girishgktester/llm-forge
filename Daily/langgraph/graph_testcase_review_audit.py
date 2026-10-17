import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.sqlite import SqliteSaver  # Un-comment for production persistence
from langgraph.types import interrupt, Command

# ─────────────────── Configuration ───────────────────
MAX_RETRIES: int = int(os.getenv("MAX_TESTCASE_RETRIES", "3"))
INTERRUPT_TIMEOUT_SECONDS: int = int(os.getenv("HITL_TIMEOUT_SECONDS", "300"))  # 5 min default
DEFAULT_REVIEWER: str = os.getenv("DEFAULT_REVIEWER", "anonymous")
ALLOWED_REVIEWERS: list[str] = [
    r.strip()
    for r in os.getenv("ALLOWED_REVIEWERS", "").split(",")
    if r.strip()
]

# ─────────────────── Audit Logger ───────────────────
LOG_DIR = Path(__file__).resolve().parent / "audit_logs"
LOG_DIR.mkdir(exist_ok=True)
AUDIT_FILE = LOG_DIR / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"

audit_logger = logging.getLogger("hitl_audit")
audit_logger.setLevel(logging.INFO)
_fh = logging.FileHandler(AUDIT_FILE, mode="a")
_fh.setFormatter(logging.Formatter("%(message)s"))
audit_logger.addHandler(_fh)


def audit(event: str, detail: dict | None = None):
    """Append an immutable audit record to the JSONL log file."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": detail or {},
    }
    audit_logger.info(json.dumps(record, default=str))
    print(f"[AUDIT] {record['event']} — {record['detail']}")


# ─────────────────── LLM Client ───────────────────
load_dotenv()
llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-6-luna"))

# ─────────────────── State Schema ───────────────────
class MyState(TypedDict, total=False):
    storyname: str
    description: str
    result: str                # Generated test cases text
    decision: str              # "approve" / "reject"
    feedback: str              # Reviewer feedback
    retry_count: int           # How many times we've looped back
    max_retries: int           # Global cap on retries (from config or override)
    reviewer: str              # Who is reviewing
    validated_result: str      # Clean, validated output
    validation_errors: list[str]
    errors: list[str]          # Runtime / side-effect errors caught
    authorized: bool
    audit_trail: list[dict]


# ─────────────────── Nodes ───────────────────

def generate_testcases(state: MyState) -> dict:
    """Generate test cases via LLM with error handling."""
    audit("generate_start", {"reviewer": state.get("reviewer", "unknown"), "retry": state.get("retry_count", 0)})
    try:
        prompt = f"""Generate exactly 5 manual test cases.

Story: {state['storyname']}
Requirement: {state['description']}

{'Previous test cases (revise based on feedback):\n' + state['result'] if state.get('result') else ''}
{'Reviewer feedback:\n' + state['feedback'] if state.get('feedback') else ''}

Rules:
- Only use stated requirements — no assumptions.
- Each test case must have: ID, Title, Steps, Expected Result.
- If feedback was provided, incorporate it fully.
"""
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") and response.content else ""
        audit("generate_success", {"len": len(content), "retry": state.get("retry_count", 0)})
        return {"result": content, "errors": []}
    except Exception as exc:
        err_msg = f"Generation failed: {exc}"
        audit("generate_error", {"error": err_msg, "retry": state.get("retry_count", 0)})
        prev_errors = state.get("errors", [])
        return {"errors": prev_errors + [err_msg]}


def validate_testcases(state: MyState) -> dict:
    """Validate test case quality and enforce retry limits."""
    audit("validate_start", {"retry": state.get("retry_count", 0)})
    errors: list[str] = list(state.get("validation_errors", []))
    result = state.get("result", "")

    # Check minimum length
    if not result or len(result.strip()) < 50:
        errors.append("Test cases too short or empty.")

    # Check for expected sections per test case (simple heuristic)
    if "ID:" not in result and "Title:" not in result and "Steps:" not in result and "Expected Result:" not in result:
        errors.append("Test cases missing required structure (ID/Title/Steps/Expected Result).")

    # Retry limit enforcement
    if state.get("retry_count", 0) >= state.get("max_retries", MAX_RETRIES):
        errors.append(f"Max retries ({state['max_retries']}) reached — cannot regenerate further.")

    # Also check runtime errors
    if state.get("errors"):
        errors.extend([f"Runtime error: {e}" for e in state["errors"]])

    has_validation_errors = len(errors) > 0
    audit("validate_complete", {"has_errors": has_validation_errors, "error_count": len(errors)})
    return {"validation_errors": errors, "validated_result": result if not has_validation_errors else ""}


def reviewer_edit(state: MyState) -> dict:
    """Allow human reviewer to directly edit test cases before final approval."""
    audit("review_edit_start", {"reviewer": state.get("reviewer", "unknown")})
    print("\n" + "=" * 60)
    print("EDIT MODE — You may modify the test cases directly below.")
    print("=" * 60)
    current = state.get("result", "(empty)")
    print(current)
    print("=" * 60)
    edited = input("Paste your edited version (or press Enter to keep as-is):\n> ").strip()
    if edited:
        audit("review_edit_update", {"reviewer": state.get("reviewer", "unknown"), "new_len": len(edited)})
        return {"result": edited, "errors": [], "authorized": True}
    audit("review_edit_keep", {"reviewer": state.get("reviewer", "unknown")})
    return {"authorized": True}


def approval_gate(state: MyState) -> dict:
    """Wait for human approval/rejection decision."""
    reviewer = state.get("reviewer", DEFAULT_REVIEWER)
    audit("approval_request", {"reviewer": reviewer})
    print("\n" + "=" * 60)
    print("Please review these test cases and decide:")
    print("=" * 60)
    print(state.get("result", "(empty)"))
    print("=" * 60)
    decision = interrupt(f"{reviewer}: Approve or Reject the test cases?").strip().lower()
    audit("approval_received", {"reviewer": reviewer, "decision": decision})
    return {"decision": decision}


def feedback_gather(state: MyState) -> dict:
    """Gather revision feedback from reviewer."""
    reviewer = state.get("reviewer", DEFAULT_REVIEWER)
    audit("feedback_request", {"reviewer": reviewer})
    print("\nType feedback (what changes are needed?):")
    feedback = interrupt(f"{reviewer}: What changes are needed?").strip()
    audit("feedback_received", {"reviewer": reviewer, "feedback_len": len(feedback)})
    return {"feedback": feedback}


def end_node(state: MyState) -> dict:
    """Finalize and close out the workflow."""
    audit("workflow_complete", {"reviewer": state.get("reviewer", "unknown")})
    return {}


# ─────────────────── Routing ───────────────────

def validation_router(state: MyState) -> str:
    """Decide where to route after generation based on validation."""
    if state.get("errors"):
        return "error_handler"
    if state.get("retry_count", 0) >= state.get("max_retries", MAX_RETRIES):
        return "end"  # Forced completion at max retries
    if state.get("validation_errors"):
        return "get_feedback"
    return "approve_or_edit"


def approval_router(state: MyState) -> str:
    if state["decision"].strip().casefold() == "approve":
        return "approved_end"
    return "get_feedback"


def edit_router(state: MyState) -> str:
    """After editing, go back to generate to regenerate improved versions."""
    return "generate_testcases"


def error_router(state: MyState) -> str:
    if state.get("retry_count", 0) >= state.get("max_retries", MAX_RETRIES):
        return "end"
    return "generate_testcases"


# ─────────────────── Graph Assembly ───────────────────
graph = StateGraph(MyState)

# Nodes
graph.add_node("generate_testcases", generate_testcases)
graph.add_node("validate_testcases", validate_testcases)
graph.add_node("reviewer_edit", reviewer_edit)
graph.add_node("approval_gate", approval_gate)
graph.add_node("get_feedback", feedback_gather)
graph.add_node("error_handler", lambda s: s)  # Pass-through that surfaces errors
graph.add_node("end_node", end_node)

# Edges
graph.add_edge(START, "generate_testcases")
graph.add_conditional_edges("generate_testcases", validation_router, {
    "get_feedback": "get_feedback",
    "approve_or_edit": "reviewer_edit",
    "error_handler": "error_handler",
    "end": "end_node",
})
graph.add_edge("get_feedback", "generate_testcases")
graph.add_conditional_edges("reviewer_edit", edit_router, {"generate_testcases": "validate_testcases"})
graph.add_edge("validate_testcases", "approval_gate")
graph.add_conditional_edges("approval_gate", approval_router, {
    "approved_end": "end_node",
    "get_feedback": "get_feedback",
})
graph.add_conditional_edges("error_handler", error_router, {"generate_testcases": "validate_testcases", "end": "end_node"})
graph.add_edge("end_node", END)

# ─────────────────── Persistence ───────────────────
# For SQLite persistence (uncomment and install `langgraph-checkpoint-sqlite`):
# checkpointer = SqliteSaver.from_conn_string(str(Path(__file__).parent / "checkpoints.db"))
checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

# ─────────────────── Main Execution Flow ───────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(datetime.now(timezone.utc).timestamp())}}

    # --- Authorization ---
    reviewer_input = input("Enter reviewer name (or leave blank for anonymous): ").strip()
    reviewer = reviewer_input or DEFAULT_REVIEWER

    if ALLOWED_REVIEWERS and reviewer not in ALLOWED_REVIEWERS:
        print(f"\n[UNAUTHORIZED] '{reviewer}' is not in the allowed reviewers list: {ALLOWED_REVIEWERS}")
        audit("auth_failure", {"reviewer": reviewer, "allowed": ALLOWED_REVIEWERS})
        exit(1)

    audit("session_start", {"reviewer": reviewer})

    # Set defaults + override retry limits if user wants
    _overrides = {"max_retries": MAX_RETRIES}
    custom_limit = input(f"Override max retries? (leave blank for {MAX_RETRIES}): ").strip()
    if custom_limit.isdigit():
        _overrides["max_retries"] = int(custom_limit)

    story = input("\nEnter story name: ")
    description = input("Enter requirement description: ")

    initial_state = {
        "storyname": story,
        "description": description,
        "reviewer": reviewer,
        "retry_count": 0,
        "max_retries": _overrides["max_retries"],
        "errors": [],
        "validation_errors": [],
        "authorized": True,
        "audit_trail": [{"ts": datetime.now(timezone.utc).isoformat(), "event": "initialized"}],
    }

    try:
        # Initial invoke — flows through generate → validate → edit/approval
        result = app.invoke(initial_state, config)
        print("\n--- Final Result ---")
        print(result.get("result", "(no result)"))
    except KeyboardInterrupt:
        print("\n\nSession interrupted. State saved to checkpointer.")
        audit("session_interrupted", {"reviewer": reviewer, "thread_id": config["configurable"]["thread_id"]})
    except Exception as exc:
        audit("fatal_error", {"error": str(exc), "reviewer": reviewer})
        print(f"\n[FATAL ERROR] {exc}")
    finally:
        print("\n--- Graph Structure ---")
        print(app.get_graph(config).draw_mermaid())
        print(f"\nAudit log: {AUDIT_FILE}")
        print(f"Checkpoint thread_id: {config['configurable']['thread_id']}")
