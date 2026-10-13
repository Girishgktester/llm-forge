from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")


class TestState(TypedDict):
    feature: str
    description: str
    type: str
    test_cases: str
    automation_code: str


def analyze_requirement(state: TestState):
    prompt = f"""
    Analyze this software requirement and classify it into one category:
    - "functional": user-facing feature (login, search, registration)
    - "non-functional": performance, security, reliability
    - "data": database, API data handling, validation
    
    Feature: {state['feature']}
    Description: {state['description']}
    
    Return only the category name.
    """
    response = llm.invoke(prompt)
    return {"type": response.content.strip()}


def generate_testcases(state: TestState):
    prompt = f"""
    Generate 3 manual test cases for this requirement.
    Feature: {state['feature']}
    Type: {state['type']}
    
    Format each test case as:
    TC1: <step-by-step>, Expected: <expected result>
    TC2: ...
    TC3: ...
    """
    response = llm.invoke(prompt)
    return {"test_cases": response.content}


def generate_automation(state: TestState):
    prompt = f"""
    Generate basic Playwright TypeScript automation code for this requirement.
    Feature: {state['feature']}
    Type: {state['type']}
    
    Keep code simple with basic locators and assertions.
    """
    response = llm.invoke(prompt)
    return {"automation_code": response.content}


def route_type(state: TestState):
    if state["type"] == "non-functional":
        return "both"
    else:
        return "split"


builder = StateGraph(TestState)

builder.add_node("analyze", analyze_requirement)
builder.add_node("testcases", generate_testcases)
builder.add_node("automation", generate_automation)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "testcases")
builder.add_conditional_edges("testcases", route_type, {"split": "automation", "both": "automation"})
builder.add_edge("automation", END)

graph = builder.compile()

print(graph.get_graph().draw_ascii())

print(graph.invoke({
    "feature": "User Login Page",
    "description": "Users should be able to login with email and password with error handling for wrong credentials"
}))
