from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")


class TestState(TypedDict):
    feature: str
    description: str
    category: str
    manual_testcases: str
    ui_automation_code: str
    api_automation_code: str


def analyze_requirement(state: TestState):
    prompt = f"""
    Classify this requirement into one category:
    - "ui": anything visible to users (pages, buttons, forms, dashboards)
    - "api": backend endpoints, data handling, database operations
    
    Feature: {state['feature']}
    Description: {state['description']}
    
    Return only the category name.
    """
    response = llm.invoke(prompt)
    return {"category": response.content.strip()}


def generate_manual_testcases(state: TestState):
    prompt = f"""
    Generate 3 manual test cases for this requirement.
    Feature: {state['feature']}
    Category: {state['category']}
    
    Format each test case as:
    TC1: <step-by-step>, Expected: <expected result>
    TC2: ...
    TC3: ...
    """
    response = llm.invoke(prompt)
    return {"manual_testcases": response.content}


def generate_ui_automation(state: TestState):
    prompt = f"""
    Generate basic Playwright TypeScript code for this UI feature.
    Feature: {state['feature']}
    Description: {state['description']}
    
    Keep it simple with basic locators and assertions.
    """
    response = llm.invoke(prompt)
    return {"ui_automation_code": response.content}


def generate_api_automation(state: TestState):
    prompt = f"""
    Generate pytest code for this API/backend feature.
    Feature: {state['feature']}
    Description: {state['description']}
    
    Include endpoint calls, status code checks, and response validation.
    """
    response = llm.invoke(prompt)
    return {"api_automation_code": response.content}


def route_category(state: TestState):
    # UI features go to UI automation, API features go to API automation
    if state["category"] == "ui":
        target = "ui_automation"
    else:
        target = "api_automation"
    print(f"  >>> Route decision: category='{state['category']}' -> {target}")
    return target


builder = StateGraph(TestState)

builder.add_node("analyze", analyze_requirement)
builder.add_node("testcases", generate_manual_testcases)
builder.add_node("ui_automation", generate_ui_automation)
builder.add_node("api_automation", generate_api_automation)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "testcases")
builder.add_conditional_edges("testcases", 
    route_category, {
    "ui_automation": "ui_automation",
    "api_automation": "api_automation"
})
builder.add_edge("ui_automation", END)
builder.add_edge("api_automation", END)

graph = builder.compile()

if __name__ == "__main__":
    print(graph.get_graph().draw_ascii())

    print('\n=== API FEATURE ===')
    print(graph.invoke({
        "feature": "Create User API",
        "description": "POST /api/users endpoint that creates user and returns JSON response"
    }))
