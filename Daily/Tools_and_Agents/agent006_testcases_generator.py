from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
import streamlit as st

@tool
def generate_Testcases(requirement: str) -> str:
    """Generate manual test cases for the given requirement using Qwen."""
    model = ChatOllama(model="qwen3:8b", temperature=0.6)
    response = model.invoke(
    f"""Generate manual test cases for this requirement: {requirement}

    Include a test case ID, 
    scenario, 
    steps, 
    test data, 
    expected result, 
    and priority.
    Return only the generated test cases. """
    )
    return str(response.content)


agent = create_agent(
    model="ollama:qwen3:8b",
    system_prompt="""
    You are a Senior QA Engineer.
    Generate test cases based on the user's requirement.
    
    Each test case must contain:
    Test Case ID
    Scenario
    Test Steps
    Test Data
    Expected Result
    Priority """,
    tools=[generate_Testcases],
)


with st.form("search_form"):
    userinput = st.text_input("Enter your requirement")
    typeoftestcases = st.text_input("Kind of test case: Functional / Positive / Negative / Boundary / All")
    Numberoftestcases = st.text_input("Number of test cases")
    submitted = st.form_submit_button("Generate Test Cases")

if submitted:
    if not userinput.strip():
        st.error("Please enter a requirement.")
    else:
        prompt = f"""Requirement:
        {userinput}

        Test Type:
        {typeoftestcases}

        Number of Test Cases:
        {Numberoftestcases} """

        response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        st.write(response["messages"][-1].content)
