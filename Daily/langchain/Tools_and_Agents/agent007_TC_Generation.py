from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

load_dotenv()


@tool
def generate_testcases(requirement: str) -> str:
    """Generate manual test cases for the given requirement."""

    print("\n--- TOOL CALLED ---")
    print("Requirement received by tool:")
    print(requirement)

    agent = create_agent(model="openai:gpt-4o-mini")

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Generate manual test cases for this requirement:

{requirement}

Generate only one scenario.

The test case should contain:
- Test Case ID
- Scenario
- Actual Result
- Expected Result
- Precondition
- Test Data
""",
                }
            ]
        }
    )

    result = response["messages"][-1].content

    print("\n--- TOOL RESULT ---")
    print(result)
    print(type(result))

    return result


@tool
def convert_to_json(testcases: str) -> dict:
    """Convert generated test cases from JSON string to Python object."""

    result = json.loads(testcases)

    print("JSON result:")
    print(result)

    return result

model = ChatOpenAI(model="gpt-4o-mini", max_tokens=40)
agent = create_agent(
    model=model, system_prompt="Act as a senior QA engineer", tools=[generate_testcases,convert_to_json]
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Generate testcases for login screen give me only 1 testcases "}]}
)

print(result["messages"][-1].content)
