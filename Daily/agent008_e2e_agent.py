from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import subprocess

load_dotenv()
@tool
def  analyze_Requirement(requirement : str) -> str:
    """Analyze a requirement and generate one manual test case."""
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(
        [
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
                """,
            }
        ]
    )
    return response.content

@tool 
def generates_tests(testcases:str)->str:
    """Generate automated tests from the given test cases."""
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(
            [
                {
                    "role": "system",
                    "content": f"""Act as a senior SDET.

                    Generate automated test cases for this requirement using playwright typescript:
                    {testcases} """,
                }
            ]
        )
    return response.content

@tool
def execute_test(test_code : str)->str:
    """Save the generated Playwright test and run it, returning the result."""

    # remove markdown code fences like ```typescript ... ``` if present
    lines = [line for line in test_code.splitlines() if not line.strip().startswith("```")]

    with open("tests/generated.spec.ts", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    process = subprocess.Popen(
        "npx playwright test tests/generated.spec.ts --reporter=list",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )

    output = ""
    for line in process.stdout:
        print(line, end="")
        output += line

    process.wait()

    return output

agent = create_agent(model="openai:gpt-4o-mini", system_prompt=
                     """Act as a senior SDET.
                     Always follow these steps in order using the tools:
                     1. Use analyze_Requirement to create the manual test case.
                     2. Use generates_tests to turn that test case into Playwright TypeScript code.
                     3. Use execute_test to save and run that code, then report the result.""",
                     tools=[analyze_Requirement, generates_tests, execute_test])

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": """
             Generate 1 scenario for the login screen, then run the automated test and tell me the result.

            Application URL:
            https://practicetestautomation.com/practice-test-login/

            Username: student
            Password:Password123 """
        }
    ]
})

print(response["messages"][-1].content)
