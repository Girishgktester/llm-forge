from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()
@tool
def analyze_Requirement(requirement: str) -> str:
    """Analyze a requirement and generate one manual test case."""
    if not requirement or not requirement.strip():
        return "ERROR: Requirement cannot be empty."
    model = ChatOpenAI(model="gpt-4o-mini")
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = model.invoke(
                [
                    {
                        "role": "user",
                        "content": f"""
                        Generate one manual test case for:
                        {requirement}
                        """
                    }
                ]
            )
            return response.content
        except Exception as e:
            if attempt == max_retries:
                return f"ERROR: analyze_Requirement failed after retries: {str(e)}"
            print(f"Attempt {attempt + 1} failed. Retrying...")

@tool
def generates_tests(testcases: str) -> str:
    """Generate automated tests from the given test cases."""
    if not testcases or not testcases.strip():
        return "ERROR: No test case details were provided."

    try:
        model = ChatOpenAI(model="gpt-4o-mini")
        response = model.invoke(
            [
                {
                    "role": "system",
                    "content": f"""Act as a senior SDET.

                    Generate automated test cases for this requirement using Playwright TypeScript:
                    {testcases} """,
                }
            ]
        )
        return response.content
    except Exception as e:
        return f"ERROR: Failed to generate test code: {str(e)}"

@tool
def execute_test(test_code: str) -> str:
    """Save the generated Playwright test and run it, returning the result."""
    if not test_code or not test_code.strip():
        return "ERROR: No test code was provided to run."

    try:
        lines = [line for line in test_code.splitlines() if not line.strip().startswith("```")]

        os.makedirs("tests", exist_ok=True)
        with open("tests/generated.spec.ts", "w", encoding="utf-8") as file:
            file.write("\n".join(lines))

        process = subprocess.run(
            "npx playwright test tests/generated.spec.ts --reporter=list",
            capture_output=True,
            text=True,
            shell=True,
        )

        output = process.stdout + process.stderr
        if process.returncode != 0:
            return f"ERROR: Playwright test failed.\n{output}"
        return output
    except FileNotFoundError:
        return "ERROR: Playwright is not installed or the 'npx' command is not available. Run: npm install -D @playwright/test"
    except Exception as e:
        return f"ERROR: Test execution failed: {str(e)}"

agent = create_agent(model="openai:gpt-4o-mini", system_prompt=
                     """Act as a senior SDET.
                     Always follow these steps in order using the tools:
                     1. Use analyze_Requirement to create the manual test case.
                     2. Use generates_tests to turn that test case into Playwright TypeScript code.
                     3. Use execute_test to save and run that code, then report the result.""",
                     tools=[analyze_Requirement, generates_tests, execute_test])

try:
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
except Exception as e:
     print(f"ERROR: Agent run failed:    {str(e)}")
