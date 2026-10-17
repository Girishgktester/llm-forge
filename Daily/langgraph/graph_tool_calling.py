from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

model = ChatOpenAI(model="gpt-6-luna")

@tool
def generate_login_testcases(requirement: str) -> str:
    """Generate basic test cases for a login requirement."""
    return f"Generate login test cases for: {requirement}"

@tool
def generate_signup_testcases(requirement: str) -> str:
    """Generate basic test cases for a signup requirement."""
    return f"Generate signup test cases for: {requirement}"

agent = create_agent(
    model=model,
    tools=[
        generate_login_testcases,
        generate_signup_testcases
    ]
)

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Generate test cases for a login page[5 scenarios only] return in json format"
            }
        ]
    })
    print(result["messages"][-1].text)
