from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from core.langsmith_config import configure_langsmith

load_dotenv()
configure_langsmith()

@tool
def generate_login_testcases(requirement: str) -> str:
    """Generate basic test cases for a login requirement."""
    return f"Generate login test cases for: {requirement}"


@tool
def generate_signup_testcases(requirement: str) -> str:
    """Generate basic test cases for a signup requirement."""
    return f"Generate signup test cases for: {requirement}"


agent = create_agent(
    model="ollama:qwen3:8b",
    tools=[
        generate_login_testcases,
        generate_signup_testcases
    ]
)


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Generate test cases for a login page[5 sceantios only]"
        }
    ]
})

for message in result["messages"]:
    print(f"{type(message).__name__}:")