import os

from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
from deepeval.test_case import ToolCall

load_dotenv()
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToolCorrectnessMetric

api_key = os.getenv("TAVILI_API_KEY")
if not api_key:
    raise ValueError("Set TAVILI_API_KEY in .env to use Tavily search.")
tavily_client = TavilyClient(api_key=api_key)

testData = [
    {
        "input": "who is the president of the united states?",
        "output": "Donald Trump is the president of the United States.",
        "tool_called": [ToolCall(name="tavily_search")],
    },
    {
        "input": "what is the sum of 2+4?",
        "output": "6",
        "tool_called": [ToolCall(name="add")],
    },
    {
        "input": "what is 4-1?",
        "output": "3",
        "tool_called": [ToolCall(name="subtract")],
    }
]

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b

@tool
def tavily_search(query: str) -> str:
    """Search the web with Tavily."""
    results = tavily_client.search(
        query=query,
        max_results=5,
    )["results"]
    return "\n\n".join(
        f"Title: {result['title']}\nURL: {result['url']}\n{result['content']}"
        for result in results
    )

tools = [add, subtract, tavily_search]

agent = create_agent(
    model="gpt-6-luna",
    system_prompt=(
        "For arithmetic questions, use the add or subtract tool. "
        "For current information, use tavily_search."
    ),
    tools=tools,
)

def query_agent(question: str):
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    tool_calls = [
        tool_call
        for message in response["messages"]
        for tool_call in getattr(message, "tool_calls", [])
    ]
    return response, tool_calls

response, tool_calls = query_agent(testData[1]["input"])

print(type(tool_calls))

test_case = LLMTestCase(
    input=testData[1]["input"],
    expected_output=testData[1]["output"],
    actual_output=response["messages"][-1].text,
    tools_called=[ToolCall(name=tools["name"]) for tools in tool_calls],
    expected_tools=testData[1]["tool_called"],
)

evaluation_result = evaluate(
    test_cases=[test_case],
    metrics=[ToolCorrectnessMetric(model="gpt-6-luna")]
)