
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.test_case import ToolCall
from dotenv import load_dotenv
load_dotenv()
from tavily import TavilyClient
import os
from langchain.agents import create_agent
from langchain.tools import tool

api_key = os.getenv("TAVILI_API_KEY")
if not api_key:
    raise ValueError("Set TAVILI_API_KEY in .env to use Tavily search.")
tavily_client = TavilyClient(api_key=api_key)

@tool
@observe(type='tool')
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
@observe(type='tool')
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b

@tool
@observe(type='tool')
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
        "For every arithmetic question, you MUST call exactly one matching tool: "
        "use add for addition and subtract for subtraction, even for simple "
        "calculations. Never calculate arithmetic directly in your response. "
        "For current information, use tavily_search."
    ),
    tools=tools,
)

@observe(type="agent")
def query_agent(question: str):
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    tool_calls = [
        tool_call
        for message in response["messages"]
        for tool_call in getattr(message, "tool_calls", [])
    ]
    return response, tool_calls

dataset = EvaluationDataset(
    goldens=[
        Golden(input="Capital of India?"),
    ]
)

metric = StepEfficiencyMetric(
    threshold=0.2,
    model="gpt-6-luna",
    include_reason=True,
    verbose_mode=True
)

for golden in dataset.evals_iterator(metrics=[metric]):
    query_agent(golden.input)
