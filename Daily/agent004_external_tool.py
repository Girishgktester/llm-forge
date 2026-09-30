import os
import sys
# Running this file directly puts Daily/ on sys.path, not the project root.
# Add the project root so the "core" package can be imported.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from tavily import TavilyClient
from core.langsmith_config import configure_langsmith

load_dotenv()
configure_langsmith()

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILI_API_KEY")
)

@tool
def get_news(topic: str):

    """Get the latest news articles for a given topic."""

    response = tavily_client.search(
        topic="news",
        query=topic,
        max_results=2
    )

    return response["results"]


@tool
def summary_news(articles: list):
    """Extract article content for the agent to summarize."""

    summaries = []
    for article in articles:
        summaries.append(article["content"])
    return summaries

agent = create_agent(
    model="ollama:qwen3:8b",
    system_prompt=(
        "Act as a news reporter. "
        "Fetch the latest news and summarize it in 5 short bullet points."
    ),
    tools=[get_news, summary_news],
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Fetch the latest news about AI."
            }
        ]
    }
)

print(result["messages"][-1].content)


# ✓ Tool works
# ✓ Tool returns empty result
# ✓ Tool throws exception
# ✓ API timeout
# ✓ Invalid input
# ✓ API unavailable
# ✓ Agent repeatedly calls tool
# ✓ Tool returns malformed data


# 2. Use LangChain Runnables

# execution order