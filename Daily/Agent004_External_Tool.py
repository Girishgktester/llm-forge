from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from tavily import TavilyClient
import os

load_dotenv()

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