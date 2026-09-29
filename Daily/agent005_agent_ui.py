import streamlit as st
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
        "Summarize latest AI news 5 bullet points."
    ),
    tools=[get_news, summary_news],
)

st.title("AI News Agent")

with st.form("search_form"):
    userquery = st.text_input("Enter your query")
    submitted = st.form_submit_button("Search")

if submitted and userquery.strip():
    with st.spinner("Hmm wait for few seconds i am running in local model"):
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": userquery
                    }
                ]
            }
        )
    st.write(result["messages"][-1].content)

