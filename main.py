from typing import Any

from langsmith import traceable
from core.agent_factory import create_basic_agent
from core.langsmith_config import configure_langsmith
from utils.env_loader import load_env
from tavily import TavilyClient
import os
import sys

def main() -> None:
    load_env()
    configure_langsmith()

    mode = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_MODE", "online")
    agent = create_basic_agent(mode)
    print(agent.invoke({"messages": [("user", "Say hello in one sentence.")]}))


if __name__ == "__main__":
    main()

@traceable(name="tavily_search", run_type="tool")
def search_web(query: str):
    client = TavilyClient(api_key=os.getenv("TAVILI_API_KEY"))
    return client.search(query)

print(search_web[Any, Any]("Who is Leo Messi?"))
