import os
import sys

from core.agent_factory import create_basic_agent
from core.langsmith_config import configure_langsmith
from utils.env_loader import load_env


def main() -> None:
    load_env()
    configure_langsmith()

    mode = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_MODE", "online")
    agent = create_basic_agent(mode)
    print(agent.invoke({"messages": [("user", "Say hello in one sentence.")]}))


if __name__ == "__main__":
    main()
