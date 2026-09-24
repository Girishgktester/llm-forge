from langchain.agents import create_agent

from core.llm_config import SYSTEM_PROMPT, resolve_model


def create_basic_agent(mode: str = "online"):
    """Build the LangChain agent for online or offline mode."""
    return create_agent(resolve_model(mode), system_prompt=SYSTEM_PROMPT)
