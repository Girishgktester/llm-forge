"""Chat model that runs on the Command Code Provider API instead of OpenAI.

Command Code exposes an OpenAI-compatible endpoint, so pointing a normal
LangChain ``ChatOpenAI`` at it is all that is needed.
"""

import os

from langchain_openai import ChatOpenAI

BASE_URL = "https://api.commandcode.ai/provider/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


def create_command_code_model(model: str | None = None, **kwargs) -> ChatOpenAI:
    """Create a chat model that runs on Command Code.

    The API key comes from ``COMMAND_CODE_API_KEY``, the model from
    ``COMMAND_CODE_MODEL`` when no model is passed in.
    """
    api_key = os.getenv("COMMAND_CODE_API_KEY")
    if not api_key:
        raise ValueError(
            "COMMAND_CODE_API_KEY is not set. Add it to your .env file."
        )

    return ChatOpenAI(
        model=model or os.getenv("COMMAND_CODE_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        base_url=os.getenv("COMMAND_CODE_BASE_URL", BASE_URL),
        **kwargs,
    )
