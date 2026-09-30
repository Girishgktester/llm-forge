from langchain_openai import ChatOpenAI

from core.command_code_llm import create_command_code_model

OFFLINE_MODEL = "ollama:qwen3:8b"
SYSTEM_PROMPT = "You are a helpful assistant."


def resolve_model(mode: str) -> ChatOpenAI | str:
    """Pick the online Command Code model or the local Ollama Qwen model."""
    normalized = mode.strip().lower()
    if normalized == "online":
        return create_command_code_model()
    if normalized == "offline":
        return OFFLINE_MODEL
    raise ValueError("mode must be 'online' or 'offline'")
