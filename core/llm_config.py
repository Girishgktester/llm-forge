ONLINE_MODEL = "openai:gpt-4o-mini"
OFFLINE_MODEL = "ollama:qwen3:8b"
SYSTEM_PROMPT = "You are a helpful assistant."


def resolve_model(mode: str) -> str:
    """Pick the online OpenAI model or the local Ollama Qwen model."""
    normalized = mode.strip().lower()
    if normalized == "online":
        return ONLINE_MODEL
    if normalized == "offline":
        return OFFLINE_MODEL
    raise ValueError("mode must be 'online' or 'offline'")
