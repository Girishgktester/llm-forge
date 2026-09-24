import os


def configure_langsmith(project: str = "LLM_Forge") -> None:
    """Turn on LangSmith tracing without overriding values already in the environment."""
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project)
