from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """Load the first .env file found, starting at the project root."""
    project_root = Path(__file__).resolve().parent.parent
    candidates = (
        project_root / ".env",
        Path.cwd() / ".env",
        project_root.parent / ".env",
    )
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return
