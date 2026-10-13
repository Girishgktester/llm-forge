from pathlib import Path

import requests
from bs4 import BeautifulSoup

from source_config import SOURCES


def load_file(location: str) -> str:
    """Read content from a local file."""
    file_path = Path(location)

    return file_path.read_text(encoding="utf-8")


def load_webpage(location: str) -> str:
    """Fetch a webpage and extract readable text."""
    response = requests.get(
        location,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(
        ["script", "style", "nav", "footer", "header"]
    ):
        element.decompose()

    return soup.get_text(separator=" ", strip=True)


def load_sources() -> list[dict]:
    """Load all configured sources."""
    loaded_sources = []

    for source in SOURCES:
        source_type = source["source_type"]
        location = source["location"]

        if source_type == "file":
            content = load_file(location)

        elif source_type == "web":
            content = load_webpage(location)

        else:
            raise ValueError(
                f"Unsupported source type: {source_type}"
            )

        loaded_sources.append({
            "source_id": source["source_id"],
            "source_type": source_type,
            "content": content,
        })

        print(f"Loaded: {source['source_id']}")

    return loaded_sources