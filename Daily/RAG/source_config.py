from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "dataset" / "qa_knowledge.txt"

SOURCES = [
    {
        "source_id": str(DATASET_PATH),
        "source_type": "file",
        "location": str(DATASET_PATH),
    },
    {
        "source_id": "https://playwright.dev/docs/test-assertions",
        "source_type": "web",
        "location": "https://playwright.dev/docs/test-assertions",
    },
]

