import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent / "source_manifest.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}

    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )