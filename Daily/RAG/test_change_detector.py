from source_loader import load_sources
from manifest_manager import load_manifest
from change_detector import detect_changes

print("Change detector test started")

sources = load_sources()
manifest = load_manifest()

results = detect_changes(sources, manifest)

for result in results:
    print(f"{result['status']}: {result['source_id']}")