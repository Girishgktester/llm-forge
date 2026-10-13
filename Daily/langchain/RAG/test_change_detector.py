from source_loader import load_sources
from manifest_manager import load_manifest
from change_detector import detect_changes
from change_detector import detect_deleted_sources

print("Change detector test started")

sources = load_sources()
manifest = load_manifest()

results = detect_changes(sources, manifest)

for result in results:
    print(f"{result['status']}: {result['source_id']}")
    
deleted = detect_deleted_sources(sources, manifest)

print("\nDeleted sources:")
for source_id in deleted:
    print(source_id)

test_sources = sources[:1]

deleted = detect_deleted_sources(test_sources, manifest)

print("\nDeleted sources:")
for source_id in deleted:
    print(source_id)