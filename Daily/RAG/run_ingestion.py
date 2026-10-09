from source_loader import load_sources
from manifest_manager import load_manifest, save_manifest
from change_detector import detect_changes
from index_manager import index_sources


def run_ingestion():
    # 1. Load current source content
    sources = load_sources()

    # 2. Read previously recorded hashes
    manifest = load_manifest()

    # 3. Detect new, changed and unchanged sources
    results = detect_changes(sources, manifest)

    # 4. Index new or changed sources
    successful_sources = index_sources(results)

    # 5. Update manifest only for successfully indexed sources
    for source in successful_sources:
        manifest[source["source_id"]] = source["content_hash"]

    save_manifest(manifest)

    print("\nIngestion completed.")
    print(f"Successfully indexed: {len(successful_sources)}")


if __name__ == "__main__":
    run_ingestion()