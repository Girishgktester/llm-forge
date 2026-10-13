from source_loader import load_sources
from content_hasher import calculate_hash


sources = load_sources()

for source in sources:
    content_hash = calculate_hash(source["content"])

    print(f"\nSource: {source['source_id']}")
    print(f"Hash: {content_hash}")