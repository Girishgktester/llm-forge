from source_loader import load_sources

sources = load_sources()

for source in sources:
    print(f"\nSource ID: {source['source_id']}")
    print(f"Type: {source['source_type']}")
    print(f"Characters: {len(source['content'])}")
    print(f"Preview: {source['content'][:150]}")