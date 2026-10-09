from source_loader import load_sources
from content_hasher import calculate_hash
from manifest_manager import load_manifest, save_manifest

sources = load_sources()

manifest = load_manifest()

for source in sources:
    source_id = source["source_id"]
    current_hash = calculate_hash(source["content"])

    previous_hash = manifest.get(source_id)

    if previous_hash is None:
        print(f"NEW SOURCE: {source_id}")
    elif previous_hash != current_hash:
        print(f"CHANGED SOURCE: {source_id}")
    else:
        print(f"UNCHANGED SOURCE: {source_id}")

    manifest[source_id] = current_hash

save_manifest(manifest)

print("Manifest saved successfully.")