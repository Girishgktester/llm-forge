from content_hasher import calculate_hash


def detect_changes(sources: list[dict], manifest: dict) -> list[dict]:
    results = []

    for source in sources:
        source_id = source["source_id"]
        current_hash = calculate_hash(source["content"])
        previous_hash = manifest.get(source_id)

        if previous_hash is None:
            status = "NEW"
        elif previous_hash != current_hash:
            status = "CHANGED"
        else:
            status = "UNCHANGED"

        results.append({
            **source,
            "content_hash": current_hash,
            "status": status,
        })

    return results