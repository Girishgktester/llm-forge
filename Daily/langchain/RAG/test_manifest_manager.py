from source_loader import load_sources
from content_hasher import calculate_hash
from manifest_manager import load_manifest, save_manifest


class ManifestSync:
    """Checks loaded sources against stored manifest hashes, updates & saves."""

    STATUS_NEW = "NEW SOURCE"
    STATUS_CHANGED = "CHANGED SOURCE"
    STATUS_UNCHANGED = "UNCHANGED SOURCE"

    def __init__(self):
        self.sources = load_sources()
        self.manifest = load_manifest()

    def check_source(self, source_id: str, content: str) -> str:
        current_hash = calculate_hash(content)
        previous_hash = self.manifest.get(source_id)

        if previous_hash is None:
            return self.STATUS_NEW
        if previous_hash != current_hash:
            return self.STATUS_CHANGED
        return self.STATUS_UNCHANGED

    def update_manifest(self, source_id: str, content: str) -> None:
        self.manifest[source_id] = calculate_hash(content)

    def sync(self) -> dict[str, str]:
        results: dict[str, str] = {}

        for source in self.sources:
            source_id = source["source_id"]
            status = self.check_source(source_id, source["content"])
            print(f"[{status}] {source_id}")
            results[source_id] = status
            self.update_manifest(source_id, source["content"])

        save_manifest(self.manifest)
        print("Manifest saved successfully.")
        return results


if __name__ == "__main__":
    mgr = ManifestSync()
    results = mgr.sync()
    for sid, status in results.items():
        print(f"  → {sid}: {status}")
