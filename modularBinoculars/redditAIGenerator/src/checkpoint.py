import json
import os


class GenerationCheckpoint:
    def __init__(self, path: str):
        self.path = path
        self.completed_ids: set[str] = set()
        self.rows: list[dict] = []
        self.seen_fingerprints: set[str] = set()

    @property
    def completed_count(self) -> int:
        return len(self.completed_ids)

    def reset(self):
        self.completed_ids = set()
        self.rows = []
        self.seen_fingerprints = set()

    def load(self):
        if not os.path.exists(self.path):
            self.reset()
            return

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.completed_ids = set(data.get("completed_ids", []))
        self.rows = data.get("rows", [])
        self.seen_fingerprints = set(data.get("seen_fingerprints", []))
        print(f"Loaded checkpoint with {self.completed_count:,} completed seeds.")

    def add_seed_rows(self, source_id: str, rows: list[dict], new_fingerprints: list[str]):
        self.completed_ids.add(str(source_id))
        self.rows.extend(rows)
        self.seen_fingerprints.update(new_fingerprints)

    def is_completed(self, source_id: str) -> bool:
        return str(source_id) in self.completed_ids

    def save(self):
        payload = {
            "completed_ids": sorted(self.completed_ids),
            "rows": self.rows,
            "seen_fingerprints": sorted(self.seen_fingerprints),
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
