from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import JournalRecord


class Journal:
    """Append-only JSONL journal; flush+fsync makes each completed attempt durable."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: JournalRecord) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> List[Dict[str, Any]]:
        result = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"corrupt journal line {line_number}: {exc}") from exc
        return result

    def scored(self) -> List[Dict[str, Any]]:
        return sorted(
            (r for r in self.records() if r["status"] == "scored"),
            key=lambda r: r["experiment_id"],
        )

    def next_experiment_id(self) -> int:
        scored = self.scored()
        return 1 if not scored else int(scored[-1]["experiment_id"]) + 1

    def latest_generation(self) -> int:
        records = self.records()
        return max((int(r["generation"]) for r in records), default=0)

    def scored_by_id(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        for record in self.scored():
            if record["experiment_id"] == experiment_id:
                return record
        return None

    def converged(self, epsilon: float = 0.002, window: int = 3) -> bool:
        scores = [float(r["metrics"]["primary"]) for r in self.scored()]
        if len(scores) < window:
            return False
        recent = scores[-window:]
        return recent[-1] - recent[0] < epsilon

    def stop_reason(self, max_experiments: int, epsilon: float, window: int) -> Optional[str]:
        scored = self.scored()
        if len(scored) >= max_experiments:
            return "experiment_cap"
        if self.converged(epsilon, window):
            return "converged"
        return None
