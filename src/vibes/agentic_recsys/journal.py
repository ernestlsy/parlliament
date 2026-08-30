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

    def primary_scores(self) -> List[float]:
        return [float(r["metrics"]["primary"]) for r in self.scored()]

    def running_best(self) -> List[float]:
        """Cumulative maximum of validation primary, in scored-experiment order.

        Convergence is measured on this series rather than on raw scores. The official
        rule asks whether the run has *improved*; a raw comparison also fires when the
        newest experiment is merely worse than an older one, so one dud halts the run
        (this ended run_2 at 3 of 50). A running best is monotone non-decreasing, so a
        dud can only fail to advance it, never reverse it.
        """
        best: List[float] = []
        current = float("-inf")
        for score in self.primary_scores():
            current = max(current, score)
            best.append(current)
        return best

    def best_record(self) -> Optional[Dict[str, Any]]:
        """The validation-best scored experiment — the checkpoint that gets submitted."""
        scored = self.scored()
        if not scored:
            return None
        return max(scored, key=lambda r: float(r["metrics"]["primary"]))

    def converged(self, epsilon: float = 0.002, window: int = 3) -> bool:
        best = self.running_best()
        if len(best) < window:
            return False
        recent = best[-window:]
        return recent[-1] - recent[0] < epsilon

    def stop_reason(self, max_experiments: int, epsilon: float, window: int) -> Optional[str]:
        scored = self.scored()
        if len(scored) >= max_experiments:
            return "experiment_cap"
        if self.converged(epsilon, window):
            return "converged"
        return None
