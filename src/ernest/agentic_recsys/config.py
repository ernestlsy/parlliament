from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class SystemConfig:
    workspace: str
    data_dir: str
    run_name: str = "run_1"
    max_experiments: int = 50
    convergence_epsilon: float = 0.002
    convergence_window: int = 3
    max_debug_attempts: int = 3
    experiment_timeout_seconds: int = 900
    max_consultant_rounds: int = 3
    max_backfills_per_slot: int = 2
    max_draft_hypotheses: int = 3
    candidate_pool_size: int = 12
    screening_timeout_seconds: int = 900
    screening_holdout_fraction: float = 0.25
    screening_seed: int = 0
    force_rescreen: bool = False
    python_executable: str = ""

    def __post_init__(self) -> None:
        self.workspace = str(Path(self.workspace).resolve())
        self.data_dir = str(Path(self.data_dir).resolve())
        if not self.python_executable:
            import sys
            self.python_executable = sys.executable
        for name in (
            "max_experiments", "convergence_window", "max_debug_attempts",
            "experiment_timeout_seconds", "max_consultant_rounds",
            "max_backfills_per_slot", "max_draft_hypotheses",
            "candidate_pool_size", "screening_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.convergence_epsilon <= 0:
            raise ValueError("convergence_epsilon must be positive")
        if not 0.1 <= self.screening_holdout_fraction <= 0.5:
            raise ValueError("screening_holdout_fraction must be between 0.1 and 0.5")

    @property
    def run_dir(self) -> Path:
        return Path(self.workspace) / "runs" / self.run_name

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SystemConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def public_dict(self) -> Dict[str, Any]:
        return asdict(self)
