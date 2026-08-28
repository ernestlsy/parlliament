from __future__ import annotations

import difflib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agents import Consultant, EvolutionJudge, Orchestrator
from .config import SystemConfig
from .experimentor import Experimentor
from .journal import Journal
from .knowledge import KnowledgeBase
from .llm import LLMClient
from .sandbox import (
    apply_agent_patches,
    config_diff,
    create_attempt_sandbox,
    finalize_sandbox,
)
from .schemas import AGENT_FILES, FailureKind, Hypothesis, JournalRecord, Mode, Status


class Overseer:
    """Sequential owner of generation scheduling, retries, counting, and stopping."""

    def __init__(self, config: SystemConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.journal = Journal(config.run_dir / "journal.jsonl")
        knowledge = KnowledgeBase(Path(__file__).parent / "knowledge").documents()
        self.judge = EvolutionJudge(llm, knowledge)
        self.consultant = Consultant(llm, config.max_consultant_rounds)
        self.orchestrator = Orchestrator(llm)
        self.experimentor = Experimentor(config.python_executable, config.data_dir)
        self.seed_dir = Path(__file__).parent / "seed"

    def initialize(self) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(self.config.run_dir / "system_config.json")
        if not self.seed_dir.is_dir():
            raise FileNotFoundError(f"seed experiment is missing: {self.seed_dir}")
        if not Path(self.config.data_dir).is_dir():
            raise FileNotFoundError(f"data directory does not exist: {self.config.data_dir}")

    def _archive(self) -> List[Dict[str, Any]]:
        seed = {
            "attempt_id": "seed", "experiment_id": 0, "generation": 0,
            "parent_experiment_id": 0,
            "hypothesis_text": "Official five-field pointwise FM seed baseline",
            "hypothesis_scores": {"interestingness": 5, "novelty": 1, "feasibility": 10},
            "mode": "seed", "code_diff": {}, "config_diff": "",
            "active_sub_agents": [],
            "metrics": {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016},
            "status": "scored", "failure_reason": None, "consultant_rounds": 0,
            "sandbox": str(self.seed_dir), "created_at": "seed",
        }
        return [seed] + self.journal.records()

    def _snapshot(self, record: Dict[str, Any]) -> Dict[str, Any]:
        experiment_id = int(record["experiment_id"])
        path = self.seed_dir if experiment_id == 0 else Path(record["sandbox"])
        return {
            "experiment_id": experiment_id,
            "generation": record["generation"],
            "hypothesis_text": record["hypothesis_text"],
            "metrics": record["metrics"],
            "files": {
                name: (path / name).read_text(encoding="utf-8")
                for names in AGENT_FILES.values() for name in names
            },
        }

    def _reference_records(self, mode: Mode, archive: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = [r for r in archive if r["status"] == "scored"]
        if mode is Mode.IMPROVE:
            latest_generation = max(int(r["generation"]) for r in scored)
            recent = [r for r in scored if int(r["generation"]) == latest_generation]
            return [max(recent, key=lambda r: float(r["metrics"]["primary"]))]
        return scored

    def _parent_dir(self, experiment_id: int) -> Path:
        if experiment_id == 0:
            return self.seed_dir
        record = self.journal.scored_by_id(experiment_id)
        if not record:
            raise ValueError(f"parent experiment {experiment_id} is not scored")
        path = Path(record["sandbox"])
        if not path.is_dir():
            raise FileNotFoundError(f"parent sandbox does not exist: {path}")
        return path

    def _consult(
        self, proposals: Sequence[Hypothesis], archive: List[Dict[str, Any]], parent_ids: Sequence[int]
    ) -> List[Tuple[Hypothesis, int, Optional[str]]]:
        accepted = []
        review_archive = list(archive)
        for proposal in proposals:
            resolved, rounds, caveat = self.consultant.resolve(
                proposal, self.judge, review_archive, parent_ids
            )
            if resolved is not None:
                accepted.append((resolved, rounds, caveat))
                review_archive.append({
                    "experiment_id": None,
                    "generation": "pending",
                    "parent_experiment_id": resolved.parent_experiment_id,
                    "hypothesis_text": resolved.text,
                    "hypothesis_scores": asdict(resolved.scores),
                    "status": "consultant_accepted_pending",
                    "failure_reason": None,
                })
        return accepted

    def _run_hypothesis(
        self,
        hypothesis: Hypothesis,
        *,
        generation: int,
        mode: Mode,
        consultant_rounds: int,
        caveat: Optional[str],
    ) -> JournalRecord:
        parent_dir = self._parent_dir(hypothesis.parent_experiment_id)
        attempt_id, sandbox = create_attempt_sandbox(self.config.run_dir, parent_dir)
        code_diffs: Dict[str, str] = {}
        active_agents: List[str] = []
        failure_reason: Optional[str] = None
        last_failure = None
        started = time.monotonic()
        try:
            plan = self.orchestrator.plan(hypothesis, parent_dir)
            active_agents = list(plan.active_agents)
            (sandbox / "interface_contract.json").write_text(
                json.dumps(asdict(plan.contract), indent=2), encoding="utf-8"
            )
            (sandbox / "hypothesis.json").write_text(
                json.dumps(asdict(hypothesis), indent=2), encoding="utf-8"
            )
            for agent in active_agents:
                patches = self.orchestrator.generate_patches(
                    agent=agent, hypothesis=hypothesis, plan=plan, sandbox=sandbox
                )
                apply_agent_patches(sandbox, patches, AGENT_FILES[agent])
                for filename, patch in patches.items():
                    code_diffs[filename] = code_diffs.get(filename, "") + patch

            resource_failures = 0
            metrics = None
            for debug_attempt in range(1, self.config.max_debug_attempts + 1):
                remaining = self.config.experiment_timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    failure_reason = "experiment wall-clock ceiling exhausted"
                    break
                metrics, failure = self.experimentor.run(
                    sandbox, plan, attempt=debug_attempt, timeout_seconds=remaining
                )
                if metrics is not None:
                    break
                assert failure is not None
                last_failure = failure
                if failure.kind in (FailureKind.RESOURCE, FailureKind.TIMEOUT):
                    resource_failures += 1
                    if resource_failures > 1:
                        failure.kind = FailureKind.SEMANTIC
                        failure.message = "repeated resource failure; configuration is invalid"
                if debug_attempt == self.config.max_debug_attempts:
                    break
                for agent in failure.responsible_agents:
                    if agent not in active_agents:
                        active_agents.append(agent)
                    patches = self.orchestrator.generate_patches(
                        agent=agent, hypothesis=hypothesis, plan=plan,
                        sandbox=sandbox, failure=failure,
                    )
                    apply_agent_patches(sandbox, patches, AGENT_FILES[agent])
                    for filename, patch in patches.items():
                        code_diffs[filename] = code_diffs.get(filename, "") + patch
            if metrics is None:
                failure_reason = failure_reason or (
                    f"{last_failure.kind.value}: {last_failure.message}\n{last_failure.traceback[-4000:]}"
                    if last_failure else "experiment failed without a structured error"
                )
                final_path = finalize_sandbox(
                    sandbox, self.config.run_dir, experiment_id=None, attempt_id=attempt_id
                )
                status, experiment_id = Status.ABANDONED, None
                metrics = {}
            else:
                experiment_id = self.journal.next_experiment_id()
                final_path = finalize_sandbox(
                    sandbox, self.config.run_dir,
                    experiment_id=experiment_id, attempt_id=attempt_id,
                )
                status = Status.SCORED
        except Exception as exc:
            failure_reason = f"agent_or_guardrail_failure: {type(exc).__name__}: {exc}"
            if last_failure is not None:
                failure_reason += (
                    f"; preceding {last_failure.kind.value}: {last_failure.message}; "
                    f"{last_failure.traceback[-2000:]}"
                )
            final_path = finalize_sandbox(
                sandbox, self.config.run_dir, experiment_id=None, attempt_id=attempt_id
            )
            status, experiment_id, metrics = Status.ABANDONED, None, {}

        record = JournalRecord(
            attempt_id=attempt_id,
            experiment_id=experiment_id,
            generation=generation,
            parent_experiment_id=hypothesis.parent_experiment_id,
            hypothesis_text=hypothesis.text + (f" [CONSULTANT CAVEAT: {caveat}]" if caveat else ""),
            hypothesis_scores=asdict(hypothesis.scores),
            mode=mode.value,
            code_diff={},
            config_diff="",
            active_sub_agents=active_agents,
            metrics=metrics,
            status=status.value,
            failure_reason=failure_reason,
            consultant_rounds=consultant_rounds,
            sandbox=str(final_path.resolve()),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        (final_path / "patch_history.json").write_text(
            json.dumps(code_diffs, indent=2), encoding="utf-8"
        )
        final_diffs = {}
        for names in AGENT_FILES.values():
            for filename in names:
                before = (parent_dir / filename).read_text(encoding="utf-8").splitlines(keepends=True)
                after = (final_path / filename).read_text(encoding="utf-8").splitlines(keepends=True)
                diff = "".join(difflib.unified_diff(
                    before, after, fromfile=filename, tofile=filename,
                ))
                if diff:
                    final_diffs[filename] = diff
        record.code_diff = {
            key: value for key, value in final_diffs.items() if key != "config.json"
        }
        record.config_diff = config_diff(final_diffs)
        self.journal.append(record)
        return record

    def run(self) -> Dict[str, Any]:
        self.initialize()
        stop = self.journal.stop_reason(
            self.config.max_experiments,
            self.config.convergence_epsilon,
            self.config.convergence_window,
        )
        generation = self.journal.latest_generation() + 1
        while stop is None:
            mode = Mode.IMPROVE if generation % 2 == 0 else Mode.DRAFT
            archive = self._archive()
            references = self._reference_records(mode, archive)
            snapshots = [self._snapshot(record) for record in references]
            requested = 1 if mode is Mode.IMPROVE else self.config.max_draft_hypotheses
            proposals = self.judge.propose(
                mode=mode,
                generation=generation,
                archive=archive,
                reference_snapshots=snapshots,
                count=requested,
            )
            parent_ids = [int(r["experiment_id"]) for r in references]
            accepted = self._consult(proposals, archive, parent_ids)
            for hypothesis, rounds, caveat in accepted:
                backfills = 0
                while True:
                    record = self._run_hypothesis(
                        hypothesis,
                        generation=generation,
                        mode=mode,
                        consultant_rounds=rounds,
                        caveat=caveat,
                    )
                    if record.status == Status.SCORED.value:
                        stop = self.journal.stop_reason(
                            self.config.max_experiments,
                            self.config.convergence_epsilon,
                            self.config.convergence_window,
                        )
                        break
                    backfills += 1
                    if backfills > self.config.max_backfills_per_slot:
                        break
                    archive = self._archive()
                    replacement = self.judge.propose(
                        mode=mode,
                        generation=generation,
                        archive=archive,
                        reference_snapshots=[self._snapshot(r) for r in references],
                        count=1,
                        failed_attempt=record.to_dict(),
                    )[0]
                    resolved = self._consult([replacement], archive, parent_ids)
                    if not resolved:
                        break
                    hypothesis, rounds, caveat = resolved[0]
                if stop is not None:
                    break
            generation += 1
        scored = self.journal.scored()
        return {
            "stop_reason": stop,
            "counted_experiments": len(scored),
            "converged_score": scored[-1]["metrics"]["primary"] if scored else None,
            "last_experiment_id": scored[-1]["experiment_id"] if scored else None,
            "journal": str(self.journal.path.resolve()),
        }
