from __future__ import annotations

import difflib
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agents import Consultant, EvolutionJudge, FeatureAnalyst, Orchestrator
from .config import SystemConfig
from .experimentor import Experimentor
from .evaluation import METRIC_CATALOG
from .journal import Journal
from .knowledge import KnowledgeBase
from .llm import AuditedLLMClient, LLMClient
from .research import ResearchEngine, ScreeningConfig
from .sandbox import (
    apply_agent_replacements,
    config_diff,
    create_attempt_sandbox,
    finalize_sandbox,
)
from .schemas import (
    AGENT_FILES, FailureKind, FailureReport, Hypothesis, JournalRecord, Mode, Status,
)


class Overseer:
    """Sequential owner of generation scheduling, retries, counting, and stopping."""

    def __init__(self, config: SystemConfig, llm: LLMClient):
        self.config = config
        self.journal = Journal(config.run_dir / "journal.jsonl")
        self.llm = AuditedLLMClient(llm, config.run_dir / "llm_events.jsonl")
        knowledge = KnowledgeBase(Path(__file__).parent / "knowledge").documents()
        self.judge = EvolutionJudge(self.llm, knowledge, METRIC_CATALOG)
        self.feature_analyst = FeatureAnalyst(self.llm)
        self.consultant = Consultant(self.llm, config.max_consultant_rounds)
        self.orchestrator = Orchestrator(self.llm)
        self.experimentor = Experimentor(config.python_executable, config.data_dir)
        self.seed_dir = Path(__file__).parent / "seed"
        self.research = ResearchEngine(
            Path(config.data_dir),
            config.run_dir,
            ScreeningConfig(
                holdout_fraction=config.screening_holdout_fraction,
                seed=config.screening_seed,
                timeout_seconds=config.screening_timeout_seconds,
                force=config.force_rescreen,
            ),
        )

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
            "hypothesis_text": "Neutral unscored additive user/item scaffold; fresh starting point",
            "hypothesis_scores": None,
            "mode": "seed", "code_diff": {}, "config_diff": "",
            "active_sub_agents": [],
            "metrics": {
                "availability": "unscored",
                "reason": "Seed is a code scaffold, not a prior experiment or published baseline.",
            },
            "status": "seed", "failure_reason": None, "consultant_rounds": 0,
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
        seed = archive[0]
        scored = [r for r in archive if r["status"] == "scored"]
        if mode is Mode.IMPROVE:
            if not scored:
                return [seed]
            return [max(scored, key=lambda r: float(r["metrics"]["primary"]))]
        return [seed] + scored

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

    def _current_judge_context(self, mode: Mode):
        """Recompute eligible parents from the latest durable archive for every Judge call."""
        archive = self._archive()
        references = self._reference_records(mode, archive)
        snapshots = [self._snapshot(record) for record in references]
        parent_ids = [int(record["experiment_id"]) for record in references]
        return archive, references, snapshots, parent_ids

    def _consult(
        self, proposals: Sequence[Hypothesis], archive: List[Dict[str, Any]], parent_ids: Sequence[int]
    ) -> List[Tuple[Hypothesis, int, Optional[str]]]:
        accepted = []
        review_archive = list(archive)
        for proposal in proposals:
            self.llm.set_context(
                phase="consultant_review",
                hypothesis_text=proposal.text,
                parent_experiment_id=proposal.parent_experiment_id,
            )
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

    def _run_tournament(
        self,
        *,
        generation: int,
        mode: Mode,
        archive: List[Dict[str, Any]],
        snapshots: List[Dict[str, Any]],
    ) -> Tuple[Hypothesis, int, Optional[str]]:
        """Spend planning calls, not scored iterations, to choose one experiment."""
        research_brief = self.research.build_brief(archive)
        evidence_ids = self.research.evidence_ids(archive)
        planning_dir = self.config.run_dir / "planning" / f"generation_{generation}"
        planning_dir.mkdir(parents=True, exist_ok=True)

        self.llm.set_context(
            phase="feature_analysis", generation=generation, mode=mode.value
        )
        assessment = self.feature_analyst.analyze(
            research_brief=research_brief,
            available_evidence_ids=evidence_ids,
        )
        (planning_dir / "feature_analysis.json").write_text(
            json.dumps(assessment, indent=2), encoding="utf-8"
        )

        self.llm.set_context(
            phase="candidate_generation", generation=generation, mode=mode.value
        )
        candidates = self.judge.generate_candidates(
            mode=mode,
            generation=generation,
            archive=archive,
            reference_snapshots=snapshots,
            candidate_count=self.config.candidate_pool_size,
            research_brief=research_brief,
            analyst_assessment=assessment,
            available_evidence_ids=evidence_ids,
        )
        (planning_dir / "candidates.json").write_text(
            json.dumps([asdict(item) for item in candidates], indent=2), encoding="utf-8"
        )

        self.llm.set_context(
            phase="candidate_ranking", generation=generation, mode=mode.value
        )
        ranking = self.consultant.rank_candidates(
            candidates=candidates,
            archive=archive,
            research_brief=research_brief,
        )
        (planning_dir / "consultant_ranking.json").write_text(
            json.dumps(ranking, indent=2), encoding="utf-8"
        )

        self.llm.set_context(
            phase="candidate_selection", generation=generation, mode=mode.value
        )
        winner, rationale = self.judge.select_winner(
            candidates=candidates,
            ranking=ranking,
            research_brief=research_brief,
        )
        selection = {
            "winner": asdict(winner),
            "selection_rationale": rationale,
            "counted_experiments_consumed": 0,
        }
        (planning_dir / "selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
        return winner, 1, None

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
        replacement_history: List[Dict[str, Any]] = []
        failure_reports: List[Dict[str, Any]] = []
        active_agents: List[str] = []
        failure_reason: Optional[str] = None
        failure_stage: Optional[str] = None
        exception_traceback: Optional[str] = None
        last_failure = None
        started = time.monotonic()
        audit_context = {
            "attempt_id": attempt_id,
            "attempt_dir": str(sandbox),
            "generation": generation,
            "mode": mode.value,
            "hypothesis_text": hypothesis.text,
            "parent_experiment_id": hypothesis.parent_experiment_id,
        }

        def request_and_apply_replacements(
            agent: str, *, phase: str, failure=None
        ) -> None:
            last_exception = None
            repair_failure = failure
            for replacement_attempt in range(1, 4):
                self.llm.set_context(
                    **audit_context,
                    phase=phase,
                    agent=agent,
                    replacement_attempt=replacement_attempt,
                )
                try:
                    replacements = self.orchestrator.generate_file_replacements(
                        agent=agent,
                        hypothesis=hypothesis,
                        plan=plan,
                        sandbox=sandbox,
                        failure=repair_failure,
                    )
                    replacement_history.append({
                        "format_version": 2,
                        "phase": phase,
                        "agent": agent,
                        "replacement_attempt": replacement_attempt,
                        "files": replacements,
                    })
                    apply_agent_replacements(sandbox, replacements, AGENT_FILES[agent])
                    return
                except Exception as exc:
                    last_exception = exc
                    repair_failure = FailureReport(
                        kind=FailureKind.CONTRACT_FULFILLMENT,
                        message=(
                            f"{agent} complete-file response was invalid: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        traceback=traceback.format_exc()[-12000:],
                        responsible_agents=[agent],
                        attempt=replacement_attempt,
                    )
                    failure_reports.append(asdict(repair_failure))
            raise RuntimeError(
                f"{agent} failed to provide valid complete files after three responses: "
                f"{type(last_exception).__name__}: {last_exception}"
            ) from last_exception

        try:
            (sandbox / "hypothesis.json").write_text(
                json.dumps(asdict(hypothesis), indent=2), encoding="utf-8"
            )
            failure_stage = "orchestrator_plan"
            self.llm.set_context(**audit_context, phase=failure_stage)
            plan = self.orchestrator.plan(hypothesis, parent_dir)
            active_agents = list(plan.active_agents)
            (sandbox / "interface_contract.json").write_text(
                json.dumps(asdict(plan.contract), indent=2), encoding="utf-8"
            )
            for agent in active_agents:
                failure_stage = f"initial_replacement:{agent}"
                request_and_apply_replacements(agent, phase=failure_stage)

            resource_failures = 0
            metrics = None
            for debug_attempt in range(1, self.config.max_debug_attempts + 1):
                failure_stage = f"experiment_execution:{debug_attempt}"
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
                failure_reports.append(asdict(failure))
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
                    failure_stage = f"debug_replacement:{debug_attempt}:{agent}"
                    request_and_apply_replacements(agent, phase=failure_stage, failure=failure)
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
                failure_stage = None
                experiment_id = self.journal.next_experiment_id()
                final_path = finalize_sandbox(
                    sandbox, self.config.run_dir,
                    experiment_id=experiment_id, attempt_id=attempt_id,
                )
                status = Status.SCORED
        except Exception as exc:
            exception_traceback = traceback.format_exc()
            failure_reason = (
                f"stage={failure_stage or 'unknown'}; agent_or_guardrail_failure: "
                f"{type(exc).__name__}: {exc}"
            )
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
            failure_stage=failure_stage if status is Status.ABANDONED else None,
            consultant_rounds=consultant_rounds,
            sandbox=str(final_path.resolve()),
            created_at=datetime.now(timezone.utc).isoformat(),
            hypothesis_prediction={
                "candidate_id": hypothesis.candidate_id,
                "evidence_ids": hypothesis.evidence_ids,
                "expected_effect": hypothesis.expected_effect,
                "expected_primary_gain": hypothesis.expected_primary_gain,
                "confidence": hypothesis.confidence,
                "leakage_risk": hypothesis.leakage_risk,
                "runtime_risk": hypothesis.runtime_risk,
                "exact_ablation": hypothesis.exact_ablation,
            },
        )
        (final_path / "patch_history.json").write_text(
            json.dumps(replacement_history, indent=2, default=str), encoding="utf-8"
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
        log_files = sorted(path.name for path in final_path.glob("*.log"))
        summary = {
            "attempt_id": attempt_id,
            "status": status.value,
            "generation": generation,
            "mode": mode.value,
            "parent_experiment_id": hypothesis.parent_experiment_id,
            "hypothesis_text": hypothesis.text,
            "active_sub_agents": active_agents,
            "consultant_rounds": consultant_rounds,
            "failure_stage": record.failure_stage,
            "failure_reason": failure_reason,
            "failure_reports": failure_reports,
            "elapsed_seconds": time.monotonic() - started,
            "available_log_files": log_files,
            "llm_log": "llm_events.jsonl" if (final_path / "llm_events.jsonl").exists() else None,
            "patch_history": "patch_history.json",
            "exception_traceback": exception_traceback,
        }
        (final_path / "attempt_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        readable = [
            f"Attempt: {attempt_id}",
            f"Status: {status.value}",
            f"Generation: {generation} ({mode.value})",
            f"Failure stage: {record.failure_stage or '-'}",
            f"Failure reason: {failure_reason or '-'}",
            f"Hypothesis: {hypothesis.text}",
            f"Active agents: {', '.join(active_agents) or '-'}",
            f"Elapsed seconds: {summary['elapsed_seconds']:.3f}",
            f"Other logs: {', '.join(log_files) or '-'}",
            "",
            "Structured failure reports:",
            json.dumps(failure_reports, indent=2, default=str),
        ]
        if exception_traceback:
            readable.extend(["", "Full exception traceback:", exception_traceback])
        if status is Status.ABANDONED:
            (final_path / "failure.log").write_text("\n".join(readable), encoding="utf-8")
        self.journal.append(record)
        return record

    def run(self) -> Dict[str, Any]:
        self.initialize()
        stop = self.journal.stop_reason(
            self.config.max_experiments,
            self.config.convergence_epsilon,
            self.config.convergence_window,
        )
        if stop is None:
            self.llm.set_context(phase="train_only_feature_screening")
            self.research.ensure()
        generation = self.journal.latest_generation() + 1
        while stop is None:
            mode = Mode.IMPROVE if generation % 2 == 0 else Mode.DRAFT
            archive, _references, snapshots, _parent_ids = self._current_judge_context(mode)
            winner = self._run_tournament(
                generation=generation,
                mode=mode,
                archive=archive,
                snapshots=snapshots,
            )
            for hypothesis, rounds, caveat in [winner]:
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
                    (
                        archive,
                        _replacement_references,
                        replacement_snapshots,
                        replacement_parent_ids,
                    ) = self._current_judge_context(mode)
                    self.llm.set_context(
                        phase="replacement_hypothesis",
                        generation=generation,
                        mode=mode.value,
                        failed_attempt_id=record.attempt_id,
                    )
                    replacement = self.judge.propose(
                        mode=mode,
                        generation=generation,
                        archive=archive,
                        reference_snapshots=replacement_snapshots,
                        count=1,
                        failed_attempt=record.to_dict(),
                    )[0]
                    resolved = self._consult(
                        [replacement], archive, replacement_parent_ids
                    )
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
            "screening_report": str(self.research.report_path.resolve()),
        }
