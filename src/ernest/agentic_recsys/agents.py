from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm import LLMClient, LLMError
from .schemas import (
    AGENT_FILES,
    ExperimentPlan,
    FailureReport,
    FIXED_CONTRACT_COMMAND,
    FIXED_TRAIN_COMMAND,
    Hypothesis,
    InterfaceContract,
    Mode,
    hypothesis_from_dict,
)


JSON_ONLY = "Return only one valid JSON object. Do not use Markdown fences."


def _files(path: Path) -> Dict[str, str]:
    return {
        name: (path / name).read_text(encoding="utf-8")
        for names in AGENT_FILES.values() for name in names
        if (path / name).is_file()
    }


def _scored_metric_history(archive: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve complete nested metric objects in execution order for Judge analysis."""
    return [
        {
            "experiment_id": item["experiment_id"],
            "generation": item["generation"],
            "parent_experiment_id": item["parent_experiment_id"],
            "hypothesis_text": item["hypothesis_text"],
            "metrics": item["metrics"],
        }
        for item in archive
        if item.get("status") == "scored"
    ]


class EvolutionJudge:
    def __init__(
        self,
        llm: LLMClient,
        knowledge_documents: Optional[List[Dict[str, str]]] = None,
        metric_catalog: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm
        self.knowledge_documents = knowledge_documents or []
        self.metric_catalog = metric_catalog or {}

    def propose(
        self,
        *,
        mode: Mode,
        generation: int,
        archive: List[Dict[str, Any]],
        reference_snapshots: List[Dict[str, Any]],
        count: int,
        revision_feedback: Optional[str] = None,
        failed_attempt: Optional[Dict[str, Any]] = None,
    ) -> List[Hypothesis]:
        system = (
            "You are the Evolution Judge for a recommender-system MLE loop. Propose concrete, "
            "testable hypotheses, not broad research themes. Honor the fixed within-user ranking "
            "metrics and avoid duplicates in all scored and abandoned history. Treat primary as "
            "the optimization/stopping objective, but use classification, ranking, and data "
            "diagnostics in every scored archive record to diagnose weaknesses and motivate changes. "
            "The full metric catalog explains every available field; do not ignore nested metrics. "
            "In improve mode, "
            "derive exactly one hypothesis from the supplied recent reference. In draft mode, mix "
            "strong and diverse/under-explored references from the full archive. Every proposal "
            "must reference exactly one available experiment ID and score interestingness, novelty, "
            "and feasibility as integers 1-10. " + JSON_ONLY
        )
        payload = {
            "generation": generation,
            "mode": mode.value,
            "requested_count": count,
            "full_archive": archive,
            "scored_metric_history": _scored_metric_history(archive),
            "metric_catalog": self.metric_catalog,
            "research_knowledge_base": self.knowledge_documents,
            "reference_experiments_with_code": reference_snapshots,
            "available_parent_ids": [int(item["experiment_id"]) for item in reference_snapshots],
            "reference_rule": (
                "parent_experiment_id must be chosen from available_parent_ids. It must identify "
                "an already-existing parent, never the experiment currently being proposed."
            ),
            "revision_feedback": revision_feedback,
            "validation_feedback": None,
            "failed_attempt_to_avoid": failed_attempt,
            "response_schema": {
                "hypotheses": [{
                    "text": "string", "parent_experiment_id": 0,
                    "scores": {"interestingness": 1, "novelty": 1, "feasibility": 1},
                    "rationale": "string",
                }]
            },
        }
        allowed = {int(s["experiment_id"]) for s in reference_snapshots}
        last_error = ""
        for response_attempt in range(1, 4):
            payload["response_attempt"] = response_attempt
            payload["validation_feedback"] = last_error or None
            raw = self.llm.complete_json(role="evolution_judge", system=system, payload=payload)
            try:
                proposals = [hypothesis_from_dict(item) for item in raw.get("hypotheses", [])]
                if mode is Mode.IMPROVE and len(proposals) != 1:
                    raise ValueError("Improve mode must return exactly one hypothesis")
                if not 1 <= len(proposals) <= count:
                    raise ValueError(
                        f"Judge returned {len(proposals)} hypotheses; expected 1-{count}"
                    )
                for proposal in proposals:
                    if proposal.parent_experiment_id not in allowed:
                        raise ValueError(
                            f"hypothesis references unavailable experiment "
                            f"{proposal.parent_experiment_id}; available parents are {sorted(allowed)}"
                        )
                if mode is Mode.DRAFT and len(proposals) > 1 and len(allowed) > 1:
                    selected = {proposal.parent_experiment_id for proposal in proposals}

                    def primary_score(item):
                        value = item.get("metrics", {}).get("primary")
                        return float(value) if value is not None else float("-inf")

                    top_id = int(max(reference_snapshots, key=primary_score)["experiment_id"])
                    child_counts = {
                        experiment_id: sum(
                            int(item.get("parent_experiment_id", -1)) == experiment_id
                            for item in archive
                        )
                        for experiment_id in allowed
                    }
                    diverse_candidates = [item for item in allowed if item != top_id] or list(allowed)
                    diverse_id = min(
                        diverse_candidates, key=lambda item: (child_counts[item], item)
                    )
                    if top_id not in selected or diverse_id not in selected:
                        raise ValueError(
                            "multi-hypothesis Draft output must mix the top-scoring and an "
                            "under-explored parent"
                        )
                return proposals
            except (TypeError, ValueError) as exc:
                last_error = f"Response validation failed: {type(exc).__name__}: {exc}"
        raise LLMError(f"Evolution Judge returned invalid proposals three times; {last_error}")

    def revise(
        self,
        *,
        hypothesis: Hypothesis,
        feedback: str,
        archive: List[Dict[str, Any]],
        available_parent_ids: Sequence[int],
    ) -> Hypothesis:
        system = (
            "Revise one rejected recommender-system hypothesis using the consultant feedback. "
            "Keep exactly one parent from available_parent_ids and return the scored replacement. "
            + JSON_ONLY
        )
        raw = self.llm.complete_json(
            role="evolution_judge_revision",
            system=system,
            payload={
                "rejected_hypothesis": asdict(hypothesis),
                "consultant_feedback": feedback,
                "full_archive": archive,
                "scored_metric_history": _scored_metric_history(archive),
                "metric_catalog": self.metric_catalog,
                "research_knowledge_base": self.knowledge_documents,
                "available_parent_ids": list(available_parent_ids),
                "response_schema": {"hypothesis": {
                    "text": "string", "parent_experiment_id": 0,
                    "scores": {"interestingness": 1, "novelty": 1, "feasibility": 1},
                    "rationale": "string",
                }},
            },
        )
        revised = hypothesis_from_dict(raw.get("hypothesis", {}))
        if revised.parent_experiment_id not in available_parent_ids:
            raise LLMError("revision references an unavailable parent")
        return revised


class Consultant:
    def __init__(self, llm: LLMClient, max_rounds: int = 3):
        self.llm = llm
        self.max_rounds = max_rounds

    def review_once(
        self, hypothesis: Hypothesis, archive: List[Dict[str, Any]], round_number: int
    ) -> Dict[str, Any]:
        system = (
            "You are the Consultant. Compare the proposal against the full hypothesis history, "
            "including failures. Reject near-duplicates and structurally infeasible changes. "
            "On the final round, if rejecting, choose final_action 'accept_with_caveat' or 'drop'. "
            + JSON_ONLY
        )
        result = self.llm.complete_json(
            role="consultant",
            system=system,
            payload={
                "hypothesis": asdict(hypothesis),
                "full_archive": archive,
                "round": round_number,
                "max_rounds": self.max_rounds,
                "response_schema": {
                    "accepted": True,
                    "feedback": "string",
                    "final_action": "accept_with_caveat|drop|not_applicable",
                },
            },
        )
        if not isinstance(result.get("accepted"), bool):
            raise LLMError("Consultant response requires boolean accepted")
        result.setdefault("feedback", "")
        result.setdefault("final_action", "not_applicable")
        return result

    def resolve(
        self,
        hypothesis: Hypothesis,
        judge: EvolutionJudge,
        archive: List[Dict[str, Any]],
        available_parent_ids: Sequence[int],
    ) -> Tuple[Optional[Hypothesis], int, Optional[str]]:
        current = hypothesis
        for round_number in range(1, self.max_rounds + 1):
            verdict = self.review_once(current, archive, round_number)
            if verdict["accepted"]:
                return current, round_number, None
            if round_number == self.max_rounds:
                if verdict["final_action"] == "accept_with_caveat":
                    return current, round_number, str(verdict["feedback"])
                return None, round_number, str(verdict["feedback"])
            current = judge.revise(
                hypothesis=current,
                feedback=str(verdict["feedback"]),
                archive=archive,
                available_parent_ids=available_parent_ids,
            )
        return None, self.max_rounds, "revision limit exhausted"


class Orchestrator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, hypothesis: Hypothesis, parent_dir: Path) -> ExperimentPlan:
        system = (
            "You are the experiment Orchestrator. Define a precise interface contract before "
            "delegating and activate only the sub-agents needed. The fixed training command must "
            "produce predictions_valid.npz with exactly row_ids and scores in canonical validation "
            "row order; ground truth is loaded only by the fixed evaluator. Available agents: "
            "feature_engineer(data.py), model_designer(model.py), trainer(train.py/config.json). "
            + JSON_ONLY
        )
        payload = {
                "hypothesis": asdict(hypothesis),
                "parent_files": _files(parent_dir),
                "validation_feedback": None,
                "response_schema": {
                    "active_agents": ["model_designer"],
                    "reasoning": "string",
                    "contract": {
                        "data_output": {"description": "string"},
                        "config_keys": ["seed"],
                        "model_input": {"description": "string"},
                    },
                },
            }
        last_error = ""
        for response_attempt in range(1, 4):
            payload["response_attempt"] = response_attempt
            payload["validation_feedback"] = last_error or None
            raw = self.llm.complete_json(
                role="orchestrator",
                system=system,
                payload=payload,
            )
            try:
                contract_data = dict(raw.get("contract", {}))
                # Execution and evaluation boundaries are system-owned, not LLM-owned.
                contract_data["train_command"] = list(FIXED_TRAIN_COMMAND)
                contract_data["contract_command"] = list(FIXED_CONTRACT_COMMAND)
                contract_data["prediction_artifact"] = {
                    "path": "predictions_valid.npz",
                    "arrays": ["row_ids", "scores"],
                }
                plan = ExperimentPlan(
                    active_agents=list(raw.get("active_agents", [])),
                    contract=InterfaceContract.from_dict(contract_data),
                    reasoning=str(raw.get("reasoning", "")),
                )
                plan.validate()
                return plan
            except (TypeError, ValueError) as exc:
                last_error = f"Response validation failed: {type(exc).__name__}: {exc}"
        raise LLMError(f"Orchestrator returned invalid plans three times; {last_error}")

    def generate_patches(
        self,
        *,
        agent: str,
        hypothesis: Hypothesis,
        plan: ExperimentPlan,
        sandbox: Path,
        failure: Optional[FailureReport] = None,
    ) -> Dict[str, str]:
        allowed_files = AGENT_FILES[agent]
        system = (
            f"You are the {agent}. Modify only {list(allowed_files)} and return unified diffs "
            "against the supplied current files. Preserve all unrelated working behavior. Never "
            "write evaluation logic. Patches must use exact file names in both --- and +++ headers. "
            + JSON_ONLY
        )
        raw = self.llm.complete_json(
            role=agent if failure is None else f"{agent}_debug",
            system=system,
            payload={
                "hypothesis": asdict(hypothesis),
                "interface_contract": asdict(plan.contract),
                "current_files": {
                    name: (sandbox / name).read_text(encoding="utf-8") for name in allowed_files
                },
                "structured_error": None if failure is None else asdict(failure),
                "response_schema": {"patches": {name: "unified diff string" for name in allowed_files}},
            },
        )
        patches = raw.get("patches")
        if not isinstance(patches, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in patches.items()):
            raise LLMError(f"{agent} must return a string-to-string patches object")
        return patches
