from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional


class Mode(str, Enum):
    IMPROVE = "improve"
    DRAFT = "draft"


class Status(str, Enum):
    SCORED = "scored"
    ABANDONED = "abandoned"


class FailureKind(str, Enum):
    SEMANTIC = "semantic_logic"
    CONTRACT_FULFILLMENT = "contract_fulfillment"
    CONTRACT_USAGE = "contract_usage"
    RESOURCE = "resource_transient"
    TIMEOUT = "timeout"


AGENT_FILES = {
    "feature_engineer": ("data.py",),
    "model_designer": ("model.py",),
    "trainer": ("train.py", "config.json"),
}

FIXED_TRAIN_COMMAND = [
    "{python}", "train.py", "--config", "config.json",
    "--data-dir", "{data_dir}", "--output", "{output}",
]
FIXED_CONTRACT_COMMAND = FIXED_TRAIN_COMMAND + ["--contract-check"]


@dataclass(frozen=True)
class HypothesisScores:
    interestingness: int
    novelty: int
    feasibility: int

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if not isinstance(value, int) or not 1 <= value <= 10:
                raise ValueError(f"{key} must be an integer from 1 to 10")


@dataclass(frozen=True)
class Hypothesis:
    text: str
    parent_experiment_id: int
    scores: HypothesisScores
    rationale: str = ""
    candidate_id: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    exact_ablation: str = ""
    expected_effect: Dict[str, Any] = field(default_factory=dict)
    expected_primary_gain: Optional[float] = None
    confidence: Optional[int] = None
    leakage_risk: str = ""
    runtime_risk: str = ""
    active_components: List[str] = field(default_factory=list)
    literature_document_ids: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("hypothesis text cannot be empty")
        if self.parent_experiment_id < 0:
            raise ValueError("parent_experiment_id must be non-negative")
        self.scores.validate()

    def validate_tournament_candidate(self, available_evidence_ids: set[str]) -> None:
        self.validate()
        if not self.candidate_id.strip():
            raise ValueError("tournament candidate_id cannot be empty")
        if not self.evidence_ids or not set(self.evidence_ids).issubset(available_evidence_ids):
            unknown = sorted(set(self.evidence_ids) - available_evidence_ids)
            raise ValueError(f"candidate must cite available evidence; unknown={unknown}")
        if not self.exact_ablation.strip():
            raise ValueError("candidate requires one exact ablation")
        if self.expected_primary_gain is None or not math.isfinite(self.expected_primary_gain):
            raise ValueError("candidate requires a finite expected_primary_gain")
        if self.confidence is None or not 1 <= self.confidence <= 10:
            raise ValueError("candidate confidence must be an integer from 1 to 10")
        if self.leakage_risk not in {"low", "medium", "high"}:
            raise ValueError("candidate leakage_risk must be low, medium, or high")
        if self.runtime_risk not in {"low", "medium", "high"}:
            raise ValueError("candidate runtime_risk must be low, medium, or high")
        if not {"GAUC", "nDCG@5"}.issubset(self.expected_effect):
            raise ValueError("candidate expected_effect requires GAUC and nDCG@5")
        # This is advisory planning metadata. The Orchestrator independently validates and chooses
        # the actual AGENT_FILES roles, so conceptual values such as "loss" are allowed here.


@dataclass
class InterfaceContract:
    data_output: Dict[str, Any]
    config_keys: List[str]
    model_input: Dict[str, Any]
    prediction_artifact: Dict[str, Any] = field(
        default_factory=lambda: {
            "path": "predictions_valid.npz",
            "arrays": ["row_ids", "scores"],
        }
    )
    train_command: List[str] = field(
        default_factory=lambda: list(FIXED_TRAIN_COMMAND)
    )
    contract_command: List[str] = field(
        default_factory=lambda: list(FIXED_CONTRACT_COMMAND)
    )

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "InterfaceContract":
        required = ("data_output", "config_keys", "model_input")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"interface contract missing keys: {missing}")
        contract = cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})
        contract.validate()
        return contract

    def validate(self) -> None:
        if not self.config_keys or not all(isinstance(x, str) for x in self.config_keys):
            raise ValueError("config_keys must be a non-empty string list")
        if self.train_command != FIXED_TRAIN_COMMAND:
            raise ValueError("train_command is fixed and cannot be changed by agents")
        if self.contract_command != FIXED_CONTRACT_COMMAND:
            raise ValueError("contract_command is fixed and cannot be changed by agents")
        if self.prediction_artifact.get("path") != "predictions_valid.npz":
            raise ValueError("prediction artifact path is fixed to predictions_valid.npz")
        if self.prediction_artifact.get("arrays") != ["row_ids", "scores"]:
            raise ValueError("prediction artifact arrays are fixed")


@dataclass
class ExperimentPlan:
    active_agents: List[str]
    contract: InterfaceContract
    agent_instructions: Dict[str, Dict[str, Any]]
    reasoning: str = ""

    def validate(self) -> None:
        unknown = set(self.active_agents) - set(AGENT_FILES)
        if unknown:
            raise ValueError(f"unknown active agents: {sorted(unknown)}")
        if not self.active_agents:
            raise ValueError("at least one sub-agent must be active")
        if len(self.active_agents) != len(set(self.active_agents)):
            raise ValueError("active_agents cannot contain duplicates")
        if not isinstance(self.agent_instructions, dict):
            raise ValueError("agent_instructions must be an object")
        instructed = set(self.agent_instructions)
        active = set(self.active_agents)
        if instructed != active:
            raise ValueError(
                "agent_instructions must contain exactly the active agents; "
                f"missing={sorted(active-instructed)}, inactive={sorted(instructed-active)}"
            )
        for agent, instruction in self.agent_instructions.items():
            if not isinstance(instruction, dict):
                raise ValueError(f"instruction for {agent} must be an object")
            objective = instruction.get("objective")
            if not isinstance(objective, str) or not objective.strip():
                raise ValueError(f"instruction for {agent} requires a non-empty objective")
            for field_name in ("required_changes", "preserve", "coordination_notes"):
                values = instruction.get(field_name, [])
                if not isinstance(values, list) or not all(
                    isinstance(value, str) and value.strip() for value in values
                ):
                    raise ValueError(
                        f"instruction for {agent} field {field_name} must be a string list"
                    )
            if not instruction.get("required_changes"):
                raise ValueError(f"instruction for {agent} requires required_changes")
        self.contract.validate()


@dataclass
class FailureReport:
    kind: FailureKind
    message: str
    traceback: str
    responsible_agents: List[str]
    attempt: int
    return_code: Optional[int] = None


@dataclass
class JournalRecord:
    attempt_id: str
    experiment_id: Optional[int]
    generation: int
    parent_experiment_id: int
    hypothesis_text: str
    hypothesis_scores: Dict[str, int]
    mode: str
    code_diff: Dict[str, str]
    config_diff: str
    active_sub_agents: List[str]
    metrics: Dict[str, Any]
    status: str
    failure_reason: Optional[str]
    failure_stage: Optional[str]
    consultant_rounds: int
    sandbox: str
    created_at: str
    hypothesis_prediction: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    status: Status
    sandbox: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    code_diff: Dict[str, str] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    active_agents: List[str] = field(default_factory=list)


def hypothesis_from_dict(value: Dict[str, Any]) -> Hypothesis:
    raw_scores = value.get("scores", value.get("hypothesis_scores", {}))
    raw_confidence = value.get("confidence")
    confidence = None
    if raw_confidence is not None:
        numeric_confidence = float(raw_confidence)
        # Models commonly express confidence as a probability despite a 1-10 schema.
        if 0.0 <= numeric_confidence <= 1.0:
            confidence = max(1, min(10, int(round(numeric_confidence * 10))))
        else:
            confidence = int(round(numeric_confidence))
    item = Hypothesis(
        text=str(value.get("text", value.get("hypothesis_text", ""))),
        parent_experiment_id=int(value.get("parent_experiment_id", -1)),
        scores=HypothesisScores(
            interestingness=int(raw_scores.get("interestingness", 0)),
            novelty=int(raw_scores.get("novelty", 0)),
            feasibility=int(raw_scores.get("feasibility", 0)),
        ),
        rationale=str(value.get("rationale", "")),
        candidate_id=str(value.get("candidate_id", "")),
        evidence_ids=[str(item) for item in value.get("evidence_ids", [])],
        exact_ablation=str(value.get("exact_ablation", "")),
        expected_effect=dict(value.get("expected_effect", {})),
        expected_primary_gain=(
            None if value.get("expected_primary_gain") is None
            else float(value["expected_primary_gain"])
        ),
        confidence=confidence,
        leakage_risk=str(value.get("leakage_risk", "")),
        runtime_risk=str(value.get("runtime_risk", "")),
        active_components=[str(item) for item in value.get("active_components", [])],
        literature_document_ids=[
            str(item) for item in value.get("literature_document_ids", [])
        ],
    )
    item.validate()
    return item
