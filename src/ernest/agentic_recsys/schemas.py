from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
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

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("hypothesis text cannot be empty")
        if self.parent_experiment_id < 0:
            raise ValueError("parent_experiment_id must be non-negative")
        self.scores.validate()


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
    reasoning: str = ""

    def validate(self) -> None:
        unknown = set(self.active_agents) - set(AGENT_FILES)
        if unknown:
            raise ValueError(f"unknown active agents: {sorted(unknown)}")
        if not self.active_agents:
            raise ValueError("at least one sub-agent must be active")
        if len(self.active_agents) != len(set(self.active_agents)):
            raise ValueError("active_agents cannot contain duplicates")
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
    metrics: Dict[str, float]
    status: str
    failure_reason: Optional[str]
    consultant_rounds: int
    sandbox: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    status: Status
    sandbox: str
    metrics: Dict[str, float] = field(default_factory=dict)
    code_diff: Dict[str, str] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    active_agents: List[str] = field(default_factory=list)


def hypothesis_from_dict(value: Dict[str, Any]) -> Hypothesis:
    raw_scores = value.get("scores", value.get("hypothesis_scores", {}))
    item = Hypothesis(
        text=str(value.get("text", value.get("hypothesis_text", ""))),
        parent_experiment_id=int(value.get("parent_experiment_id", -1)),
        scores=HypothesisScores(
            interestingness=int(raw_scores.get("interestingness", 0)),
            novelty=int(raw_scores.get("novelty", 0)),
            feasibility=int(raw_scores.get("feasibility", 0)),
        ),
        rationale=str(value.get("rationale", "")),
    )
    item.validate()
    return item
