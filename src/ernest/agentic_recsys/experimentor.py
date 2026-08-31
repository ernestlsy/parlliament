from __future__ import annotations

import ast
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

from .diagnostics import attach_segment_diagnostics
from .evaluation import score_prediction_artifact, validate_prediction_artifact
from .schemas import AGENT_FILES, ExperimentPlan, FailureKind, FailureReport


class Experimentor:
    """Runs generated code, but owns no retry policy and cannot modify evaluation."""

    def __init__(self, python_executable: str, data_dir: str):
        self.python_executable = python_executable
        self.data_dir = data_dir

    @staticmethod
    def _agent_for_file(filename: str) -> str:
        for agent, files in AGENT_FILES.items():
            if filename in files:
                return agent
        return "trainer"

    def preflight(self, sandbox: Path, plan: ExperimentPlan, attempt: int) -> Optional[FailureReport]:
        for filename in ("data.py", "model.py", "train.py", "config.json"):
            if not (sandbox / filename).is_file():
                return FailureReport(
                    FailureKind.CONTRACT_FULFILLMENT, f"missing required file {filename}", "",
                    [self._agent_for_file(filename)], attempt,
                )
        try:
            config = json.loads((sandbox / "config.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return FailureReport(
                FailureKind.CONTRACT_FULFILLMENT, f"invalid config.json: {exc}", repr(exc),
                ["trainer"], attempt,
            )
        missing = [key for key in plan.contract.config_keys if key not in config]
        if missing:
            return FailureReport(
                FailureKind.CONTRACT_FULFILLMENT,
                f"config does not fulfill contract; missing {missing}", "",
                ["trainer"], attempt,
            )
        if config.get("split") != "valid":
            return FailureReport(
                FailureKind.CONTRACT_FULFILLMENT,
                "config split must remain 'valid'; test labels cannot control model selection", "",
                ["trainer"], attempt,
            )
        for filename in ("data.py", "model.py", "train.py"):
            try:
                ast.parse((sandbox / filename).read_text(encoding="utf-8"), filename=filename)
            except (SyntaxError, UnicodeError, OSError) as exc:
                return FailureReport(
                    FailureKind.SEMANTIC, f"syntax error in {filename}", str(exc),
                    [self._agent_for_file(filename)], attempt,
                )
        return None

    @staticmethod
    def classify_process_failure(stderr: str, attempt: int, return_code: int) -> FailureReport:
        text = stderr.lower()
        if any(token in text for token in ("out of memory", "cuda oom", "memoryerror", "killed")):
            kind, agents = FailureKind.RESOURCE, ["trainer"]
        elif any(token in text for token in ("shape", "dimension", "dtype", "keyerror", "missing key")):
            kind, agents = FailureKind.CONTRACT_USAGE, ["feature_engineer", "model_designer", "trainer"]
        else:
            kind, agents = FailureKind.SEMANTIC, ["trainer"]
            if "model.py" in text:
                agents = ["model_designer"]
            elif "data.py" in text:
                agents = ["feature_engineer"]
        return FailureReport(kind, f"training process exited {return_code}", stderr[-12000:], agents, attempt, return_code)

    def run(
        self, sandbox: Path, plan: ExperimentPlan, *, attempt: int, timeout_seconds: float
    ) -> Tuple[Optional[dict], Optional[FailureReport]]:
        preflight_failure = self.preflight(sandbox, plan, attempt)
        if preflight_failure:
            return None, preflight_failure
        output = sandbox / plan.contract.prediction_artifact["path"]
        substitutions = {
            "{python}": self.python_executable,
            "{data_dir}": self.data_dir,
            "{output}": str(output),
        }
        command = [substitutions.get(arg, arg) for arg in plan.contract.train_command]
        contract_command = [substitutions.get(arg, arg) for arg in plan.contract.contract_command]
        if command[0] != self.python_executable or contract_command[0] != self.python_executable:
            return None, FailureReport(
                FailureKind.CONTRACT_USAGE, "train command must use {python}", "",
                ["trainer"], attempt,
            )
        started = time.monotonic()
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            probe = subprocess.run(
                contract_command, cwd=sandbox, text=True, capture_output=True,
                timeout=max(0.1, timeout_seconds), check=False, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            trace = ((exc.stdout or "") + "\n" + (exc.stderr or ""))[-12000:]
            return None, FailureReport(
                FailureKind.TIMEOUT, "contract probe exceeded wall-clock budget", trace,
                ["trainer"], attempt,
            )
        probe_log = f"command: {contract_command!r}\n--- stdout ---\n{probe.stdout}\n--- stderr ---\n{probe.stderr}"
        (sandbox / f"contract_attempt_{attempt}.log").write_text(probe_log, encoding="utf-8")
        if probe.returncode:
            report = self.classify_process_failure(probe.stderr, attempt, probe.returncode)
            if report.kind is FailureKind.SEMANTIC:
                report.kind = FailureKind.CONTRACT_USAGE
                report.responsible_agents = ["feature_engineer", "model_designer", "trainer"]
            report.message = f"contract probe exited {probe.returncode}"
            return None, report
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            return None, FailureReport(
                FailureKind.TIMEOUT, "contract probe exhausted wall-clock budget", "",
                ["trainer"], attempt,
            )
        try:
            proc = subprocess.run(
                command, cwd=sandbox, text=True, capture_output=True,
                timeout=max(0.1, remaining), check=False, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            trace = ((exc.stdout or "") + "\n" + (exc.stderr or ""))[-12000:]
            (sandbox / f"attempt_{attempt}.log").write_text(trace, encoding="utf-8")
            return None, FailureReport(
                FailureKind.TIMEOUT, "experiment exceeded wall-clock budget", trace,
                ["trainer"], attempt,
            )
        log = (
            f"command: {command!r}\nelapsed_seconds: {time.monotonic() - started:.3f}\n"
            f"return_code: {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        (sandbox / f"attempt_{attempt}.log").write_text(log, encoding="utf-8")
        if proc.returncode:
            return None, self.classify_process_failure(proc.stderr, attempt, proc.returncode)
        try:
            metrics = score_prediction_artifact(output, Path(self.data_dir), "valid")
            validate_prediction_artifact(
                sandbox / "predictions_test.npz", Path(self.data_dir), "test"
            )
        except (ValueError, OSError) as exc:
            return None, FailureReport(
                FailureKind.CONTRACT_FULFILLMENT,
                f"invalid prediction artifact: {exc}", repr(exc), ["trainer"], attempt,
            )
        try:
            metrics = attach_segment_diagnostics(
                metrics,
                output,
                Path(self.data_dir),
                sandbox / "segment_diagnostics.json",
            )
        except Exception as exc:
            # Diagnostics are advisory and must never invalidate a correctly scored experiment.
            metrics["segment_diagnostics"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        (sandbox / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics, None
