"""Build a self-contained submission bundle from one Ernest run."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .journal import Journal


OFFICIAL_VALID_BASELINE = {
    "GAUC": 0.6674,
    "nDCG@5": 0.5357,
    "primary": 0.6016,
}
SUBMISSION_HEADER = ["row_id", "user_id", "video_id", "score"]


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _canonical_rows(data_dir: Path, split: str) -> List[Tuple[str, str]]:
    ranges = {"valid": (20220422, 20220428), "test": (20220429, 20220508)}
    if split not in ranges:
        raise ValueError(f"unsupported submission split: {split}")
    low, high = ranges[split]
    rows: List[Tuple[str, str]] = []
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with (data_dir / filename).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if low <= int(row["date"]) <= high:
                    rows.append((row["user_id"], row["video_id"]))
    return rows


def write_submission_csv(artifact_path: Path, data_dir: Path, output_path: Path) -> int:
    """Convert a canonical test NPZ artifact to the Starter Kit CSV schema."""
    if not artifact_path.is_file():
        raise ValueError(f"test prediction artifact is missing: {artifact_path}")
    import numpy as np

    rows = _canonical_rows(data_dir, "test")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"row_ids", "scores"}:
            raise ValueError("test prediction arrays must be exactly row_ids and scores")
        row_ids = artifact["row_ids"]
        scores = artifact["scores"]
        if row_ids.ndim != 1 or scores.ndim != 1:
            raise ValueError("test row_ids and scores must be one-dimensional")
        if not np.issubdtype(row_ids.dtype, np.integer):
            raise ValueError("test row_ids must use an integer dtype")
        if len(row_ids) != len(rows) or len(scores) != len(rows):
            raise ValueError(
                f"test prediction rows ({len(scores)}) do not match canonical rows ({len(rows)})"
            )
        if not np.array_equal(row_ids, np.arange(len(rows))):
            raise ValueError("test row_ids must be consecutive and in canonical test order")
        if not np.all(np.isfinite(scores)):
            raise ValueError("test scores contain NaN or infinity")
        score_values = [float(value) for value in scores]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_HEADER)
        for row_id, ((user_id, video_id), score) in enumerate(zip(rows, score_values)):
            writer.writerow([row_id, user_id, video_id, format(score, ".17g")])
    return len(rows)


def _portable_sandbox(run_dir: Path, record: Dict[str, Any]) -> Path:
    experiment_id = record.get("experiment_id")
    if experiment_id is not None:
        return run_dir / f"experiment_{experiment_id}"
    return run_dir / "abandoned" / str(record.get("attempt_id", ""))


def _attempt_details(run_dir: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    sandbox = _portable_sandbox(run_dir, record)
    summary = _read_json(sandbox / "attempt_summary.json", {}) if sandbox.name else {}
    submission_metrics = {
        key: value
        for key, value in dict(record.get("metrics", {})).items()
        if key != "segment_diagnostics"
    }
    recovery_events = summary.get("failure_reports", [])
    if not isinstance(recovery_events, list):
        recovery_events = []
    return {
        "attempt_id": record.get("attempt_id"),
        "experiment_id": record.get("experiment_id"),
        "generation": record.get("generation"),
        "status": record.get("status"),
        "parent_experiment_id": record.get("parent_experiment_id"),
        "hypothesis": record.get("hypothesis_text", ""),
        "hypothesis_rationale": record.get("hypothesis_prediction", {}),
        "code_diff": record.get("code_diff", {}),
        "config_diff": record.get("config_diff", ""),
        "metrics": submission_metrics,
        "failure_stage": record.get("failure_stage"),
        "failure_reason": record.get("failure_reason"),
        "recovery_events": recovery_events,
        "recovery_action": (
            "The Overseer classified each failure, routed it to the responsible code agent, "
            "and retried within the configured attempt and wall-clock limits."
            if recovery_events else "No error or recovery event occurred in this attempt."
        ),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "sandbox": str(sandbox.resolve()),
    }


def _usage_summary(path: Path) -> Dict[str, Any]:
    total_input = total_output = total = 0
    calls = calls_with_usage = failed_calls = 0
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                calls += 1
                failed_calls += event.get("status") == "error"
                usage = event.get("usage")
                if not isinstance(usage, dict):
                    continue
                calls_with_usage += 1
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
                total_input += input_tokens
                total_output += output_tokens
                total += total_tokens
    complete = calls > 0 and calls_with_usage == calls - failed_calls
    return {
        "input_tokens": total_input if calls_with_usage else None,
        "output_tokens": total_output if calls_with_usage else None,
        "total_tokens": total if calls_with_usage else None,
        "llm_calls": calls,
        "failed_llm_calls": failed_calls,
        "calls_with_usage": calls_with_usage,
        "complete": complete,
        "note": (
            "Provider-reported usage covers every successful LLM call."
            if complete else
            "Token usage is incomplete because one or more provider/client events did not report usage."
        ),
    }


def _markdown_log(attempts: Iterable[Dict[str, Any]]) -> str:
    lines = [
        "# Per-iteration run log",
        "",
        "Manual interventions: **0 (none)**.",
        "",
    ]
    for item in attempts:
        label = (
            f"Experiment {item['experiment_id']}"
            if item["experiment_id"] is not None else
            f"Abandoned attempt {item['attempt_id']}"
        )
        metrics = item.get("metrics", {})
        lines.extend([
            f"## {label}", "",
            f"- Generation: {item['generation']}",
            f"- Parent experiment: {item['parent_experiment_id']}",
            f"- Status: {item['status']}",
            f"- Hypothesis: {item['hypothesis']}",
            f"- Validation GAUC: {metrics.get('GAUC', 'N/A')}",
            f"- Validation nDCG@5: {metrics.get('nDCG@5', 'N/A')}",
            f"- Validation primary: {metrics.get('primary', 'N/A')}",
            f"- Failure stage: {item.get('failure_stage') or 'none'}",
            f"- Failure reason: {item.get('failure_reason') or 'none'}",
            f"- Recovery: {item['recovery_action']}",
            "", "### Code diff", "",
        ])
        diffs = dict(item.get("code_diff", {}))
        if item.get("config_diff"):
            diffs["config.json"] = item["config_diff"]
        if not diffs:
            lines.append("No final code change was applied.")
        for filename, diff in diffs.items():
            lines.extend(["", f"#### `{filename}`", "", "```diff", str(diff).rstrip(), "```"])
        if item.get("recovery_events"):
            lines.extend([
                "", "### Error and recovery events", "", "```json",
                json.dumps(item["recovery_events"], indent=2, default=str), "```",
            ])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_submission_bundle(run_dir: Path, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Write logs, results, resource usage, and the validation-best test submission."""
    run_dir = run_dir.resolve()
    config = _read_json(run_dir / "system_config.json", {})
    resolved_data_dir = Path(data_dir or config.get("data_dir", "")).resolve()
    records = Journal(run_dir / "journal.jsonl").records()
    scored = [record for record in records if record.get("status") == "scored"]
    if not scored:
        raise ValueError("cannot build a submission bundle without a scored experiment")
    best = max(scored, key=lambda record: float(record["metrics"]["primary"]))
    attempts = [_attempt_details(run_dir, record) for record in records]
    usage = _usage_summary(run_dir / "llm_events.jsonl")
    timing = _read_json(run_dir / "run_timing.json", {})
    metrics = best["metrics"]
    result_row = {
        "benchmark": "KuaiRand-Pure",
        "best_experiment_id": best["experiment_id"],
        "validation_GAUC": float(metrics["GAUC"]),
        "baseline_GAUC": OFFICIAL_VALID_BASELINE["GAUC"],
        "absolute_delta_GAUC": float(metrics["GAUC"]) - OFFICIAL_VALID_BASELINE["GAUC"],
        "validation_nDCG@5": float(metrics["nDCG@5"]),
        "baseline_nDCG@5": OFFICIAL_VALID_BASELINE["nDCG@5"],
        "absolute_delta_nDCG@5": float(metrics["nDCG@5"]) - OFFICIAL_VALID_BASELINE["nDCG@5"],
        "validation_primary": float(metrics["primary"]),
        "baseline_primary": OFFICIAL_VALID_BASELINE["primary"],
        "absolute_delta_primary": float(metrics["primary"]) - OFFICIAL_VALID_BASELINE["primary"],
    }
    resources = {
        "token_consumption": usage,
        "agent_wall_clock_seconds": timing.get("total_wall_clock_seconds"),
        "agent_wall_clock_complete": bool(timing.get("complete", False)),
        "agent_wall_clock_note": (
            "Cumulative timing covers the run from its first invocation."
            if timing.get("complete") else
            "Wall-clock timing is unavailable or incomplete for this legacy run."
        ),
        "iterations": len(scored),
        "attempts": len(records),
        "manual_interventions": 0,
        "manual_intervention_summary": "None",
    }
    bundle_dir = run_dir / "submission"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "iteration_log.json").write_text(
        json.dumps({"manual_interventions": 0, "attempts": attempts}, indent=2, default=str),
        encoding="utf-8",
    )
    (bundle_dir / "iteration_log.md").write_text(_markdown_log(attempts), encoding="utf-8")
    (bundle_dir / "results.json").write_text(
        json.dumps({"result": result_row, "resources": resources}, indent=2), encoding="utf-8"
    )
    results_md = (
        "# Run results\n\n"
        "| Benchmark | Best experiment | Validation GAUC | Absolute delta GAUC | "
        "Validation nDCG@5 | Absolute delta nDCG@5 | Validation primary | Absolute delta primary |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| KuaiRand-Pure | {result_row['best_experiment_id']} | "
        f"{result_row['validation_GAUC']:.10f} | {result_row['absolute_delta_GAUC']:+.10f} | "
        f"{result_row['validation_nDCG@5']:.10f} | {result_row['absolute_delta_nDCG@5']:+.10f} | "
        f"{result_row['validation_primary']:.10f} | {result_row['absolute_delta_primary']:+.10f} |\n\n"
        "## Resource usage\n\n"
        f"- Total LLM tokens: {usage['total_tokens']} "
        f"(input {usage['input_tokens']}, output {usage['output_tokens']})\n"
        f"- Token accounting complete: {usage['complete']}\n"
        f"- Total agent wall-clock seconds: {resources['agent_wall_clock_seconds']}\n"
        f"- Wall-clock accounting complete: {resources['agent_wall_clock_complete']}\n"
        f"- Counted iterations: {resources['iterations']}\n"
        f"- Total attempts: {resources['attempts']}\n"
        "- Manual interventions: 0 (none)\n"
    )
    (bundle_dir / "results.md").write_text(results_md, encoding="utf-8")

    artifact = _portable_sandbox(run_dir, best) / "predictions_test.npz"
    output_csv = bundle_dir / "kuairand_pure_submission.csv"
    submission_error = None
    submission_rows = None
    try:
        submission_rows = write_submission_csv(artifact, resolved_data_dir, output_csv)
    except (OSError, ValueError) as exc:
        submission_error = f"{type(exc).__name__}: {exc}"
        if output_csv.exists():
            output_csv.unlink()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if submission_error is None else "incomplete",
        "run_dir": str(run_dir),
        "best_experiment_id": best["experiment_id"],
        "files": {
            "iteration_log_markdown": "iteration_log.md",
            "iteration_log_json": "iteration_log.json",
            "results_markdown": "results.md",
            "results_json": "results.json",
            "final_model_output": "kuairand_pure_submission.csv" if submission_error is None else None,
        },
        "submission_rows": submission_rows,
        "submission_error": submission_error,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
