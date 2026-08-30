"""Post-score diagnostics. These run only after official scoring has succeeded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .evaluation import evaluate


def _compact_metrics(users: Iterable[Any], labels: Iterable[int], scores: Iterable[float]) -> Dict[str, Any]:
    values = evaluate(users, labels, scores)
    ranking = values["ranking_diagnostics"]
    return {
        "rows": values["rows"],
        "users": values["users"],
        "positive_rate": values["data_diagnostics"]["positive_rate"],
        "GAUC": values["GAUC"],
        "nDCG@5": values["nDCG@5"],
        "primary": values["primary"],
        "Precision@5": ranking["Precision@5"],
        "Recall@5": ranking["Recall@5"],
        "MAP@5": ranking["MAP@5"],
        "MRR@5": ranking["MRR@5"],
    }


def _frequency_bucket(value: int) -> str:
    if value <= 0:
        return "cold_0"
    if value <= 4:
        return "rare_1_4"
    if value <= 19:
        return "medium_5_19"
    return "warm_20_plus"


def _positive_bucket(value: int) -> str:
    if value == 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 4:
        return "two_to_four"
    return "five_plus"


def _load_standard_rows(data_dir: Path):
    import pandas as pd

    frames = []
    for name in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        path = data_dir / name
        if path.is_file():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError("standard KuaiRand logs are missing")
    rows = pd.concat(frames, ignore_index=True)
    rows["date"] = rows["date"].astype(int)
    rows["long_view"] = (
        pd.to_numeric(rows["long_view"], errors="coerce").fillna(0).ne(0).astype("int8")
    )
    return rows


def analyze_prediction_artifact(path: Path, data_dir: Path) -> Dict[str, Any]:
    """Join scored validation rows to safe segment attributes and calculate failure slices."""
    import numpy as np
    import pandas as pd

    rows = _load_standard_rows(data_dir)
    train = rows.loc[(rows["date"] >= 20220408) & (rows["date"] <= 20220421)].copy()
    valid = rows.loc[(rows["date"] >= 20220422) & (rows["date"] <= 20220428)].copy()
    valid = valid.reset_index(drop=True)
    with np.load(path, allow_pickle=False) as artifact:
        scores = artifact["scores"]
    if len(scores) != len(valid):
        raise ValueError("segment diagnostics received non-canonical validation predictions")
    valid["__score"] = scores

    user_counts = train.groupby("user_id").size()
    item_counts = train.groupby("video_id").size()
    valid["user_frequency"] = valid["user_id"].map(user_counts).fillna(0).astype(int).map(_frequency_bucket)
    valid["item_frequency"] = valid["video_id"].map(item_counts).fillna(0).astype(int).map(_frequency_bucket)
    positives = valid.groupby("user_id")["long_view"].sum()
    valid["user_positive_count"] = valid["user_id"].map(positives).astype(int).map(_positive_bucket)
    if "hourmin" in valid:
        valid["hour"] = (valid["hourmin"].fillna(0).astype(int) // 100).astype(str)
    if "duration_ms" in valid and "duration_ms" in train:
        numeric_train = pd.to_numeric(train["duration_ms"], errors="coerce")
        edges = sorted(set(float(value) for value in numeric_train.quantile(
            [0.0, 0.25, 0.5, 0.75, 1.0]
        ).dropna()))
        if len(edges) >= 2:
            edges[0], edges[-1] = float("-inf"), float("inf")
            valid["duration_bucket"] = pd.cut(
                pd.to_numeric(valid["duration_ms"], errors="coerce"),
                bins=edges, include_lowest=True, duplicates="drop",
            ).astype(str)
    user_path = data_dir / "user_features_pure.csv"
    if user_path.is_file():
        user_features = pd.read_csv(user_path, usecols=lambda name: name in {
            "user_id", "user_active_degree"
        })
        if "user_active_degree" in user_features:
            valid = valid.merge(user_features, on="user_id", how="left")

    segment_columns = [
        name for name in (
            "user_frequency", "item_frequency", "user_active_degree", "tab", "hour",
            "duration_bucket", "user_positive_count",
        ) if name in valid.columns
    ]
    global_values = _compact_metrics(
        valid["user_id"].astype(str), valid["long_view"], valid["__score"]
    )
    segments: Dict[str, List[Dict[str, Any]]] = {}
    for column in segment_columns:
        entries = []
        for value, subset in valid.groupby(column, dropna=False, observed=True):
            metrics = _compact_metrics(
                subset["user_id"].astype(str), subset["long_view"], subset["__score"]
            )
            metrics["value"] = "__MISSING__" if pd.isna(value) else str(value)
            metrics["primary_delta_from_global"] = metrics["primary"] - global_values["primary"]
            entries.append(metrics)
        entries.sort(key=lambda item: (item["primary"], -item["rows"]))
        segments[column] = entries
    return {
        "scope": "post_score_official_validation_diagnostics",
        "global": global_values,
        "segments": segments,
        "warnings": [
            "Segments are diagnostic slices of official validation, not independent test estimates.",
            "user_positive_count uses validation labels only for retrospective error analysis.",
        ],
    }


def attach_segment_diagnostics(
    metrics: Dict[str, Any], artifact: Path, data_dir: Path, output: Path
) -> Dict[str, Any]:
    diagnostics = analyze_prediction_artifact(artifact, data_dir)
    output.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    metrics = dict(metrics)
    metrics["segment_diagnostics"] = diagnostics
    return metrics
