"""Immutable scoring implementation matching the KuaiRand starter kit."""

from __future__ import annotations

import collections
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence


def auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(labels)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    positive_rank_sum = sum(r for r, (_, label) in zip(ranks, pairs) if label == 1)
    return (positive_rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)


def ndcg_at_k(labels: Sequence[float], k: int) -> float:
    discounts = [math.log2(i + 2) for i in range(k)]
    dcg = sum(((2 ** label) - 1) / discounts[i] for i, label in enumerate(labels[:k]))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2 ** label) - 1) / discounts[i] for i, label in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate(user_ids: Iterable[object], labels: Iterable[float], scores: Iterable[float], k: int = 5) -> Dict[str, float]:
    users, ys, predictions = list(user_ids), list(labels), list(scores)
    if not (len(users) == len(ys) == len(predictions)):
        raise ValueError("user_ids, labels, and scores must have equal lengths")
    if any(not math.isfinite(float(score)) for score in predictions):
        raise ValueError("scores contain NaN or infinity")
    by_user = collections.defaultdict(list)
    for user, label, score in zip(users, ys, predictions):
        if float(label) not in (0.0, 1.0):
            raise ValueError("long_view labels must be binary")
        by_user[str(user)].append((float(score), int(label)))
    weighted_auc = auc_weight = 0.0
    ndcgs = []
    for rows in by_user.values():
        rows.sort(key=lambda pair: -pair[0])
        ranked_labels = [label for _, label in rows]
        positives = sum(ranked_labels)
        if 0 < positives < len(rows):
            weighted_auc += positives * auc(ranked_labels, [score for score, _ in rows])
            auc_weight += positives
        ndcgs.append(ndcg_at_k(ranked_labels, k))
    gauc = weighted_auc / auc_weight if auc_weight else 0.5
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    return {
        "GAUC": float(gauc),
        f"nDCG@{k}": float(ndcg),
        "primary": float((gauc + ndcg) / 2.0),
        "users": len(by_user),
        "rows": len(ys),
    }


def load_ground_truth(data_dir: Path, split: str = "valid"):
    ranges = {"valid": (20220422, 20220428), "test": (20220429, 20220508)}
    if split not in ranges:
        raise ValueError(f"unsupported evaluation split: {split}")
    low, high = ranges[split]
    users, labels = [], []
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with (data_dir / filename).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if low <= int(row["date"]) <= high:
                    users.append(row["user_id"])
                    labels.append(1 if row["long_view"] != "0" else 0)
    return users, labels


def score_prediction_artifact(path: Path, data_dir: Path, split: str = "valid") -> Dict[str, float]:
    import numpy as np

    if not path.is_file():
        raise ValueError(f"prediction artifact does not exist: {path.name}")
    with np.load(path, allow_pickle=False) as artifact:
        expected = {"row_ids", "scores"}
        if set(artifact.files) != expected:
            raise ValueError(f"prediction arrays must be exactly {sorted(expected)}")
        row_ids = artifact["row_ids"]
        scores = artifact["scores"]
        users, labels = load_ground_truth(data_dir, split)
        if row_ids.ndim != 1 or scores.ndim != 1:
            raise ValueError("row_ids and scores must be one-dimensional")
        if not np.issubdtype(row_ids.dtype, np.integer):
            raise ValueError("row_ids must use an integer dtype")
        if len(row_ids) != len(labels) or len(scores) != len(labels):
            raise ValueError(
                f"prediction rows ({len(scores)}) do not match canonical {split} rows ({len(labels)})"
            )
        if not np.array_equal(row_ids, np.arange(len(labels))):
            raise ValueError("row_ids must be consecutive and in canonical validation order")
        return evaluate(users, labels, scores)
