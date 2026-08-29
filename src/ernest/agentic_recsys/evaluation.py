"""Immutable scoring implementation matching the KuaiRand starter kit."""

from __future__ import annotations

import collections
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


METRIC_CATALOG = {
    "official_objective": {
        "GAUC": "Positive-count-weighted mean within-user ROC AUC.",
        "nDCG@5": "Mean within-user normalized discounted cumulative gain at rank 5.",
        "primary": "Mean of GAUC and nDCG@5; the only convergence and experiment-selection objective.",
    },
    "classification": {
        "threshold_strategy": (
            "Threshold is selected on validation to maximize F1; ties use balanced accuracy, "
            "accuracy, then the higher threshold. These values are diagnostic, not test estimates."
        ),
        "metrics": {
            "accuracy": "Fraction of correctly classified rows.",
            "balanced_accuracy": "Mean recall over classes present in the split.",
            "precision": "Positive predictive value at the selected threshold.",
            "recall": "True-positive rate at the selected threshold.",
            "specificity": "True-negative rate at the selected threshold.",
            "f1": "Harmonic mean of precision and recall.",
            "matthews_correlation": "Balanced confusion-matrix correlation coefficient.",
            "predicted_positive_rate": "Fraction of rows classified positive.",
            "confusion_matrix": "True/false positive and true/false negative counts.",
            "threshold": "Selected raw-score cutoff; null means no positive predictions.",
        },
    },
    "ranking_diagnostics": {
        "global_AUC": "Row-level ROC AUC without user grouping.",
        "average_precision": "Tie-aware area under the stepwise precision-recall curve.",
        "Precision@5": "Macro mean fraction of each user's top five rows that are relevant.",
        "Recall@5": "Macro mean fraction of each user's positives retrieved in the top five.",
        "MAP@5": "Macro mean average precision truncated at five.",
        "MRR@5": "Macro mean reciprocal rank of the first positive in the top five.",
        "HitRate@5": "Fraction of users with at least one positive in the top five.",
    },
    "data_diagnostics": {
        "positive_rate": "Observed long-view prevalence.",
        "score_mean": "Mean raw prediction score.",
        "score_std": "Population standard deviation of raw prediction scores.",
        "score_min": "Minimum raw prediction score.",
        "score_max": "Maximum raw prediction score.",
    },
}


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


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware area under the stepwise precision-recall curve."""
    positives = sum(labels)
    if positives == 0:
        return 0.0
    grouped = collections.defaultdict(lambda: [0, 0])
    for label, score in zip(labels, scores):
        grouped[float(score)][int(label)] += 1
    true_positives = false_positives = 0
    previous_recall = result = 0.0
    for score in sorted(grouped, reverse=True):
        negatives, positives_at_score = grouped[score]
        true_positives += positives_at_score
        false_positives += negatives
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def _classification_values(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    positives, negatives = tp + fn, tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    present_class_recalls = []
    if positives:
        present_class_recalls.append(recall)
    if negatives:
        present_class_recalls.append(specificity)
    balanced_accuracy = (
        sum(present_class_recalls) / len(present_class_recalls)
        if present_class_recalls else 0.0
    )
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "matthews_correlation": (
            (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0
        ),
        "predicted_positive_rate": (tp + fp) / total if total else 0.0,
    }


def classification_diagnostics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Any]:
    """Select a deterministic max-F1 threshold and report binary diagnostics.

    Scores are arbitrary ranking values, so a fixed probability cutoff such as 0.5 is not valid.
    Equal-score rows enter the predicted-positive set together. Ties between thresholds are resolved
    by balanced accuracy, accuracy, and then the more conservative (higher) threshold.
    """
    total = len(labels)
    positives = sum(labels)
    negatives = total - positives
    candidates = [(None, 0, 0, negatives, positives)]
    grouped = collections.defaultdict(lambda: [0, 0])
    for label, score in zip(labels, scores):
        grouped[float(score)][int(label)] += 1
    tp = fp = 0
    for threshold in sorted(grouped, reverse=True):
        negatives_at_score, positives_at_score = grouped[threshold]
        tp += positives_at_score
        fp += negatives_at_score
        candidates.append((threshold, tp, fp, negatives - fp, positives - tp))

    def candidate_key(candidate):
        threshold, candidate_tp, candidate_fp, candidate_tn, candidate_fn = candidate
        values = _classification_values(candidate_tp, candidate_fp, candidate_tn, candidate_fn)
        threshold_tiebreak = float("inf") if threshold is None else threshold
        return values["f1"], values["balanced_accuracy"], values["accuracy"], threshold_tiebreak

    threshold, tp, fp, tn, fn = max(candidates, key=candidate_key)
    values = _classification_values(tp, fp, tn, fn)
    return {
        "threshold_strategy": "maximize_f1_on_validation; ties=balanced_accuracy,accuracy,higher_threshold",
        "threshold": threshold,
        **values,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
    }


def evaluate(
    user_ids: Iterable[object], labels: Iterable[float], scores: Iterable[float], k: int = 5
) -> Dict[str, Any]:
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
    ndcgs, precisions, recalls, hit_rates, reciprocal_ranks, average_precisions = ([] for _ in range(6))
    for rows in by_user.values():
        rows.sort(key=lambda pair: -pair[0])
        ranked_labels = [label for _, label in rows]
        positives = sum(ranked_labels)
        if 0 < positives < len(rows):
            weighted_auc += positives * auc(ranked_labels, [score for score, _ in rows])
            auc_weight += positives
        ndcgs.append(ndcg_at_k(ranked_labels, k))
        top_labels = ranked_labels[:k]
        top_positives = sum(top_labels)
        precisions.append(top_positives / min(k, len(ranked_labels)))
        recalls.append(top_positives / positives if positives else 0.0)
        hit_rates.append(1.0 if top_positives else 0.0)
        first_positive = next((index for index, label in enumerate(top_labels, 1) if label), None)
        reciprocal_ranks.append(1.0 / first_positive if first_positive else 0.0)
        precision_sum = 0.0
        seen_positives = 0
        for rank, label in enumerate(top_labels, 1):
            if label:
                seen_positives += 1
                precision_sum += seen_positives / rank
        average_precisions.append(
            precision_sum / min(positives, k) if positives else 0.0
        )
    gauc = weighted_auc / auc_weight if auc_weight else 0.5
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    binary_labels = [int(label) for label in ys]
    float_scores = [float(score) for score in predictions]
    mean = sum(float_scores) / len(float_scores) if float_scores else 0.0
    variance = (
        sum((score - mean) ** 2 for score in float_scores) / len(float_scores)
        if float_scores else 0.0
    )
    mean_or_zero = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "GAUC": float(gauc),
        f"nDCG@{k}": float(ndcg),
        "primary": float((gauc + ndcg) / 2.0),
        "users": len(by_user),
        "rows": len(ys),
        "classification": classification_diagnostics(binary_labels, float_scores),
        "ranking_diagnostics": {
            "global_AUC": float(auc(binary_labels, float_scores)),
            "average_precision": float(average_precision(binary_labels, float_scores)),
            f"Precision@{k}": mean_or_zero(precisions),
            f"Recall@{k}": mean_or_zero(recalls),
            f"MAP@{k}": mean_or_zero(average_precisions),
            f"MRR@{k}": mean_or_zero(reciprocal_ranks),
            f"HitRate@{k}": mean_or_zero(hit_rates),
        },
        "data_diagnostics": {
            "positive_rate": sum(binary_labels) / len(binary_labels) if binary_labels else 0.0,
            "score_mean": mean,
            "score_std": math.sqrt(variance),
            "score_min": min(float_scores) if float_scores else None,
            "score_max": max(float_scores) if float_scores else None,
        },
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


def score_prediction_artifact(path: Path, data_dir: Path, split: str = "valid") -> Dict[str, Any]:
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
