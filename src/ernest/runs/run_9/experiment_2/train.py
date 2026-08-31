"""Train the KuaiRand FM seed and emit validation scores for fixed evaluation."""

import argparse
import collections
import json
import math

import numpy as np

from data import encode, load
from model import Model


def _auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average_rank
        index = end + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    positive_rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, pairs) if label == 1
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _ndcg_at_five(ranked_labels):
    discounts = [math.log2(index + 2) for index in range(5)]
    dcg = sum(
        ((2 ** label) - 1) / discounts[index]
        for index, label in enumerate(ranked_labels[:5])
    )
    ideal = sorted(ranked_labels, reverse=True)[:5]
    ideal_dcg = sum(
        ((2 ** label) - 1) / discounts[index]
        for index, label in enumerate(ideal)
    )
    return 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg


def training_selection_metrics(user_ids, labels, scores):
    grouped = collections.defaultdict(list)
    for user, label, score in zip(user_ids, labels, scores):
        grouped[str(user)].append((float(score), int(label)))
    auc_numerator = auc_denominator = 0.0
    ndcg_values = []
    for rows in grouped.values():
        rows.sort(key=lambda value: -value[0])
        ranked_labels = [label for _, label in rows]
        positives = sum(ranked_labels)
        if 0 < positives < len(rows):
            auc_numerator += positives * _auc(
                ranked_labels, [score for score, _ in rows]
            )
            auc_denominator += positives
        ndcg_values.append(_ndcg_at_five(ranked_labels))
    gauc = auc_numerator / auc_denominator if auc_denominator else 0.5
    ndcg = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0
    return gauc, ndcg, (gauc + ndcg) / 2.0


def _pair_features(features, labels, user_ids, generator, limit=None):
    grouped = collections.defaultdict(lambda: {0: [], 1: []})
    for row_index, (user_id, label) in enumerate(zip(user_ids, labels)):
        label_value = int(label)
        if label_value in (0, 1):
            grouped[user_id][label_value].append(row_index)

    positive_indices = []
    negative_indices = []
    for rows in grouped.values():
        if rows[1] and rows[0]:
            positive_indices.append(
                rows[1][generator.integers(len(rows[1]))]
            )
            negative_indices.append(
                rows[0][generator.integers(len(rows[0]))]
            )
            if limit is not None and len(positive_indices) >= limit:
                break
    if not positive_indices:
        return None, None
    return features[positive_indices], features[negative_indices]


def _pairwise_method(model):
    preferred_names = (
        "pairwise_step",
        "train_pairwise_step",
        "pairwise_train_step",
        "train_pairwise",
        "pairwise_update",
    )
    for name in preferred_names:
        method = getattr(model, name, None)
        if callable(method):
            return method
    for name in dir(model):
        if "pairwise" in name.lower():
            method = getattr(model, name, None)
            if callable(method):
                return method
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract-check", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
    encoded, dimension = encode(splits)
    train_features, train_labels, train_users = encoded["train"]
    valid_features, valid_labels, valid_users = encoded[config["split"]]
    model = Model(
        dimension,
        interaction_dimension=config["interaction_dimension"],
        learning_rate=config["learning_rate"],
        l2=config["l2"],
        seed=config["seed"],
    )
    pairwise_step = _pairwise_method(model)

    if args.contract_check:
        probe_size = min(8, len(train_labels))
        if probe_size == 0 or len(valid_labels) == 0:
            raise ValueError("contract probe requires non-empty train and validation slices")
        probe_generator = np.random.default_rng(config["seed"])
        positive_features, negative_features = _pair_features(
            train_features,
            train_labels,
            train_users,
            probe_generator,
            limit=8,
        )
        if positive_features is not None and pairwise_step is not None:
            loss = pairwise_step(
                positive_features=positive_features,
                negative_features=negative_features,
            )
            if not np.isfinite(loss):
                raise ValueError("model produced NaN or infinity during contract probe")
        probe_scores = model.predict(valid_features[:probe_size])
        if probe_scores.ndim != 1 or len(probe_scores) != min(
            probe_size, len(valid_features)
        ):
            raise ValueError("model prediction shape violates the interface contract")
        if not np.all(np.isfinite(probe_scores)):
            raise ValueError("model produced NaN or infinity during contract probe")
        print(json.dumps({
            "contract": "ok",
            "feature_shape": list(train_features.shape),
            "fields": 7,
            "interaction_dimension": config["interaction_dimension"],
        }))
        return

    if pairwise_step is None:
        raise AttributeError("Model does not provide a pairwise training method")

    generator = np.random.default_rng(config["seed"])
    best_primary, best_state, stale = -1.0, None, 0
    for epoch in range(1, config["max_epochs"] + 1):
        order = generator.permutation(len(train_labels))
        losses = []
        for index in range(0, len(order), config["batch_size"]):
            batch = order[index:index + config["batch_size"]]
            positive_features, negative_features = _pair_features(
                train_features[batch],
                train_labels[batch],
                train_users[batch],
                generator,
            )
            if positive_features is not None:
                losses.append(pairwise_step(
                    positive_features=positive_features,
                    negative_features=negative_features,
                ))
        scores = model.predict(valid_features)
        gauc, ndcg, primary = training_selection_metrics(
            valid_users, valid_labels, scores
        )
        loss_value = np.mean(losses) if losses else 0.0
        print(
            f"epoch={epoch} loss={loss_value:.6f} valid_GAUC={gauc:.6f} "
            f"valid_nDCG@5={ndcg:.6f} valid_primary={primary:.6f}"
        )
        if primary > best_primary + 1e-5:
            best_primary, best_state, stale = primary, model.state(), 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state(best_state)
    np.savez(
        args.output,
        row_ids=np.arange(len(valid_labels), dtype=np.int64),
        scores=model.predict(valid_features),
    )


if __name__ == "__main__":
    main()
