"""Train the additive five-field scorer and emit canonical prediction archives."""

import argparse
import json
from pathlib import Path

import numpy as np

from data import encode, load
from model import Model


def within_user_auc(user_ids, labels, scores):
    """Validation-only early-stop proxy."""
    grouped = {}
    for user, label, score in zip(user_ids, labels, scores):
        grouped.setdefault(str(user), []).append((float(score), int(label)))
    numerator = denominator = 0.0
    for rows in grouped.values():
        positives = sum(label for _, label in rows)
        negatives = len(rows) - positives
        if positives == 0 or negatives == 0:
            continue
        wins = 0.0
        for positive_score, label in rows:
            if not label:
                continue
            for negative_score, other_label in rows:
                if other_label:
                    continue
                wins += positive_score > negative_score
                wins += 0.5 * (positive_score == negative_score)
        numerator += positives * wins / (positives * negatives)
        denominator += positives
    return numerator / denominator if denominator else 0.5


def within_user_ndcg_at_5(user_ids, labels, scores):
    """Compute unweighted validation NDCG@5 with canonical-order tie breaking."""
    grouped = {}
    for position, (user, label, score) in enumerate(zip(user_ids, labels, scores)):
        grouped.setdefault(str(user), []).append((float(score), int(label), position))

    ndcgs = []
    for rows in grouped.values():
        labels_for_user = [label for _, label, _ in rows]
        if not any(labels_for_user):
            continue

        ranked_rows = sorted(rows, key=lambda row: (-row[0], row[2]))
        dcg = sum(
            (2 ** label - 1) / np.log2(rank + 1)
            for rank, (_, label, _) in enumerate(ranked_rows[:5], start=1)
        )
        ideal_labels = sorted(labels_for_user, reverse=True)
        ideal_dcg = sum(
            (2 ** label - 1) / np.log2(rank + 1)
            for rank, label in enumerate(ideal_labels[:5], start=1)
        )
        if ideal_dcg > 0:
            ndcgs.append(dcg / ideal_dcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def build_train_pairs(train_features, train_labels, train_users, rng):
    """Sample K=4 same-user negatives uniformly without replacement when possible, using all available negatives when fewer than four exist."""
    del train_features
    k = 4
    grouped = {}
    for index, user in enumerate(train_users):
        grouped.setdefault(str(user), []).append(index)

    positive_indices = []
    negative_indices = []
    for indices in grouped.values():
        positives = [index for index in indices if train_labels[index] == 1]
        negatives = [index for index in indices if train_labels[index] == 0]
        if not positives or not negatives:
            continue
        sample_size = min(k, len(negatives))
        for positive_index in positives:
            sampled_negatives = rng.choice(
                negatives, size=sample_size, replace=False
            )
            for negative_index in sampled_negatives:
                positive_indices.append(positive_index)
                negative_indices.append(negative_index)

    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract-check", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    splits = load(
        args.data_dir,
        max_rows_per_split=64 if args.contract_check else None,
        include_test_labels=False,
    )
    encoded, dimension = encode(splits)
    train_features, train_labels, train_users = encoded["train"]
    valid_features, valid_labels, valid_users = encoded["valid"]
    test_features, test_labels, test_users = encoded["test"]
    assert test_labels is None

    model = Model(
        dimension,
        learning_rate=config["learning_rate"],
        l2=config["l2"],
    )

    if args.contract_check:
        if len(train_labels) == 0 or len(valid_labels) == 0:
            raise ValueError("contract probe requires non-empty train and validation slices")
        probe_size = min(8, len(train_labels))
        probe_rng = np.random.default_rng(config["seed"])
        positive_indices, negative_indices = build_train_pairs(
            train_features[:probe_size],
            train_labels[:probe_size],
            train_users[:probe_size],
            probe_rng,
        )
        if len(positive_indices) == 0:
            positive_indices, negative_indices = build_train_pairs(
                train_features,
                train_labels,
                train_users,
                probe_rng,
            )
        if len(positive_indices) == 0:
            raise ValueError(
                "contract probe requires a same-user positive-negative pair in the capped training split"
            )
        probe_pairs = min(8, len(positive_indices))
        loss = model.step(
            train_features[positive_indices[:probe_pairs]],
            train_features[negative_indices[:probe_pairs]],
        )
        probe_scores = model.predict(valid_features)
        if probe_scores.ndim != 1 or len(probe_scores) != len(valid_features):
            raise ValueError("model prediction shape violates the interface contract")
        if not np.isfinite(loss) or not np.all(np.isfinite(probe_scores)):
            raise ValueError("model produced NaN or infinity during contract probe")
        print(json.dumps({"contract": "ok", "feature_shape": list(train_features.shape)}))
        return

    best_score, best_state, stale = -1.0, None, 0
    for epoch in range(1, config["max_epochs"] + 1):
        epoch_rng = np.random.default_rng(config["seed"] + epoch)
        positive_indices, negative_indices = build_train_pairs(
            train_features, train_labels, train_users, epoch_rng
        )
        if len(positive_indices) == 0:
            raise ValueError("training split contains no same-user positive-negative pairs")
        order = epoch_rng.permutation(len(positive_indices))
        losses = []
        for index in range(0, len(order), config["batch_size"]):
            batch_positions = order[index:index + config["batch_size"]]
            losses.append(
                model.step(
                    train_features[positive_indices[batch_positions]],
                    train_features[negative_indices[batch_positions]],
                )
            )
        predictions = model.predict(valid_features)
        valid_gauc = within_user_auc(valid_users, valid_labels, predictions)
        valid_ndcg_at_5 = within_user_ndcg_at_5(
            valid_users, valid_labels, predictions
        )
        valid_primary = 0.5 * (valid_gauc + valid_ndcg_at_5)
        print(
            f"epoch={epoch} loss={np.mean(losses):.6f} "
            f"valid_gauc={valid_gauc:.6f} "
            f"valid_ndcg_at_5={valid_ndcg_at_5:.6f} "
            f"valid_primary={valid_primary:.6f} "
            f"checkpoint_selection_metric=valid_primary"
        )
        if valid_primary > best_score + 1e-5:
            best_score, best_state, stale = valid_primary, model.state(), 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state(best_state)

    output_path = Path(args.output)
    np.savez(
        output_path.with_name("predictions_valid.npz"),
        row_ids=np.arange(len(valid_features), dtype=np.int64),
        scores=model.predict(valid_features),
    )
    np.savez(
        output_path.with_name("predictions_test.npz"),
        row_ids=np.arange(len(test_features), dtype=np.int64),
        scores=model.predict(test_features),
    )


if __name__ == "__main__":
    main()
