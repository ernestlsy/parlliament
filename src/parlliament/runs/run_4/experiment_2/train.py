"""Train the context-augmented additive scorer and emit validation and test scores."""

import argparse
import json
from pathlib import Path

import numpy as np

from data import encode, load
from model import Model


def within_user_auc(user_ids, labels, scores):
    """Compute the validation-only within-user AUC proxy."""
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


def _assert_artifact(path, row_ids, scores):
    row_ids = np.asarray(row_ids)
    scores = np.asarray(scores)
    if row_ids.ndim != 1 or scores.ndim != 1:
        raise ValueError("prediction artifact arrays must be one-dimensional")
    if len(row_ids) != len(scores):
        raise ValueError("prediction artifact arrays must have equal lengths")
    with np.load(path) as artifact:
        if set(artifact.files) != {"row_ids", "scores"}:
            raise ValueError("prediction artifact has an unexpected key set")
        if artifact["row_ids"].ndim != 1 or artifact["scores"].ndim != 1:
            raise ValueError("prediction artifact arrays must be one-dimensional")
        if len(artifact["row_ids"]) != len(artifact["scores"]):
            raise ValueError("prediction artifact arrays must have equal lengths")


def _group_complete_batches(user_ids, batch_size, rng):
    groups = {}
    for index, user_id in enumerate(user_ids):
        groups.setdefault(user_id, []).append(index)

    group_values = list(groups.values())
    shuffled_order = rng.permutation(len(group_values))
    batches = []
    current = []
    for group_index in shuffled_order:
        group = group_values[int(group_index)]
        if current and len(current) + len(group) > batch_size:
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
        if len(group) > batch_size:
            if current:
                batches.append(np.asarray(current, dtype=np.int64))
                current = []
            batches.append(np.asarray(group, dtype=np.int64))
        else:
            current.extend(group)
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


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
    train_features, train_labels, train_users, _train_row_ids = encoded["train"]
    valid_features, valid_labels, valid_users, valid_row_ids = encoded["valid"]
    test_features = encoded["test"][0]
    test_row_ids = encoded["test"][3]

    for name, features in (
        ("train", train_features),
        ("valid", valid_features),
        ("test", test_features),
    ):
        if features.ndim != 2 or features.shape[1] != 5:
            raise ValueError(f"{name} features must have shape (n, 5)")

    model = Model(
        dimension,
        learning_rate=config["learning_rate"],
        l2=config["l2"],
    )

    if args.contract_check:
        probe_size = min(8, len(train_labels))
        if probe_size == 0 or len(valid_labels) == 0:
            raise ValueError("contract probe requires non-empty train and validation slices")
        loss = model.step(
            train_features[:probe_size],
            train_labels[:probe_size],
            train_users[:probe_size],
        )
        probe_scores = model.predict(valid_features[:probe_size])
        expected_size = min(probe_size, len(valid_features))
        if probe_scores.ndim != 1 or len(probe_scores) != expected_size:
            raise ValueError("model prediction shape violates the interface contract")
        if not np.isfinite(loss) or not np.all(np.isfinite(probe_scores)):
            raise ValueError("model produced NaN or infinity during contract probe")
        print(json.dumps({"contract": "ok", "feature_shape": list(train_features.shape)}))
        return

    rng = np.random.default_rng(config["seed"])
    best_score, best_state, stale = -1.0, None, 0
    for epoch in range(1, config["max_epochs"] + 1):
        losses = []
        for batch in _group_complete_batches(train_users, config["batch_size"], rng):
            losses.append(
                model.step(
                    train_features[batch],
                    train_labels[batch],
                    train_users[batch],
                )
            )
        predictions = model.predict(valid_features)
        proxy = within_user_auc(valid_users, valid_labels, predictions)
        print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_gauc_proxy={proxy:.6f}")
        if proxy > best_score + 1e-5:
            best_score, best_state, stale = proxy, model.state(), 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")

    model.load_state(best_state)
    valid_scores = np.asarray(model.predict(valid_features), dtype=np.float32)
    test_scores = np.asarray(model.predict(test_features), dtype=np.float32)
    valid_path = Path(args.output)
    test_path = valid_path.with_name("predictions_test.npz")
    valid_row_ids = np.asarray(valid_row_ids, dtype=np.int64)
    test_row_ids = np.asarray(test_row_ids, dtype=np.int64)
    np.savez(valid_path, row_ids=valid_row_ids, scores=valid_scores)
    np.savez(test_path, row_ids=test_row_ids, scores=test_scores)
    _assert_artifact(valid_path, valid_row_ids, valid_scores)
    _assert_artifact(test_path, test_row_ids, test_scores)


if __name__ == "__main__":
    main()
