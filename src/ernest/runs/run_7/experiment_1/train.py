"""Train the neutral seed scaffold and emit validation scores for fixed evaluation."""

import argparse
import json

import numpy as np

from data import encode, load
from model import Model


def within_user_auc(user_ids, labels, scores):
    """Training-only early-stop proxy; final scoring is owned by the Experimentor."""
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
    train_features, train_labels, _ = encoded["train"]
    valid_features, valid_labels, valid_users = encoded[config["split"]]
    model = Model(
        dimension,
        learning_rate=config["learning_rate"],
        l2=config["l2"],
    )
    if args.contract_check:
        probe_size = min(8, len(train_labels))
        validation_count = len(valid_features)
        if probe_size == 0 or validation_count == 0:
            raise ValueError("contract probe requires non-empty train and validation slices")
        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
        probe_scores = model.predict(valid_features)
        row_ids = np.arange(validation_count, dtype=np.int64)
        if probe_scores.ndim != 1 or len(probe_scores) != validation_count:
            raise ValueError("model prediction shape violates the interface contract")
        if len(valid_users) != validation_count or not np.array_equal(
            row_ids, np.arange(validation_count, dtype=np.int64)
        ):
            raise ValueError("validation alignment violates the interface contract")
        if not np.isfinite(loss) or not np.all(np.isfinite(probe_scores)):
            raise ValueError("model produced NaN or infinity during contract probe")
        print(json.dumps({"contract": "ok", "feature_shape": list(train_features.shape)}))
        return
    rng = np.random.default_rng(config["seed"])
    best_score, best_state, stale = -1.0, None, 0
    for epoch in range(1, config["max_epochs"] + 1):
        order = rng.permutation(len(train_labels))
        losses = []
        for index in range(0, len(order), config["batch_size"]):
            batch = order[index:index + config["batch_size"]]
            losses.append(model.step(train_features[batch], train_labels[batch]))
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
    scores = model.predict(valid_features)
    row_ids = np.arange(len(valid_features), dtype=np.int64)
    if scores.ndim != 1 or len(scores) != len(row_ids):
        raise ValueError("model prediction shape violates the interface contract")
    if not np.all(np.isfinite(scores)):
        raise ValueError("model produced NaN or infinity in validation predictions")
    np.savez(
        args.output,
        row_ids=row_ids,
        scores=scores,
    )


if __name__ == "__main__":
    main()
