"""Train the configured factorization machine and emit validation and test scores."""

import argparse
import json
from pathlib import Path

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
    valid_features, valid_labels, valid_users = encoded["valid"]
    test_features, _, _ = encoded["test"]
    model = Model(
        dimension,
        learning_rate=config["learning_rate"],
        l2=config["l2"],
        factor_rank=config["factor_rank"],
    )
    if args.contract_check:
        probe_size = min(8, len(train_labels))
        if probe_size == 0 or len(valid_labels) == 0:
            raise ValueError("contract probe requires non-empty train and validation slices")
        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
        probe_scores = model.predict(valid_features)
        if probe_scores.ndim != 1 or len(probe_scores) != len(valid_features):
            raise ValueError("model prediction shape violates the interface contract")
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
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "predictions_valid.npz",
        row_ids=np.arange(len(valid_features), dtype=np.int64),
        scores=model.predict(valid_features),
    )
    np.savez(
        output_dir / "predictions_test.npz",
        row_ids=np.arange(len(test_features), dtype=np.int64),
        scores=model.predict(test_features),
    )


if __name__ == "__main__":
    main()
