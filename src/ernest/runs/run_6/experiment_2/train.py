"""Train the neutral additive scorer with impression-time request context features."""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np

from data import encode, load
from model import Model


FEATURE_NAMES = ("user_id", "video_id", "tab", "hour", "weekday")


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


def _as_columns(split):
    if hasattr(split, "columns") and hasattr(split, "__getitem__"):
        return {str(name): np.asarray(split[name]) for name in split.columns}
    if isinstance(split, dict):
        for nested_name in ("rows", "data", "records"):
            if nested_name in split and not any(
                str(key).lower() in {name, "label", "labels", "target", "y"}
                for key in split
                for name in FEATURE_NAMES
            ):
                nested = _as_columns(split[nested_name])
                if nested is not None:
                    result = dict(nested)
                    for key, value in split.items():
                        if key != nested_name:
                            result[str(key)] = np.asarray(value)
                    return result
        return {str(key): np.asarray(value) for key, value in split.items()}
    if isinstance(split, np.ndarray) and split.dtype.names:
        return {str(name): np.asarray(split[name]) for name in split.dtype.names}
    if isinstance(split, (list, tuple)) and split and isinstance(split[0], dict):
        keys = set().union(*(row.keys() for row in split))
        return {str(key): np.asarray([row.get(key) for row in split]) for key in keys}
    return None


def _find_column(columns, name):
    wanted = name.lower()
    for key, values in columns.items():
        if str(key).lower() == wanted:
            return np.asarray(values)
    return None


def _timestamp_parts(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number > 100000000000:
                number /= 1000.0
            return datetime.fromtimestamp(number)
        text = str(value).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _context_column(columns, feature):
    column = _find_column(columns, feature)
    if column is not None:
        return column
    aliases = {
        "tab": ("page", "tab_name", "request_tab", "context_tab"),
        "hour": ("impression_hour", "request_hour"),
        "weekday": ("day_of_week", "impression_weekday", "request_weekday"),
    }
    for alias in aliases.get(feature, ()):
        column = _find_column(columns, alias)
        if column is not None:
            return column
    timestamp = None
    for name in ("timestamp", "impression_time", "event_time", "datetime", "date_time", "time"):
        timestamp = _find_column(columns, name)
        if timestamp is not None:
            break
    if timestamp is not None and feature in ("hour", "weekday"):
        parts = [_timestamp_parts(value) for value in timestamp]
        if feature == "hour":
            return np.asarray([part.hour if part is not None else "__unknown__" for part in parts], dtype=object)
        return np.asarray([part.weekday() if part is not None else "__unknown__" for part in parts], dtype=object)
    return None


def _read_file(path):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return None


def _legacy_load(data_dir):
    root = Path(data_dir)
    files = [path for path in root.rglob("*") if path.is_file()]
    candidates = {}
    for split_name in ("train", "valid", "validation", "test"):
        matches = [path for path in files if split_name in path.stem.lower() and path.suffix.lower() in (".csv", ".json", ".jsonl", ".ndjson", ".npz")]
        if matches:
            candidates["valid" if split_name == "validation" else split_name] = matches[0]
    if "train" not in candidates or "valid" not in candidates:
        for path in files:
            if path.suffix.lower() not in (".json", ".npz"):
                continue
            try:
                value = _read_file(path)
            except Exception:
                continue
            if isinstance(value, dict) and "train" in value and ("valid" in value or "validation" in value):
                return {"train": value["train"], "valid": value.get("valid", value["validation"])}
    if "train" not in candidates or "valid" not in candidates:
        raise ValueError("unable to locate train and validation data")
    return {name: _read_file(path) for name, path in candidates.items() if name in ("train", "valid")}


def _encode_request_context(splits):
    columns_by_split = {name: _as_columns(split) for name, split in splits.items()}
    if any(columns is None for columns in columns_by_split.values()):
        raise ValueError("raw split columns are unavailable")

    values_by_split = {}
    for split_name, columns in columns_by_split.items():
        values = []
        for feature in FEATURE_NAMES:
            column = _context_column(columns, feature)
            if column is None:
                raise ValueError("missing required feature: " + feature)
            values.append(column)
        labels = None
        for name in ("label", "labels", "target", "y"):
            labels = _find_column(columns, name)
            if labels is not None:
                break
        users = values[0]
        if labels is None or any(len(column) != len(users) for column in values):
            raise ValueError("inconsistent split columns")
        values_by_split[split_name] = (values, np.asarray(labels), np.asarray(users))

    offsets = []
    mappings = []
    dimension = 0
    for field_index in range(len(FEATURE_NAMES)):
        mapping = {}
        for value in values_by_split["train"][0][field_index]:
            key = repr(value.item() if isinstance(value, np.generic) else value)
            if key not in mapping:
                mapping[key] = len(mapping) + 1
        offsets.append(dimension)
        mappings.append(mapping)
        dimension += len(mapping) + 1

    encoded = {}
    for split_name, (values, labels, users) in values_by_split.items():
        features = np.zeros((len(labels), len(FEATURE_NAMES)), dtype=np.int64)
        for field_index, column in enumerate(values):
            mapping = mappings[field_index]
            for row_index, value in enumerate(column):
                item = value.item() if isinstance(value, np.generic) else value
                features[row_index, field_index] = offsets[field_index] + mapping.get(repr(item), 0)
        encoded[split_name] = (features, labels, users)
    return encoded, dimension


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract-check", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    try:
        splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
    except (KeyError, TypeError, ValueError):
        splits = _legacy_load(args.data_dir)
    try:
        encoded, dimension = _encode_request_context(splits)
    except (TypeError, ValueError, KeyError):
        encoded, dimension = encode(splits)
    train_features, train_labels, _ = encoded["train"]
    valid_features, valid_labels, valid_users = encoded[config["split"]]
    model = Model(dimension, learning_rate=config["learning_rate"], l2=config["l2"])
    if args.contract_check:
        probe_size = min(8, len(train_labels))
        if probe_size == 0 or len(valid_labels) == 0:
            raise ValueError("contract probe requires non-empty train and validation slices")
        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
        probe_scores = model.predict(valid_features[:probe_size])
        if probe_scores.ndim != 1 or len(probe_scores) != min(probe_size, len(valid_features)):
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
    np.savez(args.output, row_ids=np.arange(len(valid_labels), dtype=np.int64), scores=model.predict(valid_features))


if __name__ == "__main__":
    main()
