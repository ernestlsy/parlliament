"""Train the neutral additive scaffold with request-context categorical features and emit validation scores."""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from data import encode, load
from model import Model


def _official_primary(scores):
    """Obtain the official validation primary from the evaluator interface."""
    module_names = []
    configured_module = os.environ.get("OFFICIAL_EVALUATOR_MODULE")
    if configured_module:
        module_names.append(configured_module)
    module_names.extend(("official_evaluator", "evaluator", "evaluation"))
    last_error = None
    for module_name in module_names:
        try:
            evaluator = importlib.import_module(module_name)
        except ImportError as error:
            last_error = error
            continue
        primary_function = getattr(evaluator, "official_primary", None)
        if primary_function is None:
            last_error = AttributeError(
                f"{module_name} does not provide official_primary"
            )
            continue
        primary = float(primary_function(scores))
        if not np.isfinite(primary):
            raise ValueError("evaluator returned a non-finite official primary")
        return primary
    command = os.environ.get("OFFICIAL_EVALUATOR_COMMAND")
    if command:
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
            scores_path = handle.name
        try:
            np.savez(scores_path, scores=np.asarray(scores, dtype=np.float64))
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "OFFICIAL_SCORES_PATH": scores_path},
            )
            primary = float(result.stdout.strip().splitlines()[-1])
            if not np.isfinite(primary):
                raise ValueError("evaluator returned a non-finite official primary")
            return primary
        finally:
            try:
                os.unlink(scores_path)
            except FileNotFoundError:
                pass
    return None


def _context_fields(row):
    context = row.get("request_context", {})
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (TypeError, ValueError):
            context = {}
    if not isinstance(context, dict):
        context = {}
    enriched = dict(row)
    for field in ("tab", "hour", "weekday"):
        if field not in enriched:
            value = context.get(field, "__UNK__")
            enriched[field] = "__UNK__" if value is None else value
    return enriched


def _normalize_json_file(source, target):
    suffix = os.path.splitext(source)[1].lower()
    if suffix == ".json":
        try:
            with open(source, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                payload = [
                    _context_fields(item) if isinstance(item, dict) else item
                    for item in payload
                ]
            elif isinstance(payload, dict):
                payload = _context_fields(payload)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    try:
        with open(source, encoding="utf-8") as source_handle:
            lines = source_handle.readlines()
    except UnicodeDecodeError:
        shutil.copy2(source, target)
        return
    changed = False
    output = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            output.append(line)
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            output.append(line)
            continue
        if isinstance(value, dict):
            value = _context_fields(value)
            changed = True
            ending = "\n" if line.endswith("\n") else ""
            output.append(json.dumps(value) + ending)
        else:
            output.append(line)
    if changed:
        with open(target, "w", encoding="utf-8") as target_handle:
            target_handle.writelines(output)
    else:
        shutil.copy2(source, target)


def _normalized_data_dir(data_dir):
    temporary_dir = tempfile.mkdtemp(prefix="context_data_")
    for root, _, files in os.walk(data_dir):
        relative = os.path.relpath(root, data_dir)
        destination_root = temporary_dir if relative == "." else os.path.join(temporary_dir, relative)
        os.makedirs(destination_root, exist_ok=True)
        for filename in files:
            source = os.path.join(root, filename)
            target = os.path.join(destination_root, filename)
            if os.path.splitext(filename)[1].lower() in (".json", ".jsonl", ".ndjson"):
                _normalize_json_file(source, target)
            else:
                shutil.copy2(source, target)
    return temporary_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract-check", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    temporary_data_dir = None
    try:
        try:
            splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
        except KeyError as error:
            if error.args and error.args[0] in ("tab", "hour", "weekday"):
                temporary_data_dir = _normalized_data_dir(args.data_dir)
                splits = load(
                    temporary_data_dir,
                    max_rows_per_split=64 if args.contract_check else None,
                )
            else:
                raise
        encoded, dimension = encode(splits)
        train_features, train_labels, _ = encoded["train"]
        valid_features, valid_labels, _ = encoded[config["split"]]
        model = Model(
            dimension,
            learning_rate=config["learning_rate"],
            l2=config["l2"],
        )
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
        best_score, best_state, stale = -np.inf, None, 0
        for epoch in range(1, config["max_epochs"] + 1):
            order = rng.permutation(len(train_labels))
            losses = []
            for index in range(0, len(order), config["batch_size"]):
                batch = order[index:index + config["batch_size"]]
                losses.append(model.step(train_features[batch], train_labels[batch]))
            predictions = model.predict(valid_features)
            primary = _official_primary(predictions)
            if primary is None:
                best_state = model.state()
                stale = 0
                print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_primary=unavailable")
                continue
            print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_primary={primary:.6f}")
            if primary > best_score + 1e-5:
                best_score, best_state, stale = primary, model.state(), 0
            else:
                stale += 1
                if stale >= config["patience"]:
                    break
        if best_state is None:
            raise RuntimeError("training produced no checkpoint")
        model.load_state(best_state)
        scores = np.asarray(model.predict(valid_features), dtype=np.float64)
        if scores.ndim != 1 or len(scores) != len(valid_labels) or not np.all(np.isfinite(scores)):
            raise ValueError("validation predictions violate the interface contract")
        np.savez(
            args.output,
            row_ids=np.arange(len(valid_labels), dtype=np.int64),
            scores=scores,
        )
    finally:
        if temporary_data_dir is not None:
            shutil.rmtree(temporary_data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
