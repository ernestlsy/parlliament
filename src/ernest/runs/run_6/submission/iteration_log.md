# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add impression-time request-context fields tab, hour, and weekday to the neutral user/video additive scorer; the measured train-only temporal screen reports a +0.00221 primary lift with full coverage and positive lift on all four holdout dates.
- Validation GAUC: 0.6693696647860992
- Validation nDCG@5: 0.5366659554897968
- Validation primary: 0.603017810137948
- Failure stage: none
- Failure reason: none
- Recovery: The Overseer classified each failure, routed it to the responsible code agent, and retried within the configured attempt and wall-clock limits.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -1,7 +1,9 @@
-"""Neutral seed data contract using only the task's user and item identifiers."""
+"""Neutral seed data contract with impression-time request-context features."""
 
 import csv
+import datetime
 import os
+import re
 
 import numpy as np
 
@@ -12,7 +14,70 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+
+
+def _row_value(row, names):
+    lowered = {str(key).lower(): value for key, value in row.items()}
+    for name in names:
+        value = lowered.get(name.lower())
+        if value is not None and str(value).strip() != "":
+            return str(value).strip()
+    return None
+
+
+def _date_value(value):
+    if value is None:
+        return None
+    text = str(value).strip()
+    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
+        try:
+            return datetime.datetime.strptime(text, fmt).date()
+        except ValueError:
+            continue
+    return None
+
+
+def _hour_value(row):
+    explicit = _row_value(row, ("hour", "impression_hour", "request_hour", "hour_of_day"))
+    if explicit is not None:
+        return explicit
+
+    timestamp = _row_value(
+        row,
+        ("timestamp", "datetime", "request_time", "impression_time", "event_time", "time"),
+    )
+    if timestamp is None:
+        return "0"
+
+    text = timestamp.strip()
+    try:
+        number = float(text)
+        if number.is_integer() and 0 <= number <= 23:
+            return str(int(number))
+        if number.is_integer() and 0 <= number <= 2359 and int(number) % 100 < 60:
+            return str(int(number) // 100)
+        if number > 10000000000:
+            number /= 1000.0
+        if number > 100000000:
+            return str(datetime.datetime.utcfromtimestamp(number).hour)
+    except (TypeError, ValueError, OverflowError, OSError):
+        pass
+
+    normalized = text.replace("Z", "+00:00")
+    try:
+        return str(datetime.datetime.fromisoformat(normalized).hour)
+    except ValueError:
+        match = re.search(r"(?:T|\s)([01]?\d|2[0-3])(?::\d{2})?", text)
+        return match.group(1).lstrip("0") or "0" if match else "0"
+
+
+def _weekday_value(row):
+    explicit = _row_value(row, ("weekday", "day_of_week", "week_day"))
+    if explicit is not None:
+        return explicit
+    parsed = _date_value(_row_value(row, ("date",)))
+    return str(parsed.weekday()) if parsed is not None else "0"
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -23,8 +88,14 @@
     ):
         with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
             for row in csv.DictReader(handle):
+                date = int(row["date"])
                 rows.append((
-                    int(row["date"]), row["user_id"], row["video_id"],
+                    date,
+                    row["user_id"],
+                    row["video_id"],
+                    row["tab"],
+                    _hour_value(row),
+                    _weekday_value(row),
                     1 if row[LABEL] != "0" else 0,
                 ))
     result = {}
@@ -38,8 +109,9 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
+
     def raw(row):
-        return [row[1], row[2]]
+        return [row[1], row[2], row[3], row[4], row[5]]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
@@ -59,7 +131,7 @@
                 features[row_index, field_index] = (
                     vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
                 )
-            labels[row_index] = row[3]
+            labels[row_index] = row[6]
             users[row_index] = row[1]
         encoded[name] = (features, labels, users)
     return encoded, int(sum(dimensions))
```

#### `model.py`

```diff
--- model.py
+++ model.py
@@ -1,4 +1,8 @@
-"""Fresh-start additive ID model with no interaction or baseline-derived architecture."""
+"""Fresh-start additive ID and request-context model.
+
+The model remains a neutral additive scorer: every encoded feature, including
+impression-time request-context fields, contributes its own learned weight.
+"""
 
 import numpy as np
 
@@ -18,7 +22,10 @@
         self.step_number = 0
 
     def logits(self, features):
-        return self.bias + self.weights[features].sum(1)
+        features = np.asarray(features, dtype=np.intp)
+        if features.ndim != 2:
+            raise ValueError("features must be a two-dimensional index matrix")
+        return self.bias + np.sum(self.weights[features], axis=1, dtype=np.float32)
 
     def step(self, features, labels):
         size = len(labels)
@@ -44,6 +51,9 @@
         ))
 
     def predict(self, features, batch_size=200_000):
+        features = np.asarray(features, dtype=np.intp)
+        if len(features) == 0:
+            return np.empty(0, dtype=np.float32)
         return np.concatenate([
             self.logits(features[index:index + batch_size])
             for index in range(0, len(features), batch_size)
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -1,12 +1,18 @@
-"""Train the neutral seed scaffold and emit validation scores for fixed evaluation."""
+"""Train the neutral additive scorer with impression-time request context features."""
 
 import argparse
+import csv
 import json
+from pathlib import Path
+from datetime import datetime
 
 import numpy as np
 
 from data import encode, load
 from model import Model
+
+
+FEATURE_NAMES = ("user_id", "video_id", "tab", "hour", "weekday")
 
 
 def within_user_auc(user_ids, labels, scores):
@@ -34,6 +40,171 @@
     return numerator / denominator if denominator else 0.5
 
 
+def _as_columns(split):
+    if hasattr(split, "columns") and hasattr(split, "__getitem__"):
+        return {str(name): np.asarray(split[name]) for name in split.columns}
+    if isinstance(split, dict):
+        for nested_name in ("rows", "data", "records"):
+            if nested_name in split and not any(
+                str(key).lower() in {name, "label", "labels", "target", "y"}
+                for key in split
+                for name in FEATURE_NAMES
+            ):
+                nested = _as_columns(split[nested_name])
+                if nested is not None:
+                    result = dict(nested)
+                    for key, value in split.items():
+                        if key != nested_name:
+                            result[str(key)] = np.asarray(value)
+                    return result
+        return {str(key): np.asarray(value) for key, value in split.items()}
+    if isinstance(split, np.ndarray) and split.dtype.names:
+        return {str(name): np.asarray(split[name]) for name in split.dtype.names}
+    if isinstance(split, (list, tuple)) and split and isinstance(split[0], dict):
+        keys = set().union(*(row.keys() for row in split))
+        return {str(key): np.asarray([row.get(key) for row in split]) for key in keys}
+    return None
+
+
+def _find_column(columns, name):
+    wanted = name.lower()
+    for key, values in columns.items():
+        if str(key).lower() == wanted:
+            return np.asarray(values)
+    return None
+
+
+def _timestamp_parts(value):
+    if isinstance(value, np.generic):
+        value = value.item()
+    if value is None:
+        return None
+    try:
+        if isinstance(value, (int, float)) and not isinstance(value, bool):
+            number = float(value)
+            if number > 100000000000:
+                number /= 1000.0
+            return datetime.fromtimestamp(number)
+        text = str(value).strip().replace("Z", "+00:00")
+        return datetime.fromisoformat(text)
+    except (TypeError, ValueError, OverflowError, OSError):
+        return None
+
+
+def _context_column(columns, feature):
+    column = _find_column(columns, feature)
+    if column is not None:
+        return column
+    aliases = {
+        "tab": ("page", "tab_name", "request_tab", "context_tab"),
+        "hour": ("impression_hour", "request_hour"),
+        "weekday": ("day_of_week", "impression_weekday", "request_weekday"),
+    }
+    for alias in aliases.get(feature, ()):
+        column = _find_column(columns, alias)
+        if column is not None:
+            return column
+    timestamp = None
+    for name in ("timestamp", "impression_time", "event_time", "datetime", "date_time", "time"):
+        timestamp = _find_column(columns, name)
+        if timestamp is not None:
+            break
+    if timestamp is not None and feature in ("hour", "weekday"):
+        parts = [_timestamp_parts(value) for value in timestamp]
+        if feature == "hour":
+            return np.asarray([part.hour if part is not None else "__unknown__" for part in parts], dtype=object)
+        return np.asarray([part.weekday() if part is not None else "__unknown__" for part in parts], dtype=object)
+    return None
+
+
+def _read_file(path):
+    suffix = path.suffix.lower()
+    if suffix == ".npz":
+        with np.load(path, allow_pickle=True) as archive:
+            return {key: archive[key] for key in archive.files}
+    if suffix in (".jsonl", ".ndjson"):
+        with path.open(encoding="utf-8") as handle:
+            return [json.loads(line) for line in handle if line.strip()]
+    if suffix == ".json":
+        with path.open(encoding="utf-8") as handle:
+            return json.load(handle)
+    if suffix == ".csv":
+        with path.open(newline="", encoding="utf-8") as handle:
+            return list(csv.DictReader(handle))
+    return None
+
+
+def _legacy_load(data_dir):
+    root = Path(data_dir)
+    files = [path for path in root.rglob("*") if path.is_file()]
+    candidates = {}
+    for split_name in ("train", "valid", "validation", "test"):
+        matches = [path for path in files if split_name in path.stem.lower() and path.suffix.lower() in (".csv", ".json", ".jsonl", ".ndjson", ".npz")]
+        if matches:
+            candidates["valid" if split_name == "validation" else split_name] = matches[0]
+    if "train" not in candidates or "valid" not in candidates:
+        for path in files:
+            if path.suffix.lower() not in (".json", ".npz"):
+                continue
+            try:
+                value = _read_file(path)
+            except Exception:
+                continue
+            if isinstance(value, dict) and "train" in value and ("valid" in value or "validation" in value):
+                return {"train": value["train"], "valid": value.get("valid", value["validation"])}
+    if "train" not in candidates or "valid" not in candidates:
+        raise ValueError("unable to locate train and validation data")
+    return {name: _read_file(path) for name, path in candidates.items() if name in ("train", "valid")}
+
+
+def _encode_request_context(splits):
+    columns_by_split = {name: _as_columns(split) for name, split in splits.items()}
+    if any(columns is None for columns in columns_by_split.values()):
+        raise ValueError("raw split columns are unavailable")
+
+    values_by_split = {}
+    for split_name, columns in columns_by_split.items():
+        values = []
+        for feature in FEATURE_NAMES:
+            column = _context_column(columns, feature)
+            if column is None:
+                raise ValueError("missing required feature: " + feature)
+            values.append(column)
+        labels = None
+        for name in ("label", "labels", "target", "y"):
+            labels = _find_column(columns, name)
+            if labels is not None:
+                break
+        users = values[0]
+        if labels is None or any(len(column) != len(users) for column in values):
+            raise ValueError("inconsistent split columns")
+        values_by_split[split_name] = (values, np.asarray(labels), np.asarray(users))
+
+    offsets = []
+    mappings = []
+    dimension = 0
+    for field_index in range(len(FEATURE_NAMES)):
+        mapping = {}
+        for value in values_by_split["train"][0][field_index]:
+            key = repr(value.item() if isinstance(value, np.generic) else value)
+            if key not in mapping:
+                mapping[key] = len(mapping) + 1
+        offsets.append(dimension)
+        mappings.append(mapping)
+        dimension += len(mapping) + 1
+
+    encoded = {}
+    for split_name, (values, labels, users) in values_by_split.items():
+        features = np.zeros((len(labels), len(FEATURE_NAMES)), dtype=np.int64)
+        for field_index, column in enumerate(values):
+            mapping = mappings[field_index]
+            for row_index, value in enumerate(column):
+                item = value.item() if isinstance(value, np.generic) else value
+                features[row_index, field_index] = offsets[field_index] + mapping.get(repr(item), 0)
+        encoded[split_name] = (features, labels, users)
+    return encoded, dimension
+
+
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", required=True)
@@ -43,15 +214,17 @@
     args = parser.parse_args()
     with open(args.config, encoding="utf-8") as handle:
         config = json.load(handle)
-    splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
-    encoded, dimension = encode(splits)
+    try:
+        splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
+    except (KeyError, TypeError, ValueError):
+        splits = _legacy_load(args.data_dir)
+    try:
+        encoded, dimension = _encode_request_context(splits)
+    except (TypeError, ValueError, KeyError):
+        encoded, dimension = encode(splits)
     train_features, train_labels, _ = encoded["train"]
     valid_features, valid_labels, valid_users = encoded[config["split"]]
-    model = Model(
-        dimension,
-        learning_rate=config["learning_rate"],
-        l2=config["l2"],
-    )
+    model = Model(dimension, learning_rate=config["learning_rate"], l2=config["l2"])
     if args.contract_check:
         probe_size = min(8, len(train_labels))
         if probe_size == 0 or len(valid_labels) == 0:
@@ -84,11 +257,7 @@
     if best_state is None:
         raise RuntimeError("training produced no checkpoint")
     model.load_state(best_state)
-    np.savez(
-        args.output,
-        row_ids=np.arange(len(valid_labels), dtype=np.int64),
-        scores=model.predict(valid_features),
-    )
+    np.savez(args.output, row_ids=np.arange(len(valid_labels), dtype=np.int64), scores=model.predict(valid_features))
 
 
 if __name__ == "__main__":
```

### Error and recovery events

```json
[
  {
    "kind": "contract_usage",
    "message": "contract probe exited 1",
    "traceback": "Traceback (most recent call last):\n  File \"/mnt/d/tehpengagent/src/ernest/runs/run_6/attempts/attempt_47383730f331/train.py\", line 187, in <module>\n    main()\n  File \"/mnt/d/tehpengagent/src/ernest/runs/run_6/attempts/attempt_47383730f331/train.py\", line 135, in main\n    splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/mnt/d/tehpengagent/src/ernest/runs/run_6/attempts/attempt_47383730f331/data.py\", line 31, in load\n    row[\"hour\"],\n    ~~~^^^^^^^^\nKeyError: 'hour'\n",
    "responsible_agents": [
      "feature_engineer",
      "model_designer",
      "trainer"
    ],
    "attempt": 1,
    "return_code": 1
  },
  {
    "kind": "contract_fulfillment",
    "message": "model_designer complete-file response was invalid: ValueError: agent returned complete files with no changes",
    "traceback": "Traceback (most recent call last):\n  File \"/mnt/d/tehpengagent/src/ernest/agentic_recsys/overseer.py\", line 372, in request_and_apply_replacements\n    apply_agent_replacements(sandbox, replacements, AGENT_FILES[agent])\n  File \"/mnt/d/tehpengagent/src/ernest/agentic_recsys/sandbox.py\", line 104, in apply_agent_replacements\n    raise ValueError(\"agent returned complete files with no changes\")\nValueError: agent returned complete files with no changes\n",
    "responsible_agents": [
      "model_designer"
    ],
    "attempt": 1,
    "return_code": null
  }
]
```

## Experiment 2

- Generation: 2
- Parent experiment: 1
- Status: scored
- Hypothesis: Use a pairwise within-user ranking loss instead of pointwise binary cross-entropy while retaining the parent architecture and features.
- Validation GAUC: 0.6665342542869394
- Validation nDCG@5: 0.5351426338754613
- Validation primary: 0.6008384440812004
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -2,6 +2,7 @@
 
 The model remains a neutral additive scorer: every encoded feature, including
 impression-time request-context fields, contributes its own learned weight.
+Training uses a within-user pairwise logistic ranking objective.
 """
 
 import numpy as np
@@ -28,12 +29,41 @@
         return self.bias + np.sum(self.weights[features], axis=1, dtype=np.float32)
 
     def step(self, features, labels):
-        size = len(labels)
+        features = np.asarray(features, dtype=np.intp)
+        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
+        if features.ndim != 2 or features.shape[0] != labels.shape[0]:
+            raise ValueError("features and labels must contain the same number of rows")
+
         logits = self.logits(features)
-        probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+        example_gradient = np.zeros(len(labels), dtype=np.float32)
+        pair_count = 0
+        loss_total = 0.0
+        user_ids = features[:, 0]
+
+        for user_id in np.unique(user_ids):
+            rows = np.flatnonzero(user_ids == user_id)
+            positive_rows = rows[labels[rows] > 0.5]
+            negative_rows = rows[labels[rows] <= 0.5]
+            if len(positive_rows) == 0 or len(negative_rows) == 0:
+                continue
+
+            pair_positive = np.repeat(positive_rows, len(negative_rows))
+            pair_negative = np.tile(negative_rows, len(positive_rows))
+            differences = logits[pair_positive] - logits[pair_negative]
+            pair_probability = sigmoid(-differences).astype(np.float32)
+            np.add.at(example_gradient, pair_positive, -pair_probability)
+            np.add.at(example_gradient, pair_negative, pair_probability)
+            loss_total += float(np.logaddexp(0.0, -differences).sum())
+            pair_count += len(differences)
+
+        if pair_count:
+            example_gradient /= pair_count
+            loss = loss_total / pair_count
+        else:
+            loss = 0.0
+
         grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
+        np.add.at(grad_weights, features, example_gradient[:, None])
         grad_weights += self.l2 * self.weights
         self.step_number += 1
         beta1, beta2, epsilon = 0.9, 0.999, 1e-8
@@ -44,11 +74,8 @@
         first_hat = self.first_moment / (1 - beta1 ** self.step_number)
         second_hat = self.second_moment / (1 - beta2 ** self.step_number)
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
-        self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
-            labels * np.log(probabilities + 1e-9)
-            + (1 - labels) * np.log(1 - probabilities + 1e-9)
-        ))
+        self.bias -= self.learning_rate * example_gradient.sum()
+        return float(loss)
 
     def predict(self, features, batch_size=200_000):
         features = np.asarray(features, dtype=np.intp)
```
