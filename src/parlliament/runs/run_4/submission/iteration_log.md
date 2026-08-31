# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add the impression-time request_context feature family—tab, hour, and weekday—to the neutral user/video ID scorer, leaving architecture, optimizer, and loss unchanged. This tests whether observed context signal improves within-user ordering.
- Validation GAUC: 0.6689116170510226
- Validation nDCG@5: 0.5369580777053825
- Validation primary: 0.6029348473782026
- Failure stage: none
- Failure reason: none
- Recovery: The Overseer classified each failure, routed it to the responsible code agent, and retried within the configured attempt and wall-clock limits.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -1,6 +1,8 @@
-"""Neutral seed data contract using only the task's user and item identifiers."""
+"""Neutral seed data contract using user, item, and request-context identifiers."""
 
 import csv
+import datetime
+import math
 import os
 
 import numpy as np
@@ -12,7 +14,16 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+
+
+def _hour(hourmin):
+    value = int(math.floor(float(hourmin) / 100.0))
+    return max(0, min(23, value))
+
+
+def _weekday(date):
+    return datetime.datetime.strptime(str(date), "%Y%m%d").weekday()
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -23,8 +34,14 @@
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
+                    _hour(row["hourmin"]),
+                    _weekday(date),
                     1 if row[LABEL] != "0" else 0,
                 ))
     result = {}
@@ -38,8 +55,9 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
+
     def raw(row):
-        return [row[1], row[2]]
+        return [row[1], row[2], row[3], row[4], row[5]]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
@@ -54,12 +72,14 @@
         features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
         labels = np.empty(len(rows), dtype=np.float32)
         users = np.empty(len(rows), dtype="U32")
+        row_ids = np.arange(len(rows), dtype=np.int64)
         for row_index, row in enumerate(rows):
             for field_index, value in enumerate(raw(row)):
                 features[row_index, field_index] = (
-                    vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
+                    vocabs[field_index].get(value, unknown[field_index])
+                    + offsets[field_index]
                 )
-            labels[row_index] = row[3]
+            labels[row_index] = row[6]
             users[row_index] = row[1]
-        encoded[name] = (features, labels, users)
+        encoded[name] = (features, labels, users, row_ids)
     return encoded, int(sum(dimensions))
```

#### `model.py`

```diff
--- model.py
+++ model.py
@@ -9,6 +9,8 @@
 
 class Model:
     def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
+        # The shared vector covers every encoded column supplied by data.py,
+        # including the user, video, and request-context fields.
         self.weights = np.zeros(dimension, dtype=np.float32)
         self.bias = np.float32(0.0)
         self.learning_rate = learning_rate
@@ -18,7 +20,7 @@
         self.step_number = 0
 
     def logits(self, features):
-        return self.bias + self.weights[features].sum(1)
+        return self.bias + self.weights[features].sum(axis=1)
 
     def step(self, features, labels):
         size = len(labels)
@@ -26,7 +28,11 @@
         probabilities = sigmoid(logits)
         gradient = ((probabilities - labels) / size).astype(np.float32)
         grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
+        np.add.at(
+            grad_weights,
+            features,
+            np.broadcast_to(gradient[:, None], features.shape),
+        )
         grad_weights += self.l2 * self.weights
         self.step_number += 1
         beta1, beta2, epsilon = 0.9, 0.999, 1e-8
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -1,4 +1,4 @@
-"""Train the neutral seed scaffold and emit validation scores for fixed evaluation."""
+"""Train the context-augmented additive scorer and emit validation and test scores."""
 
 import argparse
 import json
@@ -11,7 +11,7 @@
 
 
 def within_user_auc(user_ids, labels, scores):
-    """Training-only early-stop proxy; final scoring is owned by the Experimentor."""
+    """Compute the validation-only within-user AUC proxy."""
     grouped = {}
     for user, label, score in zip(user_ids, labels, scores):
         grouped.setdefault(str(user), []).append((float(score), int(label)))
@@ -35,6 +35,16 @@
     return numerator / denominator if denominator else 0.5
 
 
+def _unpack_train_or_test(encoded_split):
+    if len(encoded_split) == 3:
+        features, labels, row_ids = encoded_split
+    elif len(encoded_split) == 4:
+        features, labels, _, row_ids = encoded_split
+    else:
+        raise ValueError("encoded split must contain three or four values")
+    return features, labels, row_ids
+
+
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", required=True)
@@ -42,30 +52,44 @@
     parser.add_argument("--output", required=True)
     parser.add_argument("--contract-check", action="store_true")
     args = parser.parse_args()
+
     with open(args.config, encoding="utf-8") as handle:
         config = json.load(handle)
+
     splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
     encoded, dimension = encode(splits)
-    train_features, train_labels, _ = encoded["train"]
-    valid_features, valid_labels, valid_users = encoded["valid"]
-    test_features, test_labels, _ = encoded["test"]
+    train_features, train_labels, _train_row_ids = _unpack_train_or_test(encoded["train"])
+    valid_features, valid_labels, valid_users, valid_row_ids = encoded["valid"]
+    test_features, _test_labels, test_row_ids = _unpack_train_or_test(encoded["test"])
+
+    for name, features in (
+        ("train", train_features),
+        ("valid", valid_features),
+        ("test", test_features),
+    ):
+        if features.ndim != 2 or features.shape[1] != 5:
+            raise ValueError(f"{name} features must have shape (n, 5)")
+
     model = Model(
         dimension,
         learning_rate=config["learning_rate"],
         l2=config["l2"],
     )
+
     if args.contract_check:
         probe_size = min(8, len(train_labels))
         if probe_size == 0 or len(valid_labels) == 0:
             raise ValueError("contract probe requires non-empty train and validation slices")
         loss = model.step(train_features[:probe_size], train_labels[:probe_size])
         probe_scores = model.predict(valid_features[:probe_size])
-        if probe_scores.ndim != 1 or len(probe_scores) != min(probe_size, len(valid_features)):
+        expected_size = min(probe_size, len(valid_features))
+        if probe_scores.ndim != 1 or len(probe_scores) != expected_size:
             raise ValueError("model prediction shape violates the interface contract")
         if not np.isfinite(loss) or not np.all(np.isfinite(probe_scores)):
             raise ValueError("model produced NaN or infinity during contract probe")
         print(json.dumps({"contract": "ok", "feature_shape": list(train_features.shape)}))
         return
+
     rng = np.random.default_rng(config["seed"])
     best_score, best_state, stale = -1.0, None, 0
     for epoch in range(1, config["max_epochs"] + 1):
@@ -83,18 +107,22 @@
             stale += 1
             if stale >= config["patience"]:
                 break
+
     if best_state is None:
         raise RuntimeError("training produced no checkpoint")
+
     model.load_state(best_state)
+    valid_scores = np.asarray(model.predict(valid_features), dtype=np.float32)
+    test_scores = np.asarray(model.predict(test_features), dtype=np.float32)
     np.savez(
         args.output,
-        row_ids=np.arange(len(valid_labels), dtype=np.int64),
-        scores=model.predict(valid_features),
+        row_ids=np.asarray(valid_row_ids, dtype=np.int64),
+        scores=valid_scores,
     )
     np.savez(
         Path(args.output).with_name("predictions_test.npz"),
-        row_ids=np.arange(len(test_labels), dtype=np.int64),
-        scores=model.predict(test_features),
+        row_ids=np.asarray(test_row_ids, dtype=np.int64),
+        scores=test_scores,
     )
```

### Error and recovery events

```json
[
  {
    "kind": "contract_fulfillment",
    "message": "model_designer complete-file response was invalid: ValueError: agent returned complete files with no changes",
    "traceback": "Traceback (most recent call last):\n  File \"/mnt/d/parlliament/src/parlliament/parlliament/overseer.py\", line 416, in request_and_apply_replacements\n    apply_agent_replacements(sandbox, replacements, AGENT_FILES[agent])\n  File \"/mnt/d/parlliament/src/parlliament/parlliament/sandbox.py\", line 104, in apply_agent_replacements\n    raise ValueError(\"agent returned complete files with no changes\")\nValueError: agent returned complete files with no changes\n",
    "responsible_agents": [
      "model_designer"
    ],
    "attempt": 1,
    "return_code": null
  },
  {
    "kind": "contract_usage",
    "message": "contract probe exited 1",
    "traceback": "Traceback (most recent call last):\n  File \"/mnt/d/parlliament/src/parlliament/runs/run_4/attempts/attempt_432f633d660e/train.py\", line 118, in <module>\n    main()\n  File \"/mnt/d/parlliament/src/parlliament/runs/run_4/attempts/attempt_432f633d660e/train.py\", line 51, in main\n    train_features, train_labels, _, _ = encoded[\"train\"]\n    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nValueError: not enough values to unpack (expected 4, got 3)\n",
    "responsible_agents": [
      "feature_engineer",
      "model_designer",
      "trainer"
    ],
    "attempt": 1,
    "return_code": 1
  }
]
```

## Experiment 2

- Generation: 2
- Parent experiment: 1
- Status: scored
- Hypothesis: Replace pointwise BCE with within-user RankNet pairwise loss while retaining the current user, video, and request-context inputs, optimizer family, and canonical displayed-candidate evaluation.
- Validation GAUC: 0.6685063951303031
- Validation nDCG@5: 0.5367903359407852
- Validation primary: 0.6026483655355441
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -1,4 +1,4 @@
-"""Fresh-start additive ID model with no interaction or baseline-derived architecture."""
+"""Fresh-start additive ID model with within-user RankNet optimization."""
 
 import numpy as np
 
@@ -22,16 +22,43 @@
     def logits(self, features):
         return self.bias + self.weights[features].sum(axis=1)
 
-    def step(self, features, labels):
-        size = len(labels)
+    def step(self, features, labels, user_ids):
+        labels = np.asarray(labels)
+        user_ids = np.asarray(user_ids)
         logits = self.logits(features)
-        probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+        row_gradient = np.zeros(len(labels), dtype=np.float64)
+        total_loss = 0.0
+        pair_count = 0
+
+        for user_id in np.unique(user_ids):
+            group = np.flatnonzero(user_ids == user_id)
+            positive = group[labels[group] == 1]
+            negative = group[labels[group] == 0]
+            if len(positive) == 0 or len(negative) == 0:
+                continue
+
+            differences = (
+                logits[positive, None].astype(np.float64)
+                - logits[negative][None, :].astype(np.float64)
+            )
+            pair_probabilities = sigmoid(differences)
+            positive_gradient = pair_probabilities - 1.0
+            negative_gradient = 1.0 - pair_probabilities
+
+            row_gradient[positive] += positive_gradient.sum(axis=1)
+            row_gradient[negative] += negative_gradient.sum(axis=0)
+            total_loss += float(np.logaddexp(0.0, -differences).sum())
+            pair_count += differences.size
+
+        if pair_count == 0:
+            return 0.0
+
+        row_gradient /= pair_count
         grad_weights = np.zeros_like(self.weights)
         np.add.at(
             grad_weights,
             features,
-            np.broadcast_to(gradient[:, None], features.shape),
+            np.broadcast_to(row_gradient[:, None], features.shape),
         )
         grad_weights += self.l2 * self.weights
         self.step_number += 1
@@ -43,11 +70,8 @@
         first_hat = self.first_moment / (1 - beta1 ** self.step_number)
         second_hat = self.second_moment / (1 - beta2 ** self.step_number)
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
-        self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
-            labels * np.log(probabilities + 1e-9)
-            + (1 - labels) * np.log(1 - probabilities + 1e-9)
-        ))
+        self.bias -= self.learning_rate * row_gradient.sum()
+        return float(total_loss / pair_count)
 
     def predict(self, features, batch_size=200_000):
         return np.concatenate([
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -35,14 +35,46 @@
     return numerator / denominator if denominator else 0.5
 
 
-def _unpack_train_or_test(encoded_split):
-    if len(encoded_split) == 3:
-        features, labels, row_ids = encoded_split
-    elif len(encoded_split) == 4:
-        features, labels, _, row_ids = encoded_split
-    else:
-        raise ValueError("encoded split must contain three or four values")
-    return features, labels, row_ids
+def _assert_artifact(path, row_ids, scores):
+    row_ids = np.asarray(row_ids)
+    scores = np.asarray(scores)
+    if row_ids.ndim != 1 or scores.ndim != 1:
+        raise ValueError("prediction artifact arrays must be one-dimensional")
+    if len(row_ids) != len(scores):
+        raise ValueError("prediction artifact arrays must have equal lengths")
+    with np.load(path) as artifact:
+        if set(artifact.files) != {"row_ids", "scores"}:
+            raise ValueError("prediction artifact has an unexpected key set")
+        if artifact["row_ids"].ndim != 1 or artifact["scores"].ndim != 1:
+            raise ValueError("prediction artifact arrays must be one-dimensional")
+        if len(artifact["row_ids"]) != len(artifact["scores"]):
+            raise ValueError("prediction artifact arrays must have equal lengths")
+
+
+def _group_complete_batches(user_ids, batch_size, rng):
+    groups = {}
+    for index, user_id in enumerate(user_ids):
+        groups.setdefault(user_id, []).append(index)
+
+    group_values = list(groups.values())
+    shuffled_order = rng.permutation(len(group_values))
+    batches = []
+    current = []
+    for group_index in shuffled_order:
+        group = group_values[int(group_index)]
+        if current and len(current) + len(group) > batch_size:
+            batches.append(np.asarray(current, dtype=np.int64))
+            current = []
+        if len(group) > batch_size:
+            if current:
+                batches.append(np.asarray(current, dtype=np.int64))
+                current = []
+            batches.append(np.asarray(group, dtype=np.int64))
+        else:
+            current.extend(group)
+    if current:
+        batches.append(np.asarray(current, dtype=np.int64))
+    return batches
 
 
 def main():
@@ -58,9 +90,10 @@
 
     splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
     encoded, dimension = encode(splits)
-    train_features, train_labels, _train_row_ids = _unpack_train_or_test(encoded["train"])
+    train_features, train_labels, train_users, _train_row_ids = encoded["train"]
     valid_features, valid_labels, valid_users, valid_row_ids = encoded["valid"]
-    test_features, _test_labels, test_row_ids = _unpack_train_or_test(encoded["test"])
+    test_features = encoded["test"][0]
+    test_row_ids = encoded["test"][3]
 
     for name, features in (
         ("train", train_features),
@@ -80,7 +113,11 @@
         probe_size = min(8, len(train_labels))
         if probe_size == 0 or len(valid_labels) == 0:
             raise ValueError("contract probe requires non-empty train and validation slices")
-        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
+        loss = model.step(
+            train_features[:probe_size],
+            train_labels[:probe_size],
+            train_users[:probe_size],
+        )
         probe_scores = model.predict(valid_features[:probe_size])
         expected_size = min(probe_size, len(valid_features))
         if probe_scores.ndim != 1 or len(probe_scores) != expected_size:
@@ -93,11 +130,15 @@
     rng = np.random.default_rng(config["seed"])
     best_score, best_state, stale = -1.0, None, 0
     for epoch in range(1, config["max_epochs"] + 1):
-        order = rng.permutation(len(train_labels))
         losses = []
-        for index in range(0, len(order), config["batch_size"]):
-            batch = order[index:index + config["batch_size"]]
-            losses.append(model.step(train_features[batch], train_labels[batch]))
+        for batch in _group_complete_batches(train_users, config["batch_size"], rng):
+            losses.append(
+                model.step(
+                    train_features[batch],
+                    train_labels[batch],
+                    train_users[batch],
+                )
+            )
         predictions = model.predict(valid_features)
         proxy = within_user_auc(valid_users, valid_labels, predictions)
         print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_gauc_proxy={proxy:.6f}")
@@ -114,16 +155,14 @@
     model.load_state(best_state)
     valid_scores = np.asarray(model.predict(valid_features), dtype=np.float32)
     test_scores = np.asarray(model.predict(test_features), dtype=np.float32)
-    np.savez(
-        args.output,
-        row_ids=np.asarray(valid_row_ids, dtype=np.int64),
-        scores=valid_scores,
-    )
-    np.savez(
-        Path(args.output).with_name("predictions_test.npz"),
-        row_ids=np.asarray(test_row_ids, dtype=np.int64),
-        scores=test_scores,
-    )
+    valid_path = Path(args.output)
+    test_path = valid_path.with_name("predictions_test.npz")
+    valid_row_ids = np.asarray(valid_row_ids, dtype=np.int64)
+    test_row_ids = np.asarray(test_row_ids, dtype=np.int64)
+    np.savez(valid_path, row_ids=valid_row_ids, scores=valid_scores)
+    np.savez(test_path, row_ids=test_row_ids, scores=test_scores)
+    _assert_artifact(valid_path, valid_row_ids, valid_scores)
+    _assert_artifact(test_path, test_row_ids, test_scores)
 
 
 if __name__ == "__main__":
```

## Experiment 3

- Generation: 3
- Parent experiment: 2
- Status: scored
- Hypothesis: Use a hybrid loss that combines within-user RankNet with a small pointwise BCE anchor over the same displayed impressions.
- Validation GAUC: 0.668322714126043
- Validation nDCG@5: 0.5365871373160064
- Validation primary: 0.6024549257210248
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -50,10 +50,21 @@
             total_loss += float(np.logaddexp(0.0, -differences).sum())
             pair_count += differences.size
 
-        if pair_count == 0:
-            return 0.0
+        bce_logits = logits.astype(np.float64)
+        bce_labels = labels.astype(np.float64)
+        bce_loss = np.logaddexp(0.0, -bce_logits) - bce_labels * bce_logits
+        bce_weight = 0.10
+        if len(labels) > 0:
+            row_gradient /= pair_count if pair_count else 1
+            row_gradient += bce_weight * (
+                sigmoid(bce_logits) - bce_labels
+            ) / len(labels)
+            total_loss = total_loss / pair_count if pair_count else 0.0
+            total_loss += bce_weight * float(bce_loss.mean())
+        elif pair_count:
+            row_gradient /= pair_count
+            total_loss /= pair_count
 
-        row_gradient /= pair_count
         grad_weights = np.zeros_like(self.weights)
         np.add.at(
             grad_weights,
@@ -71,7 +82,7 @@
         second_hat = self.second_moment / (1 - beta2 ** self.step_number)
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
         self.bias -= self.learning_rate * row_gradient.sum()
-        return float(total_loss / pair_count)
+        return float(total_loss)
 
     def predict(self, features, batch_size=200_000):
         return np.concatenate([
```

## Experiment 4

- Generation: 4
- Parent experiment: 1
- Status: scored
- Hypothesis: Use context-conditioned linear interactions between each context field and the video identity.
- Validation GAUC: 0.6573385117880062
- Validation nDCG@5: 0.5323570976841144
- Validation primary: 0.5948478047360604
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -14,7 +14,16 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+FIELDS = [
+    "user_id",
+    "video_id",
+    "tab",
+    "hour",
+    "weekday",
+    "video_id-tab",
+    "video_id-hour",
+    "video_id-weekday",
+]
 
 
 def _hour(hourmin):
@@ -57,7 +66,20 @@
         raise ValueError("training split is empty")
 
     def raw(row):
-        return [row[1], row[2], row[3], row[4], row[5]]
+        video_id = row[2]
+        tab = row[3]
+        hour = row[4]
+        weekday = row[5]
+        return [
+            row[1],
+            video_id,
+            tab,
+            hour,
+            weekday,
+            (video_id, tab),
+            (video_id, hour),
+            (video_id, weekday),
+        ]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
```

#### `model.py`

```diff
--- model.py
+++ model.py
@@ -10,7 +10,7 @@
 class Model:
     def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
         # The shared vector covers every encoded column supplied by data.py,
-        # including the user, video, and request-context fields.
+        # including the user, video, request-context, and supplied interaction fields.
         self.weights = np.zeros(dimension, dtype=np.float32)
         self.bias = np.float32(0.0)
         self.learning_rate = learning_rate
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -45,6 +45,24 @@
     return features, labels, row_ids
 
 
+def _check_prediction_archive(path, expected_row_ids, expected_scores):
+    with np.load(path, allow_pickle=False) as archive:
+        if set(archive.files) != {"row_ids", "scores"}:
+            raise ValueError(f"{path} must contain exactly row_ids and scores")
+        row_ids = archive["row_ids"]
+        scores = archive["scores"]
+        if row_ids.ndim != 1 or scores.ndim != 1:
+            raise ValueError(f"{path} arrays must be one-dimensional")
+        if len(row_ids) != len(scores):
+            raise ValueError(f"{path} row_ids and scores must have matching lengths")
+        if len(row_ids) != len(expected_row_ids) or len(scores) != len(expected_scores):
+            raise ValueError(f"{path} has an unexpected number of predictions")
+        if row_ids.dtype != np.dtype(np.int64):
+            raise ValueError(f"{path} row_ids must have dtype int64")
+        if scores.dtype != np.dtype(np.float32):
+            raise ValueError(f"{path} scores must have dtype float32")
+
+
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", required=True)
@@ -60,15 +78,15 @@
     encoded, dimension = encode(splits)
     train_features, train_labels, _train_row_ids = _unpack_train_or_test(encoded["train"])
     valid_features, valid_labels, valid_users, valid_row_ids = encoded["valid"]
-    test_features, _test_labels, test_row_ids = _unpack_train_or_test(encoded["test"])
+    test_features, _, test_row_ids = _unpack_train_or_test(encoded["test"])
 
     for name, features in (
         ("train", train_features),
         ("valid", valid_features),
         ("test", test_features),
     ):
-        if features.ndim != 2 or features.shape[1] != 5:
-            raise ValueError(f"{name} features must have shape (n, 5)")
+        if features.ndim != 2 or features.shape[1] != 8:
+            raise ValueError(f"{name} features must have shape (n, 8)")
 
     model = Model(
         dimension,
@@ -114,16 +132,20 @@
     model.load_state(best_state)
     valid_scores = np.asarray(model.predict(valid_features), dtype=np.float32)
     test_scores = np.asarray(model.predict(test_features), dtype=np.float32)
+    valid_output = Path(args.output)
+    test_output = valid_output.with_name("predictions_test.npz")
     np.savez(
-        args.output,
+        valid_output,
         row_ids=np.asarray(valid_row_ids, dtype=np.int64),
         scores=valid_scores,
     )
     np.savez(
-        Path(args.output).with_name("predictions_test.npz"),
+        test_output,
         row_ids=np.asarray(test_row_ids, dtype=np.int64),
         scores=test_scores,
     )
+    _check_prediction_archive(valid_output, valid_row_ids, valid_scores)
+    _check_prediction_archive(test_output, test_row_ids, test_scores)
 
 
 if __name__ == "__main__":
```
