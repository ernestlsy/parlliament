# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add the predeclared request-context family of tab, hour, and weekday to the seed scorer.
- Validation GAUC: 0.6689116170510226
- Validation nDCG@5: 0.5369580777053825
- Validation primary: 0.6029348473782026
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -1,7 +1,9 @@
-"""Neutral seed data contract using only the task's user and item identifiers."""
+"""Seed data contract with training-only categorical request-context vocabularies."""
 
 import csv
+import math
 import os
+from datetime import datetime
 
 import numpy as np
 
@@ -12,20 +14,47 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+
+
+def _hour_from_hourmin(hourmin):
+    hour = math.floor(float(hourmin) / 100)
+    if not 0 <= hour <= 23:
+        raise ValueError("derived hour must be in 0..23")
+    return hour
+
+
+def _weekday_from_date(date):
+    return datetime.strptime(str(date), "%Y%m%d").weekday()
+
+
+def raw(row):
+    return [row[1], row[2], row[3], row[4], row[5]]
 
 
 def load(data_dir, max_rows_per_split=None):
     rows = []
+    test_low, test_high = SPLITS["test"]
     for filename in (
         "log_standard_4_08_to_4_21_pure.csv",
         "log_standard_4_22_to_5_08_pure.csv",
     ):
         with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
             for row in csv.DictReader(handle):
+                date = int(row["date"])
+                hour = _hour_from_hourmin(row["hourmin"])
+                weekday = _weekday_from_date(date)
+                label = None
+                if not test_low <= date <= test_high:
+                    label = 1 if row[LABEL] != "0" else 0
                 rows.append((
-                    int(row["date"]), row["user_id"], row["video_id"],
-                    1 if row[LABEL] != "0" else 0,
+                    date,
+                    row["user_id"],
+                    row["video_id"],
+                    row["tab"],
+                    hour,
+                    weekday,
+                    label,
                 ))
     result = {}
     for name, (low, high) in SPLITS.items():
@@ -38,8 +67,6 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
-    def raw(row):
-        return [row[1], row[2]]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
@@ -52,14 +79,15 @@
     encoded = {}
     for name, rows in splits.items():
         features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
-        labels = np.empty(len(rows), dtype=np.float32)
+        labels = None if name == "test" else np.empty(len(rows), dtype=np.float32)
         users = np.empty(len(rows), dtype="U32")
         for row_index, row in enumerate(rows):
             for field_index, value in enumerate(raw(row)):
                 features[row_index, field_index] = (
                     vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
                 )
-            labels[row_index] = row[3]
+            if labels is not None:
+                labels[row_index] = row[6]
             users[row_index] = row[1]
         encoded[name] = (features, labels, users)
     return encoded, int(sum(dimensions))
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -1,4 +1,4 @@
-"""Train the neutral seed scaffold and emit validation scores for fixed evaluation."""
+"""Train the neutral seed scaffold and emit validation and test prediction artifacts."""
 
 import argparse
 import json
@@ -48,7 +48,7 @@
     encoded, dimension = encode(splits)
     train_features, train_labels, _ = encoded["train"]
     valid_features, valid_labels, valid_users = encoded["valid"]
-    test_features, test_labels, _ = encoded["test"]
+    test_features, _, _ = encoded["test"]
     model = Model(
         dimension,
         learning_rate=config["learning_rate"],
@@ -86,14 +86,15 @@
     if best_state is None:
         raise RuntimeError("training produced no checkpoint")
     model.load_state(best_state)
+    output_dir = Path(args.output).parent
     np.savez(
-        args.output,
-        row_ids=np.arange(len(valid_labels), dtype=np.int64),
+        output_dir / "predictions_valid.npz",
+        row_ids=np.arange(len(valid_features), dtype=np.int64),
         scores=model.predict(valid_features),
     )
     np.savez(
-        Path(args.output).with_name("predictions_test.npz"),
-        row_ids=np.arange(len(test_labels), dtype=np.int64),
+        output_dir / "predictions_test.npz",
+        row_ids=np.arange(len(test_features), dtype=np.int64),
         scores=model.predict(test_features),
     )
```

## Experiment 2

- Generation: 2
- Parent experiment: 1
- Status: scored
- Hypothesis: Replace row-averaged BCE with user-normalized BCE so each user's training impressions contribute equal total loss mass.
- Validation GAUC: 0.663056132708663
- Validation nDCG@5: 0.533770726157447
- Validation primary: 0.598413429433055
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -20,11 +20,25 @@
     def logits(self, features):
         return self.bias + self.weights[features].sum(1)
 
-    def step(self, features, labels):
+    def step(self, features, labels, sample_weights):
+        sample_weights = np.asarray(sample_weights)
+        if sample_weights.ndim != 1:
+            raise ValueError("sample_weights must be one-dimensional")
+        if len(sample_weights) != len(labels):
+            raise ValueError("sample_weights must have the same length as labels")
+        try:
+            finite_weights = np.isfinite(sample_weights)
+        except TypeError as error:
+            raise ValueError("sample_weights must be finite") from error
+        if not np.all(finite_weights):
+            raise ValueError("sample_weights must be finite")
+
         size = len(labels)
         logits = self.logits(features)
         probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+        gradient = (
+            (probabilities - labels) * sample_weights / size
+        ).astype(np.float32)
         grad_weights = np.zeros_like(self.weights)
         np.add.at(grad_weights, features, gradient[:, None])
         grad_weights += self.l2 * self.weights
@@ -38,10 +52,10 @@
         second_hat = self.second_moment / (1 - beta2 ** self.step_number)
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
         self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
+        return float(np.mean(sample_weights * -(
             labels * np.log(probabilities + 1e-9)
             + (1 - labels) * np.log(1 - probabilities + 1e-9)
-        ))
+        )))
 
     def predict(self, features, batch_size=200_000):
         return np.concatenate([
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -46,7 +46,20 @@
         config = json.load(handle)
     splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
     encoded, dimension = encode(splits)
-    train_features, train_labels, _ = encoded["train"]
+    train_features, train_labels, train_users = encoded["train"]
+    if len(train_users) == 0:
+        raise ValueError("training split must be non-empty")
+    _, user_inverse, user_counts = np.unique(
+        train_users, return_inverse=True, return_counts=True
+    )
+    preliminary_weights = (1.0 / user_counts[user_inverse]).astype(np.float32)
+    train_weights = (
+        preliminary_weights * (len(train_users) / preliminary_weights.sum())
+    ).astype(np.float32)
+    if len(train_weights) != len(train_labels):
+        raise ValueError("training weights must align with training labels")
+    if not np.all(np.isfinite(train_weights)):
+        raise ValueError("training weights must be finite")
     valid_features, valid_labels, valid_users = encoded["valid"]
     test_features, _, _ = encoded["test"]
     model = Model(
@@ -58,7 +71,11 @@
         probe_size = min(8, len(train_labels))
         if probe_size == 0 or len(valid_labels) == 0:
             raise ValueError("contract probe requires non-empty train and validation slices")
-        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
+        loss = model.step(
+            train_features[:probe_size],
+            train_labels[:probe_size],
+            train_weights[:probe_size],
+        )
         probe_scores = model.predict(valid_features[:probe_size])
         if probe_scores.ndim != 1 or len(probe_scores) != min(probe_size, len(valid_features)):
             raise ValueError("model prediction shape violates the interface contract")
@@ -73,7 +90,13 @@
         losses = []
         for index in range(0, len(order), config["batch_size"]):
             batch = order[index:index + config["batch_size"]]
-            losses.append(model.step(train_features[batch], train_labels[batch]))
+            losses.append(
+                model.step(
+                    train_features[batch],
+                    train_labels[batch],
+                    train_weights[batch],
+                )
+            )
         predictions = model.predict(valid_features)
         proxy = within_user_auc(valid_users, valid_labels, predictions)
         print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_gauc_proxy={proxy:.6f}")
```

## Experiment 3

- Generation: 3
- Parent experiment: 1
- Status: scored
- Hypothesis: Add a video-by-tab categorical interaction to learn surface-specific item appeal.
- Validation GAUC: 0.6680981874521777
- Validation nDCG@5: 0.535791111677589
- Validation primary: 0.6019446495648834
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -14,7 +14,12 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+BASE_FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+FIELDS = BASE_FIELDS + ["hashed_video_id_tab"]
+INTERACTION_DIMENSION = 262144
+_FNV_OFFSET_BASIS_64 = 14695981039346656037
+_FNV_PRIME_64 = 1099511628211
+_FNV_MASK_64 = (1 << 64) - 1
 
 
 def _hour_from_hourmin(hourmin):
@@ -28,8 +33,24 @@
     return datetime.strptime(str(date), "%Y%m%d").weekday()
 
 
+def _video_tab_bucket(video_id, tab):
+    value = (str(video_id) + "\x1f" + str(tab)).encode("utf-8")
+    hashed = _FNV_OFFSET_BASIS_64
+    for byte in value:
+        hashed ^= byte
+        hashed = (hashed * _FNV_PRIME_64) & _FNV_MASK_64
+    return hashed % INTERACTION_DIMENSION
+
+
 def raw(row):
-    return [row[1], row[2], row[3], row[4], row[5]]
+    return [
+        row[1],
+        row[2],
+        row[3],
+        row[4],
+        row[5],
+        _video_tab_bucket(row[2], row[3]),
+    ]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -68,13 +89,13 @@
     if not train:
         raise ValueError("training split is empty")
 
-    vocabs = [dict() for _ in FIELDS]
+    vocabs = [dict() for _ in BASE_FIELDS]
     for row in train:
-        for index, value in enumerate(raw(row)):
+        for index, value in enumerate(raw(row)[:len(BASE_FIELDS)]):
             if value not in vocabs[index]:
                 vocabs[index][value] = len(vocabs[index])
     unknown = [len(vocab) for vocab in vocabs]
-    dimensions = [len(vocab) + 1 for vocab in vocabs]
+    dimensions = [len(vocab) + 1 for vocab in vocabs] + [INTERACTION_DIMENSION]
     offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)
     encoded = {}
     for name, rows in splits.items():
@@ -82,10 +103,14 @@
         labels = None if name == "test" else np.empty(len(rows), dtype=np.float32)
         users = np.empty(len(rows), dtype="U32")
         for row_index, row in enumerate(rows):
-            for field_index, value in enumerate(raw(row)):
+            values = raw(row)
+            for field_index, value in enumerate(values[:len(BASE_FIELDS)]):
                 features[row_index, field_index] = (
                     vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
                 )
+            features[row_index, len(BASE_FIELDS)] = (
+                values[len(BASE_FIELDS)] + offsets[len(BASE_FIELDS)]
+            )
             if labels is not None:
                 labels[row_index] = row[6]
             users[row_index] = row[1]
```

## Experiment 4

- Generation: 4
- Parent experiment: 1
- Status: scored
- Hypothesis: Change early stopping from a GAUC-only proxy to the official primary proxy computed as the mean of GAUC and nDCG@5.
- Validation GAUC: 0.6689116170510226
- Validation nDCG@5: 0.5369580777053825
- Validation primary: 0.6029348473782026
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `train.py`

```diff
--- train.py
+++ train.py
@@ -33,6 +33,29 @@
         numerator += positives * wins / (positives * negatives)
         denominator += positives
     return numerator / denominator if denominator else 0.5
+
+
+def within_user_ndcg_at_5(user_ids, labels, scores):
+    """Compute mean within-user nDCG@5 in canonical validation row order."""
+    grouped = {}
+    for index, (user, label, score) in enumerate(zip(user_ids, labels, scores)):
+        grouped.setdefault(str(user), []).append((float(score), int(label), index))
+
+    ndcgs = []
+    for rows in grouped.values():
+        ranked_rows = sorted(rows, key=lambda row: (-row[0], row[2]))
+        dcg = sum(
+            (2 ** label - 1) / np.log2(rank + 1)
+            for rank, (_, label, _) in enumerate(ranked_rows[:5], start=1)
+        )
+        ideal_labels = sorted((label for _, label, _ in rows), reverse=True)
+        idcg = sum(
+            (2 ** label - 1) / np.log2(rank + 1)
+            for rank, label in enumerate(ideal_labels[:5], start=1)
+        )
+        if idcg > 0:
+            ndcgs.append(dcg / idcg)
+    return float(np.mean(ndcgs)) if ndcgs else 0.0
 
 
 def main():
@@ -75,10 +98,17 @@
             batch = order[index:index + config["batch_size"]]
             losses.append(model.step(train_features[batch], train_labels[batch]))
         predictions = model.predict(valid_features)
-        proxy = within_user_auc(valid_users, valid_labels, predictions)
-        print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_gauc_proxy={proxy:.6f}")
-        if proxy > best_score + 1e-5:
-            best_score, best_state, stale = proxy, model.state(), 0
+        gauc_proxy = within_user_auc(valid_users, valid_labels, predictions)
+        ndcg5_proxy = within_user_ndcg_at_5(valid_users, valid_labels, predictions)
+        primary_proxy = 0.5 * gauc_proxy + 0.5 * ndcg5_proxy
+        print(
+            f"epoch={epoch} loss={np.mean(losses):.6f} "
+            f"valid_gauc_proxy={gauc_proxy:.6f} "
+            f"valid_ndcg5_proxy={ndcg5_proxy:.6f} "
+            f"valid_primary_proxy={primary_proxy:.6f}"
+        )
+        if primary_proxy > best_score + 1e-5:
+            best_score, best_state, stale = primary_proxy, model.state(), 0
         else:
             stale += 1
             if stale >= config["patience"]:
```
