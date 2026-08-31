# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add leakage-safe request-context categorical fields tab, hour, and weekday to the five-field FM baseline.
- Validation GAUC: 0.6670174915586686
- Validation nDCG@5: 0.5358419331326125
- Validation primary: 0.6014297123456406
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -1,6 +1,8 @@
-"""KuaiRand five-field baseline data loader adapted to Ernest's fixed contract."""
+"""KuaiRand seven-field request-context FM data loader."""
 
 import csv
+from datetime import datetime
+import math
 import os
 
 import numpy as np
@@ -12,7 +14,7 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
+FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "weekday"]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -38,6 +40,8 @@
                     row["tab"],
                     float(row["duration_ms"]),
                     1 if row[LABEL] != "0" else 0,
+                    row["hourmin"],
+                    row["date"],
                 ))
     result = {}
     for name, (low, high) in SPLITS.items():
@@ -62,9 +66,15 @@
     edges = _bucket_edges([row[5] for row in train])
 
     def raw(row):
+        hour = math.floor(float(row[7]) / 100)
+        if not 0 <= hour <= 23:
+            raise ValueError("invalid hour derived from hourmin: {}".format(row[7]))
+        weekday = datetime.strptime(str(int(row[8])), "%Y%m%d").weekday()
         return [
             row[1], row[2], row[3], row[4],
             str(int(np.searchsorted(edges, row[5]))),
+            str(hour),
+            str(weekday),
         ]
 
     vocabularies = [dict() for _ in FIELDS]
```

## Experiment 2

- Generation: 2
- Parent experiment: 1
- Status: scored
- Hypothesis: Add one leakage-safe categorical cross field formed as tab×hour to let the FM model assign distinct interaction effects to request surface and time of day.
- Validation GAUC: 0.665599963961718
- Validation nDCG@5: 0.5356908783200283
- Validation primary: 0.6006454211408732
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -1,4 +1,4 @@
-"""KuaiRand seven-field request-context FM data loader."""
+"""KuaiRand eight-field request-context FM data loader."""
 
 import csv
 from datetime import datetime
@@ -14,7 +14,7 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "weekday"]
+FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "weekday", "tab_hour"]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -75,6 +75,7 @@
             str(int(np.searchsorted(edges, row[5]))),
             str(hour),
             str(weekday),
+            row[4] + "\t" + str(hour),
         ]
 
     vocabularies = [dict() for _ in FIELDS]
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -107,7 +107,7 @@
         print(json.dumps({
             "contract": "ok",
             "feature_shape": list(train_features.shape),
-            "fields": 5,
+            "fields": 8,
             "interaction_dimension": config["interaction_dimension"],
         }))
         return
```

## Experiment 3

- Generation: 3
- Parent experiment: 1
- Status: scored
- Hypothesis: Use inverse user-impression-count weights in binary cross-entropy so highly active users do not dominate training updates.
- Validation GAUC: 0.6593875977895494
- Validation nDCG@5: 0.5327619814212058
- Validation primary: 0.5960747896053775
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -43,13 +43,29 @@
             summed,
         )
 
-    def step(self, features, labels):
+    def step(self, features, labels, sample_weights=None):
         size = len(labels)
         if size == 0:
             return 0.0
+        if sample_weights is not None:
+            try:
+                sample_weights = np.asarray(sample_weights, dtype=np.float32)
+            except (TypeError, ValueError) as error:
+                raise ValueError("sample_weights must be a float32-compatible vector") from error
+            if sample_weights.ndim != 1 or len(sample_weights) != size:
+                raise ValueError("sample_weights must be one-dimensional and match labels")
+            if not np.all(np.isfinite(sample_weights)):
+                raise ValueError("sample_weights must be finite")
+            if np.any(sample_weights < 0.0):
+                raise ValueError("sample_weights must be nonnegative")
         logits, field_embeddings, summed = self.logits(features)
         probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+        if sample_weights is None:
+            gradient = ((probabilities - labels) / size).astype(np.float32)
+        else:
+            gradient = (
+                (probabilities - labels) * sample_weights / size
+            ).astype(np.float32)
         embedding_gradient = np.zeros_like(self.embeddings)
         weight_gradient = np.zeros_like(self.weights)
         np.add.at(weight_gradient, features, gradient[:, None])
@@ -86,10 +102,15 @@
             second_hat = second / (1.0 - beta2 ** self.step_number)
             parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
         self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
+        if sample_weights is None:
+            return float(-np.mean(
+                labels * np.log(probabilities + 1e-9)
+                + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
+            ))
+        return float(np.mean(sample_weights * -(
             labels * np.log(probabilities + 1e-9)
             + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
-        ))
+        )))
 
     def predict(self, features, batch_size=200_000):
         if len(features) == 0:
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -70,6 +70,23 @@
     return gauc, ndcg, (gauc + ndcg) / 2.0
 
 
+def _train_weights(train_users):
+    if len(train_users) == 0:
+        raise ValueError("training data must be non-empty for user weighting")
+    user_counts = collections.Counter(train_users.tolist())
+    train_weights = np.asarray(
+        [1.0 / user_counts[user] for user in train_users.tolist()],
+        dtype=np.float32,
+    )
+    if not np.all(np.isfinite(train_weights)) or not np.all(train_weights > 0):
+        raise ValueError("training weights must be finite and positive")
+    mean_weight = train_weights.mean()
+    if not np.isfinite(mean_weight) or mean_weight <= 0:
+        raise ValueError("training weight mean must be finite and positive")
+    train_weights /= mean_weight
+    return train_weights
+
+
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", required=True)
@@ -82,7 +99,8 @@
 
     splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
     encoded, dimension = encode(splits)
-    train_features, train_labels, _ = encoded["train"]
+    train_features, train_labels, train_users = encoded["train"]
+    train_weights = _train_weights(train_users)
     valid_features, valid_labels, valid_users = encoded[config["split"]]
     model = Model(
         dimension,
@@ -96,7 +114,11 @@
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
         if probe_scores.ndim != 1 or len(probe_scores) != min(
             probe_size, len(valid_features)
@@ -119,7 +141,11 @@
         losses = []
         for index in range(0, len(order), config["batch_size"]):
             batch = order[index:index + config["batch_size"]]
-            losses.append(model.step(train_features[batch], train_labels[batch]))
+            losses.append(
+                model.step(
+                    train_features[batch], train_labels[batch], train_weights[batch]
+                )
+            )
         scores = model.predict(valid_features)
         gauc, ndcg, primary = training_selection_metrics(
             valid_users, valid_labels, scores
```
