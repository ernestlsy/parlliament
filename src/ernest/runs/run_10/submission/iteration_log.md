# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add request-context categorical features (tab, hour, weekday) to the additive user/video-ID scorer.
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
@@ -1,6 +1,7 @@
-"""Neutral seed data contract using only the task's user and item identifiers."""
+"""Data contract for additive ID and request-context categorical features."""
 
 import csv
+import datetime
 import os
 
 import numpy as np
@@ -12,7 +13,7 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -23,8 +24,20 @@
     ):
         with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
             for row in csv.DictReader(handle):
+                date_value = int(row["date"])
+                hour = int(float(row["hourmin"]) // 100)
+                if not 0 <= hour <= 23:
+                    raise ValueError("hour derived from hourmin is out of range")
+                weekday = datetime.datetime.strptime(
+                    str(date_value), "%Y%m%d"
+                ).weekday()
                 rows.append((
-                    int(row["date"]), row["user_id"], row["video_id"],
+                    date_value,
+                    row["user_id"],
+                    row["video_id"],
+                    row["tab"],
+                    hour,
+                    weekday,
                     1 if row[LABEL] != "0" else 0,
                 ))
     result = {}
@@ -38,8 +51,9 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
+
     def raw(row):
-        return [row[1], row[2]]
+        return [row[1], row[2], row[3], row[4], row[5]]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
@@ -59,7 +73,7 @@
                 features[row_index, field_index] = (
                     vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
                 )
-            labels[row_index] = row[3]
+            labels[row_index] = row[6]
             users[row_index] = row[1]
         encoded[name] = (features, labels, users)
     return encoded, int(sum(dimensions))
```

## Experiment 2

- Generation: 2
- Parent experiment: 1
- Status: scored
- Hypothesis: Add a categorical tab-by-hour interaction feature to the additive scorer.
- Validation GAUC: 0.6691931537691496
- Validation nDCG@5: 0.5365763254446154
- Validation primary: 0.6028847396068825
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -13,7 +13,7 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday", "tab_hour"]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -53,7 +53,7 @@
         raise ValueError("training split is empty")
 
     def raw(row):
-        return [row[1], row[2], row[3], row[4], row[5]]
+        return [row[1], row[2], row[3], row[4], row[5], (row[3], row[4])]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
```

## Experiment 3

- Generation: 3
- Parent experiment: 1
- Status: scored
- Hypothesis: Use pairwise within-user BPR loss instead of pointwise binary cross-entropy.
- Validation GAUC: 0.6704966836327051
- Validation nDCG@5: 0.537679764075165
- Validation primary: 0.6040882238539351
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -20,14 +20,19 @@
     def logits(self, features):
         return self.bias + self.weights[features].sum(1)
 
-    def step(self, features, labels):
-        size = len(labels)
-        logits = self.logits(features)
-        probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+    def step(self, positive_features, negative_features):
+        pair_count = len(positive_features)
+        positive_logits = self.logits(positive_features)
+        negative_logits = self.logits(negative_features)
+        margins = positive_logits - negative_logits
+        loss = np.mean(np.logaddexp(0.0, -margins))
+        pair_gradient = ((sigmoid(margins) - 1.0) / pair_count).astype(np.float32)
+
         grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
+        np.add.at(grad_weights, positive_features, pair_gradient[:, None])
+        np.add.at(grad_weights, negative_features, -pair_gradient[:, None])
         grad_weights += self.l2 * self.weights
+
         self.step_number += 1
         beta1, beta2, epsilon = 0.9, 0.999, 1e-8
         self.first_moment *= beta1
@@ -37,11 +42,8 @@
         first_hat = self.first_moment / (1 - beta1 ** self.step_number)
         second_hat = self.second_moment / (1 - beta2 ** self.step_number)
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
-        self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
-            labels * np.log(probabilities + 1e-9)
-            + (1 - labels) * np.log(1 - probabilities + 1e-9)
-        ))
+
+        return float(loss)
 
     def predict(self, features, batch_size=200_000):
         return np.concatenate([
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -34,6 +34,30 @@
     return numerator / denominator if denominator else 0.5
 
 
+def sample_within_user_pairs(users, labels, rng):
+    """Sample one within-user negative for every eligible positive impression."""
+    pools = {}
+    for index, (user, label) in enumerate(zip(users, labels)):
+        positive_indices, negative_indices = pools.setdefault(user, ([], []))
+        if label == 1:
+            positive_indices.append(index)
+        elif label == 0:
+            negative_indices.append(index)
+
+    positive_pairs = []
+    negative_pairs = []
+    for positive_indices, negative_indices in pools.values():
+        if not positive_indices or not negative_indices:
+            continue
+        positive_pairs.extend(positive_indices)
+        negative_pairs.extend(
+            rng.choice(negative_indices, size=len(positive_indices)).tolist()
+        )
+    return np.asarray(positive_pairs, dtype=np.int64), np.asarray(
+        negative_pairs, dtype=np.int64
+    )
+
+
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", required=True)
@@ -45,33 +69,53 @@
         config = json.load(handle)
     splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
     encoded, dimension = encode(splits)
-    train_features, train_labels, _ = encoded["train"]
+    train_features, train_labels, train_users = encoded["train"]
     valid_features, valid_labels, valid_users = encoded[config["split"]]
     model = Model(
         dimension,
         learning_rate=config["learning_rate"],
         l2=config["l2"],
     )
+    rng = np.random.default_rng(config["seed"])
     if args.contract_check:
-        probe_size = min(8, len(train_labels))
-        if probe_size == 0 or len(valid_labels) == 0:
+        positive_indices, negative_indices = sample_within_user_pairs(
+            train_users, train_labels, rng
+        )
+        if len(positive_indices) == 0:
+            raise ValueError(
+                "BPR contract probe requires at least one user with both long_view classes."
+            )
+        if len(valid_labels) == 0:
             raise ValueError("contract probe requires non-empty train and validation slices")
-        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
+        probe_size = min(8, len(positive_indices), len(valid_labels))
+        loss = model.step(
+            train_features[positive_indices[:probe_size]],
+            train_features[negative_indices[:probe_size]],
+        )
         probe_scores = model.predict(valid_features[:probe_size])
-        if probe_scores.ndim != 1 or len(probe_scores) != min(probe_size, len(valid_features)):
+        if probe_scores.ndim != 1 or len(probe_scores) != probe_size:
             raise ValueError("model prediction shape violates the interface contract")
         if not np.isfinite(loss) or not np.all(np.isfinite(probe_scores)):
             raise ValueError("model produced NaN or infinity during contract probe")
         print(json.dumps({"contract": "ok", "feature_shape": list(train_features.shape)}))
         return
-    rng = np.random.default_rng(config["seed"])
     best_score, best_state, stale = -1.0, None, 0
     for epoch in range(1, config["max_epochs"] + 1):
-        order = rng.permutation(len(train_labels))
+        positive_indices, negative_indices = sample_within_user_pairs(
+            train_users, train_labels, rng
+        )
+        if len(positive_indices) == 0:
+            raise ValueError("BPR training requires at least one user with both long_view classes")
+        order = rng.permutation(len(positive_indices))
         losses = []
         for index in range(0, len(order), config["batch_size"]):
             batch = order[index:index + config["batch_size"]]
-            losses.append(model.step(train_features[batch], train_labels[batch]))
+            losses.append(
+                model.step(
+                    train_features[positive_indices[batch]],
+                    train_features[negative_indices[batch]],
+                )
+            )
         predictions = model.predict(valid_features)
         proxy = within_user_auc(valid_users, valid_labels, predictions)
         print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_gauc_proxy={proxy:.6f}")
```
