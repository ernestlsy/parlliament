# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add impression-time request context (tab, hour, weekday) to the additive user/video-ID scorer.
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
@@ -1,7 +1,8 @@
-"""Neutral seed data contract using only the task's user and item identifiers."""
+"""Data contract for additive user, item, and request-context identifiers."""
 
 import csv
 import os
+from datetime import datetime
 
 import numpy as np
 
@@ -12,7 +13,7 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -23,8 +24,25 @@
     ):
         with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
             for row in csv.DictReader(handle):
+                date_value = row["date"]
+                user_id = row["user_id"]
+                video_id = row["video_id"]
+                tab = row["tab"]
+                hourmin = row["hourmin"]
+
+                date = int(date_value)
+                hour = int(float(hourmin) // 100)
+                if not 0 <= hour <= 23:
+                    raise ValueError("hour derived from hourmin must be in 0..23")
+                weekday = datetime.strptime(date_value, "%Y%m%d").date().weekday()
+
                 rows.append((
-                    int(row["date"]), row["user_id"], row["video_id"],
+                    date,
+                    user_id,
+                    video_id,
+                    tab,
+                    hour,
+                    weekday,
                     1 if row[LABEL] != "0" else 0,
                 ))
     result = {}
@@ -38,8 +56,9 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
+
     def raw(row):
-        return [row[1], row[2]]
+        return [row[1], row[2], row[3], row[4], row[5]]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
@@ -59,7 +78,7 @@
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
- Hypothesis: Replace the additive linear scorer with a second-order factorization machine over the existing user, video, tab, hour, and weekday fields.
- Validation GAUC: 0.6658897820004246
- Validation nDCG@5: 0.5350294342803582
- Validation primary: 0.6004596081403915
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
+"""Second-order factorization-machine model over encoded categorical IDs."""
 
 import numpy as np
 
@@ -8,49 +8,120 @@
 
 
 class Model:
-    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
+    def __init__(self, dimension, learning_rate=0.01, l2=1e-6, factor_rank=None):
+        if factor_rank is None:
+            raise ValueError("factor_rank must be provided")
+        if factor_rank <= 0:
+            raise ValueError("factor_rank must be positive")
+
         self.weights = np.zeros(dimension, dtype=np.float32)
+        rng = np.random.RandomState(0)
+        self.factors = rng.normal(
+            loc=0.0, scale=0.01, size=(dimension, factor_rank)
+        ).astype(np.float32)
         self.bias = np.float32(0.0)
         self.learning_rate = learning_rate
         self.l2 = l2
+
         self.first_moment = np.zeros_like(self.weights)
         self.second_moment = np.zeros_like(self.weights)
+        self.factor_first_moment = np.zeros_like(self.factors)
+        self.factor_second_moment = np.zeros_like(self.factors)
+        self.bias_first_moment = np.float32(0.0)
+        self.bias_second_moment = np.float32(0.0)
         self.step_number = 0
 
     def logits(self, features):
-        return self.bias + self.weights[features].sum(1)
+        selected_weights = self.weights[features]
+        selected_factors = self.factors[features]
+        factor_sums = selected_factors.sum(axis=1)
+        interactions = np.float32(0.5) * (
+            (factor_sums * factor_sums).sum(axis=1)
+            - (selected_factors * selected_factors).sum(axis=(1, 2))
+        )
+        return self.bias + selected_weights.sum(axis=1) + interactions
 
     def step(self, features, labels):
         size = len(labels)
         logits = self.logits(features)
         probabilities = sigmoid(logits)
         gradient = ((probabilities - labels) / size).astype(np.float32)
+
         grad_weights = np.zeros_like(self.weights)
         np.add.at(grad_weights, features, gradient[:, None])
         grad_weights += self.l2 * self.weights
+
+        selected_factors = self.factors[features]
+        factor_sums = selected_factors.sum(axis=1)
+        selected_factor_gradients = (
+            gradient[:, None, None]
+            * (factor_sums[:, None, :] - selected_factors)
+        )
+        grad_factors = np.zeros_like(self.factors)
+        np.add.at(grad_factors, features, selected_factor_gradients)
+        grad_factors += self.l2 * self.factors
+        grad_bias = np.float32(gradient.sum())
+
         self.step_number += 1
         beta1, beta2, epsilon = 0.9, 0.999, 1e-8
+
         self.first_moment *= beta1
         self.first_moment += (1 - beta1) * grad_weights
         self.second_moment *= beta2
         self.second_moment += (1 - beta2) * (grad_weights * grad_weights)
-        first_hat = self.first_moment / (1 - beta1 ** self.step_number)
-        second_hat = self.second_moment / (1 - beta2 ** self.step_number)
+
+        self.factor_first_moment *= beta1
+        self.factor_first_moment += (1 - beta1) * grad_factors
+        self.factor_second_moment *= beta2
+        self.factor_second_moment += (1 - beta2) * (grad_factors * grad_factors)
+
+        self.bias_first_moment = np.float32(
+            beta1 * self.bias_first_moment + (1 - beta1) * grad_bias
+        )
+        self.bias_second_moment = np.float32(
+            beta2 * self.bias_second_moment + (1 - beta2) * grad_bias * grad_bias
+        )
+
+        correction1 = 1 - beta1 ** self.step_number
+        correction2 = 1 - beta2 ** self.step_number
+        first_hat = self.first_moment / correction1
+        second_hat = self.second_moment / correction2
+        factor_first_hat = self.factor_first_moment / correction1
+        factor_second_hat = self.factor_second_moment / correction2
+        bias_first_hat = self.bias_first_moment / correction1
+        bias_second_hat = self.bias_second_moment / correction2
+
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
-        self.bias -= self.learning_rate * gradient.sum()
+        self.factors -= (
+            self.learning_rate
+            * factor_first_hat
+            / (np.sqrt(factor_second_hat) + epsilon)
+        )
+        self.bias = np.float32(
+            self.bias
+            - self.learning_rate
+            * bias_first_hat
+            / (np.sqrt(bias_second_hat) + epsilon)
+        )
+
         return float(-np.mean(
             labels * np.log(probabilities + 1e-9)
             + (1 - labels) * np.log(1 - probabilities + 1e-9)
         ))
 
     def predict(self, features, batch_size=200_000):
+        if len(features) == 0:
+            return np.empty(0, dtype=np.float32)
         return np.concatenate([
             self.logits(features[index:index + batch_size])
             for index in range(0, len(features), batch_size)
         ])
 
     def state(self):
-        return self.weights.copy(), np.float32(self.bias)
+        return self.weights.copy(), self.factors.copy(), np.float32(self.bias)
 
     def load_state(self, state):
-        self.weights, self.bias = state
+        weights, factors, bias = state
+        self.weights = np.asarray(weights, dtype=np.float32).copy()
+        self.factors = np.asarray(factors, dtype=np.float32).copy()
+        self.bias = np.float32(bias)
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -1,4 +1,4 @@
-"""Train the neutral seed scaffold and emit validation scores for fixed evaluation."""
+"""Train the configured factorization machine and emit validation and test scores."""
 
 import argparse
 import json
@@ -48,19 +48,20 @@
     encoded, dimension = encode(splits)
     train_features, train_labels, _ = encoded["train"]
     valid_features, valid_labels, valid_users = encoded["valid"]
-    test_features, test_labels, _ = encoded["test"]
+    test_features, _, _ = encoded["test"]
     model = Model(
         dimension,
         learning_rate=config["learning_rate"],
         l2=config["l2"],
+        factor_rank=config["factor_rank"],
     )
     if args.contract_check:
         probe_size = min(8, len(train_labels))
         if probe_size == 0 or len(valid_labels) == 0:
             raise ValueError("contract probe requires non-empty train and validation slices")
         loss = model.step(train_features[:probe_size], train_labels[:probe_size])
-        probe_scores = model.predict(valid_features[:probe_size])
-        if probe_scores.ndim != 1 or len(probe_scores) != min(probe_size, len(valid_features)):
+        probe_scores = model.predict(valid_features)
+        if probe_scores.ndim != 1 or len(probe_scores) != len(valid_features):
             raise ValueError("model prediction shape violates the interface contract")
         if not np.isfinite(loss) or not np.all(np.isfinite(probe_scores)):
             raise ValueError("model produced NaN or infinity during contract probe")
@@ -86,14 +87,16 @@
     if best_state is None:
         raise RuntimeError("training produced no checkpoint")
     model.load_state(best_state)
+    output_dir = Path(args.output).parent
+    output_dir.mkdir(parents=True, exist_ok=True)
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

#### `config.json`

```diff
--- config.json
+++ config.json
@@ -2,6 +2,7 @@
   "seed": 0,
   "learning_rate": 0.01,
   "l2": 0.000001,
+  "factor_rank": 16,
   "batch_size": 8192,
   "max_epochs": 10,
   "patience": 3,
```

## Experiment 3

- Generation: 3
- Parent experiment: 1
- Status: scored
- Hypothesis: Train the additive five-field scorer with a within-user pairwise logistic ranking loss instead of pointwise BCE.
- Validation GAUC: 0.6705985549046443
- Validation nDCG@5: 0.5375321904955523
- Validation primary: 0.6040653727000983
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -16,8 +16,9 @@
 FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
 
 
-def load(data_dir, max_rows_per_split=None):
+def load(data_dir, max_rows_per_split=None, include_test_labels=False):
     rows = []
+    test_low, test_high = SPLITS["test"]
     for filename in (
         "log_standard_4_08_to_4_21_pure.csv",
         "log_standard_4_22_to_5_08_pure.csv",
@@ -35,6 +36,10 @@
                 if not 0 <= hour <= 23:
                     raise ValueError("hour derived from hourmin must be in 0..23")
                 weekday = datetime.strptime(date_value, "%Y%m%d").date().weekday()
+                if test_low <= date <= test_high and not include_test_labels:
+                    label = None
+                else:
+                    label = 1 if row[LABEL] != "0" else 0
 
                 rows.append((
                     date,
@@ -43,7 +48,7 @@
                     tab,
                     hour,
                     weekday,
-                    1 if row[LABEL] != "0" else 0,
+                    label,
                 ))
     result = {}
     for name, (low, high) in SPLITS.items():
@@ -71,14 +76,17 @@
     encoded = {}
     for name, rows in splits.items():
         features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
-        labels = np.empty(len(rows), dtype=np.float32)
+        labels = None if name == "test" and all(row[6] is None for row in rows) else np.empty(
+            len(rows), dtype=np.float32
+        )
         users = np.empty(len(rows), dtype="U32")
         for row_index, row in enumerate(rows):
             for field_index, value in enumerate(raw(row)):
                 features[row_index, field_index] = (
                     vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
                 )
-            labels[row_index] = row[6]
+            if labels is not None:
+                labels[row_index] = row[6]
             users[row_index] = row[1]
         encoded[name] = (features, labels, users)
     return encoded, int(sum(dimensions))
```

#### `model.py`

```diff
--- model.py
+++ model.py
@@ -20,14 +20,33 @@
     def logits(self, features):
         return self.bias + self.weights[features].sum(1)
 
-    def step(self, features, labels):
-        size = len(labels)
-        logits = self.logits(features)
-        probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+    def step(self, positive_features, negative_features):
+        positive_features = np.asarray(positive_features)
+        negative_features = np.asarray(negative_features)
+        if positive_features.ndim != 2 or negative_features.ndim != 2:
+            raise ValueError("positive_features and negative_features must be two-dimensional")
+        if positive_features.shape != negative_features.shape:
+            raise ValueError("positive_features and negative_features must have the same shape")
+        if positive_features.shape[1] != 5:
+            raise ValueError("pair feature matrices must have five columns")
+        pair_count = positive_features.shape[0]
+        if pair_count == 0:
+            raise ValueError("pair batches must not be empty")
+
+        positive_logits = self.logits(positive_features)
+        negative_logits = self.logits(negative_features)
+        delta = positive_logits - negative_logits
+        pair_losses = np.logaddexp(np.float32(0.0), -delta)
+        loss = float(np.mean(pair_losses, dtype=np.float64))
+        gradient = (-sigmoid(-delta) / np.float32(pair_count)).astype(np.float32)
+
         grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
+        np.add.at(grad_weights, positive_features, gradient[:, None])
+        np.add.at(grad_weights, negative_features, -gradient[:, None])
         grad_weights += self.l2 * self.weights
+        if not np.isfinite(loss) or not np.all(np.isfinite(grad_weights)):
+            raise ValueError("pairwise loss and gradients must be finite")
+
         self.step_number += 1
         beta1, beta2, epsilon = 0.9, 0.999, 1e-8
         self.first_moment *= beta1
@@ -37,11 +56,7 @@
         first_hat = self.first_moment / (1 - beta1 ** self.step_number)
         second_hat = self.second_moment / (1 - beta2 ** self.step_number)
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
-        self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
-            labels * np.log(probabilities + 1e-9)
-            + (1 - labels) * np.log(1 - probabilities + 1e-9)
-        ))
+        return loss
 
     def predict(self, features, batch_size=200_000):
         return np.concatenate([
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -1,4 +1,4 @@
-"""Train the neutral seed scaffold and emit validation scores for fixed evaluation."""
+"""Train the additive five-field scorer and emit canonical prediction archives."""
 
 import argparse
 import json
@@ -11,7 +11,7 @@
 
 
 def within_user_auc(user_ids, labels, scores):
-    """Training-only early-stop proxy; final scoring is owned by the Experimentor."""
+    """Validation-only early-stop proxy."""
     grouped = {}
     for user, label, score in zip(user_ids, labels, scores):
         grouped.setdefault(str(user), []).append((float(score), int(label)))
@@ -35,6 +35,30 @@
     return numerator / denominator if denominator else 0.5
 
 
+def build_train_pairs(train_features, train_labels, train_users, rng):
+    """Sample one same-user negative for each eligible positive training row."""
+    del train_features
+    grouped = {}
+    for index, user in enumerate(train_users):
+        grouped.setdefault(str(user), []).append(index)
+
+    positive_indices = []
+    negative_indices = []
+    for indices in grouped.values():
+        positives = [index for index in indices if train_labels[index] == 1]
+        negatives = [index for index in indices if train_labels[index] == 0]
+        if not positives or not negatives:
+            continue
+        for positive_index in positives:
+            positive_indices.append(positive_index)
+            negative_indices.append(rng.choice(negatives))
+
+    return (
+        np.asarray(positive_indices, dtype=np.int64),
+        np.asarray(negative_indices, dtype=np.int64),
+    )
+
+
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", required=True)
@@ -42,38 +66,80 @@
     parser.add_argument("--output", required=True)
     parser.add_argument("--contract-check", action="store_true")
     args = parser.parse_args()
+
     with open(args.config, encoding="utf-8") as handle:
         config = json.load(handle)
-    splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
+
+    splits = load(
+        args.data_dir,
+        max_rows_per_split=64 if args.contract_check else None,
+        include_test_labels=False,
+    )
     encoded, dimension = encode(splits)
-    train_features, train_labels, _ = encoded["train"]
+    train_features, train_labels, train_users = encoded["train"]
     valid_features, valid_labels, valid_users = encoded["valid"]
-    test_features, test_labels, _ = encoded["test"]
+    test_features, test_labels, test_users = encoded["test"]
+    assert test_labels is None
+
     model = Model(
         dimension,
         learning_rate=config["learning_rate"],
         l2=config["l2"],
     )
+
     if args.contract_check:
+        if len(train_labels) == 0 or len(valid_labels) == 0:
+            raise ValueError("contract probe requires non-empty train and validation slices")
         probe_size = min(8, len(train_labels))
-        if probe_size == 0 or len(valid_labels) == 0:
-            raise ValueError("contract probe requires non-empty train and validation slices")
-        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
-        probe_scores = model.predict(valid_features[:probe_size])
-        if probe_scores.ndim != 1 or len(probe_scores) != min(probe_size, len(valid_features)):
+        probe_rng = np.random.default_rng(config["seed"])
+        positive_indices, negative_indices = build_train_pairs(
+            train_features[:probe_size],
+            train_labels[:probe_size],
+            train_users[:probe_size],
+            probe_rng,
+        )
+        if len(positive_indices) == 0:
+            positive_indices, negative_indices = build_train_pairs(
+                train_features,
+                train_labels,
+                train_users,
+                probe_rng,
+            )
+        if len(positive_indices) == 0:
+            raise ValueError(
+                "contract probe requires a same-user positive-negative pair in the capped training split"
+            )
+        probe_pairs = min(8, len(positive_indices))
+        loss = model.step(
+            train_features[positive_indices[:probe_pairs]],
+            train_features[negative_indices[:probe_pairs]],
+        )
+        probe_scores = model.predict(valid_features)
+        if probe_scores.ndim != 1 or len(probe_scores) != len(valid_features):
             raise ValueError("model prediction shape violates the interface contract")
         if not np.isfinite(loss) or not np.all(np.isfinite(probe_scores)):
             raise ValueError("model produced NaN or infinity during contract probe")
         print(json.dumps({"contract": "ok", "feature_shape": list(train_features.shape)}))
         return
-    rng = np.random.default_rng(config["seed"])
+
     best_score, best_state, stale = -1.0, None, 0
     for epoch in range(1, config["max_epochs"] + 1):
-        order = rng.permutation(len(train_labels))
+        epoch_rng = np.random.default_rng(config["seed"] + epoch)
+        positive_indices, negative_indices = build_train_pairs(
+            train_features, train_labels, train_users, epoch_rng
+        )
+        if len(positive_indices) == 0:
+            raise ValueError("training split contains no same-user positive-negative pairs")
+        order = epoch_rng.permutation(len(positive_indices))
         losses = []
         for index in range(0, len(order), config["batch_size"]):
-            batch = order[index:index + config["batch_size"]]
-            losses.append(model.step(train_features[batch], train_labels[batch]))
+            batch_positions = order[index:index + config["batch_size"]]
+            losses.append(
+                model.step(
+                    train_features[positive_indices[batch_positions]],
+                    train_features[negative_indices[batch_positions]],
+                )
+            )
         predictions = model.predict(valid_features)
         proxy = within_user_auc(valid_users, valid_labels, predictions)
         print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_gauc_proxy={proxy:.6f}")
@@ -83,17 +149,20 @@
             stale += 1
             if stale >= config["patience"]:
                 break
+
     if best_state is None:
         raise RuntimeError("training produced no checkpoint")
     model.load_state(best_state)
+
+    output_path = Path(args.output)
     np.savez(
-        args.output,
-        row_ids=np.arange(len(valid_labels), dtype=np.int64),
+        output_path.with_name("predictions_valid.npz"),
+        row_ids=np.arange(len(valid_features), dtype=np.int64),
         scores=model.predict(valid_features),
     )
     np.savez(
-        Path(args.output).with_name("predictions_test.npz"),
-        row_ids=np.arange(len(test_labels), dtype=np.int64),
+        output_path.with_name("predictions_test.npz"),
+        row_ids=np.arange(len(test_features), dtype=np.int64),
         scores=model.predict(test_features),
     )
```

## Experiment 4

- Generation: 4
- Parent experiment: 3
- Status: scored
- Hypothesis: Increase same-user negative sampling from one to four negatives per eligible positive in the pairwise logistic objective.
- Validation GAUC: 0.6707057443888901
- Validation nDCG@5: 0.5382868021095627
- Validation primary: 0.6044962732492264
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `train.py`

```diff
--- train.py
+++ train.py
@@ -36,8 +36,9 @@
 
 
 def build_train_pairs(train_features, train_labels, train_users, rng):
-    """Sample one same-user negative for each eligible positive training row."""
+    """Sample K=4 same-user negatives uniformly without replacement when possible, using all available negatives when fewer than four exist."""
     del train_features
+    k = 4
     grouped = {}
     for index, user in enumerate(train_users):
         grouped.setdefault(str(user), []).append(index)
@@ -49,9 +50,14 @@
         negatives = [index for index in indices if train_labels[index] == 0]
         if not positives or not negatives:
             continue
+        sample_size = min(k, len(negatives))
         for positive_index in positives:
-            positive_indices.append(positive_index)
-            negative_indices.append(rng.choice(negatives))
+            sampled_negatives = rng.choice(
+                negatives, size=sample_size, replace=False
+            )
+            for negative_index in sampled_negatives:
+                positive_indices.append(positive_index)
+                negative_indices.append(negative_index)
 
     return (
         np.asarray(positive_indices, dtype=np.int64),
```

## Experiment 5

- Generation: 5
- Parent experiment: 4
- Status: scored
- Hypothesis: Add an explicit user-by-tab categorical interaction to the current additive five-field pairwise ranker.
- Validation GAUC: 0.6685006427875048
- Validation nDCG@5: 0.5363084065155106
- Validation primary: 0.6024045246515077
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
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday", "user_id_tab"]
 
 
 def load(data_dir, max_rows_per_split=None, include_test_labels=False):
@@ -63,7 +63,7 @@
         raise ValueError("training split is empty")
 
     def raw(row):
-        return [row[1], row[2], row[3], row[4], row[5]]
+        return [row[1], row[2], row[3], row[4], row[5], (row[1], row[3])]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
```

#### `model.py`

```diff
--- model.py
+++ model.py
@@ -27,8 +27,8 @@
             raise ValueError("positive_features and negative_features must be two-dimensional")
         if positive_features.shape != negative_features.shape:
             raise ValueError("positive_features and negative_features must have the same shape")
-        if positive_features.shape[1] != 5:
-            raise ValueError("pair feature matrices must have five columns")
+        if positive_features.shape[1] != 6:
+            raise ValueError("pair feature matrices must have six columns")
         pair_count = positive_features.shape[0]
         if pair_count == 0:
             raise ValueError("pair batches must not be empty")
```

## Experiment 6

- Generation: 6
- Parent experiment: 4
- Status: scored
- Hypothesis: Select the early-stopping checkpoint by validation primary instead of validation GAUC.
- Validation GAUC: 0.6707057443888901
- Validation nDCG@5: 0.5382868021095627
- Validation primary: 0.6044962732492264
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `train.py`

```diff
--- train.py
+++ train.py
@@ -35,6 +35,34 @@
     return numerator / denominator if denominator else 0.5
 
 
+def within_user_ndcg_at_5(user_ids, labels, scores):
+    """Compute unweighted validation NDCG@5 with canonical-order tie breaking."""
+    grouped = {}
+    for position, (user, label, score) in enumerate(zip(user_ids, labels, scores)):
+        grouped.setdefault(str(user), []).append((float(score), int(label), position))
+
+    ndcgs = []
+    for rows in grouped.values():
+        labels_for_user = [label for _, label, _ in rows]
+        if not any(labels_for_user):
+            continue
+
+        ranked_rows = sorted(rows, key=lambda row: (-row[0], row[2]))
+        dcg = sum(
+            (2 ** label - 1) / np.log2(rank + 1)
+            for rank, (_, label, _) in enumerate(ranked_rows[:5], start=1)
+        )
+        ideal_labels = sorted(labels_for_user, reverse=True)
+        ideal_dcg = sum(
+            (2 ** label - 1) / np.log2(rank + 1)
+            for rank, label in enumerate(ideal_labels[:5], start=1)
+        )
+        if ideal_dcg > 0:
+            ndcgs.append(dcg / ideal_dcg)
+
+    return float(np.mean(ndcgs)) if ndcgs else 0.0
+
+
 def build_train_pairs(train_features, train_labels, train_users, rng):
     """Sample K=4 same-user negatives uniformly without replacement when possible, using all available negatives when fewer than four exist."""
     del train_features
@@ -147,10 +175,20 @@
                 )
             )
         predictions = model.predict(valid_features)
-        proxy = within_user_auc(valid_users, valid_labels, predictions)
-        print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_gauc_proxy={proxy:.6f}")
-        if proxy > best_score + 1e-5:
-            best_score, best_state, stale = proxy, model.state(), 0
+        valid_gauc = within_user_auc(valid_users, valid_labels, predictions)
+        valid_ndcg_at_5 = within_user_ndcg_at_5(
+            valid_users, valid_labels, predictions
+        )
+        valid_primary = 0.5 * (valid_gauc + valid_ndcg_at_5)
+        print(
+            f"epoch={epoch} loss={np.mean(losses):.6f} "
+            f"valid_gauc={valid_gauc:.6f} "
+            f"valid_ndcg_at_5={valid_ndcg_at_5:.6f} "
+            f"valid_primary={valid_primary:.6f} "
+            f"checkpoint_selection_metric=valid_primary"
+        )
+        if valid_primary > best_score + 1e-5:
+            best_score, best_state, stale = valid_primary, model.state(), 0
         else:
             stale += 1
             if stale >= config["patience"]:
```

## Experiment 7

- Generation: 7
- Parent experiment: 6
- Status: scored
- Hypothesis: Use a 50/50 uniform and hard-negative sampler within each user's training impressions.
- Validation GAUC: 0.6391794510641228
- Validation nDCG@5: 0.5258758731018784
- Validation primary: 0.5825276620830007
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `train.py`

```diff
--- train.py
+++ train.py
@@ -63,8 +63,10 @@
     return float(np.mean(ndcgs)) if ndcgs else 0.0
 
 
-def build_train_pairs(train_features, train_labels, train_users, rng):
-    """Sample K=4 same-user negatives uniformly without replacement when possible, using all available negatives when fewer than four exist."""
+def build_train_pairs(
+    train_features, train_labels, train_users, frozen_train_scores, rng
+):
+    """Sample frozen hard and uniform same-user negatives without replacement."""
     del train_features
     k = 4
     grouped = {}
@@ -78,12 +80,28 @@
         negatives = [index for index in indices if train_labels[index] == 0]
         if not positives or not negatives:
             continue
-        sample_size = min(k, len(negatives))
+
+        ranked_negatives = sorted(
+            negatives, key=lambda index: (-frozen_train_scores[index], index)
+        )
+        hard_count = min(2, len(negatives))
+        hard_negatives = ranked_negatives[:hard_count]
+        remaining_negatives = [
+            index for index in negatives if index not in set(hard_negatives)
+        ]
+        uniform_count = min(k - hard_count, len(remaining_negatives))
+
         for positive_index in positives:
-            sampled_negatives = rng.choice(
-                negatives, size=sample_size, replace=False
-            )
-            for negative_index in sampled_negatives:
+            if uniform_count:
+                uniform_negatives = rng.choice(
+                    remaining_negatives, size=uniform_count, replace=False
+                )
+            else:
+                uniform_negatives = []
+            for negative_index in hard_negatives:
+                positive_indices.append(positive_index)
+                negative_indices.append(negative_index)
+            for negative_index in uniform_negatives:
                 positive_indices.append(positive_index)
                 negative_indices.append(negative_index)
 
@@ -126,17 +144,21 @@
             raise ValueError("contract probe requires non-empty train and validation slices")
         probe_size = min(8, len(train_labels))
         probe_rng = np.random.default_rng(config["seed"])
+        probe_frozen_scores = model.predict(train_features[:probe_size])
         positive_indices, negative_indices = build_train_pairs(
             train_features[:probe_size],
             train_labels[:probe_size],
             train_users[:probe_size],
+            probe_frozen_scores,
             probe_rng,
         )
         if len(positive_indices) == 0:
+            probe_frozen_scores = model.predict(train_features)
             positive_indices, negative_indices = build_train_pairs(
                 train_features,
                 train_labels,
                 train_users,
+                probe_frozen_scores,
                 probe_rng,
             )
         if len(positive_indices) == 0:
@@ -159,8 +181,13 @@
     best_score, best_state, stale = -1.0, None, 0
     for epoch in range(1, config["max_epochs"] + 1):
         epoch_rng = np.random.default_rng(config["seed"] + epoch)
+        frozen_train_scores = model.predict(train_features)
         positive_indices, negative_indices = build_train_pairs(
-            train_features, train_labels, train_users, epoch_rng
+            train_features,
+            train_labels,
+            train_users,
+            frozen_train_scores,
+            epoch_rng,
         )
         if len(positive_indices) == 0:
             raise ValueError("training split contains no same-user positive-negative pairs")
```

## Experiment 8

- Generation: 8
- Parent experiment: 4
- Status: scored
- Hypothesis: Replace the raw 24-category hour feature with a train-fitted six-bin four-hour categorical feature.
- Validation GAUC: 0.670861646390768
- Validation nDCG@5: 0.5375787910842658
- Validation primary: 0.6042202187375169
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
+FIELDS = ["user_id", "video_id", "tab", "hour_bin", "weekday"]
 
 
 def load(data_dir, max_rows_per_split=None, include_test_labels=False):
@@ -35,6 +35,7 @@
                 hour = int(float(hourmin) // 100)
                 if not 0 <= hour <= 23:
                     raise ValueError("hour derived from hourmin must be in 0..23")
+                hour_bin = hour // 4
                 weekday = datetime.strptime(date_value, "%Y%m%d").date().weekday()
                 if test_low <= date <= test_high and not include_test_labels:
                     label = None
@@ -46,7 +47,7 @@
                     user_id,
                     video_id,
                     tab,
-                    hour,
+                    hour_bin,
                     weekday,
                     label,
                 ))
```
