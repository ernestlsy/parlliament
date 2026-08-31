# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add the screened request-context feature family to the additive user/video-ID scorer.
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
+"""Data loading and train-fitted categorical encoding for additive ID/context scoring."""
 
 import csv
+import math
 import os
+from datetime import datetime
 
 import numpy as np
 
@@ -12,21 +14,43 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
 
 
 def load(data_dir, max_rows_per_split=None):
     rows = []
+    test_low, test_high = SPLITS["test"]
     for filename in (
         "log_standard_4_08_to_4_21_pure.csv",
         "log_standard_4_22_to_5_08_pure.csv",
     ):
         with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
             for row in csv.DictReader(handle):
-                rows.append((
-                    int(row["date"]), row["user_id"], row["video_id"],
-                    1 if row[LABEL] != "0" else 0,
-                ))
+                raw_date = row["date"]
+                date_value = int(raw_date)
+                calendar_date = datetime.strptime(raw_date, "%Y%m%d").date()
+
+                hourmin = float(row["hourmin"])
+                if not math.isfinite(hourmin):
+                    raise ValueError("hourmin must be finite")
+                hour = math.floor(hourmin / 100)
+                if not 0 <= hour <= 23:
+                    raise ValueError("derived hour must be in 0..23")
+                weekday = calendar_date.weekday()
+
+                values = (
+                    date_value,
+                    row["user_id"],
+                    row["video_id"],
+                    row["tab"],
+                    hour,
+                    weekday,
+                )
+                if test_low <= date_value <= test_high:
+                    rows.append(values)
+                else:
+                    rows.append(values + (1 if row[LABEL] != "0" else 0,))
+
     result = {}
     for name, (low, high) in SPLITS.items():
         selected = [row for row in rows if low <= row[0] <= high]
@@ -38,28 +62,38 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
+
     def raw(row):
-        return [row[1], row[2]]
+        return [row[1], row[2], row[3], row[4], row[5]]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
         for index, value in enumerate(raw(row)):
             if value not in vocabs[index]:
                 vocabs[index][value] = len(vocabs[index])
+
     unknown = [len(vocab) for vocab in vocabs]
     dimensions = [len(vocab) + 1 for vocab in vocabs]
     offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)
+
     encoded = {}
     for name, rows in splits.items():
         features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
-        labels = np.empty(len(rows), dtype=np.float32)
         users = np.empty(len(rows), dtype="U32")
         for row_index, row in enumerate(rows):
             for field_index, value in enumerate(raw(row)):
                 features[row_index, field_index] = (
-                    vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
+                    vocabs[field_index].get(value, unknown[field_index])
+                    + offsets[field_index]
                 )
-            labels[row_index] = row[3]
             users[row_index] = row[1]
-        encoded[name] = (features, labels, users)
+
+        if name == "test":
+            encoded[name] = (features, users)
+        else:
+            labels = np.empty(len(rows), dtype=np.float32)
+            for row_index, row in enumerate(rows):
+                labels[row_index] = row[6]
+            encoded[name] = (features, labels, users)
+
     return encoded, int(sum(dimensions))
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -48,7 +48,7 @@
     encoded, dimension = encode(splits)
     train_features, train_labels, _ = encoded["train"]
     valid_features, valid_labels, valid_users = encoded["valid"]
-    test_features, test_labels, _ = encoded["test"]
+    test_features, _ = encoded["test"]
     model = Model(
         dimension,
         learning_rate=config["learning_rate"],
@@ -86,14 +86,15 @@
     if best_state is None:
         raise RuntimeError("training produced no checkpoint")
     model.load_state(best_state)
+    output_dir = Path(args.output)
     np.savez(
-        args.output,
-        row_ids=np.arange(len(valid_labels), dtype=np.int64),
+        output_dir.with_name("predictions_valid.npz"),
+        row_ids=np.arange(len(valid_features), dtype=np.int64),
         scores=model.predict(valid_features),
     )
     np.savez(
-        Path(args.output).with_name("predictions_test.npz"),
-        row_ids=np.arange(len(test_labels), dtype=np.int64),
+        output_dir.with_name("predictions_test.npz"),
+        row_ids=np.arange(len(test_features), dtype=np.int64),
         scores=model.predict(test_features),
     )
```

## Experiment 2

- Generation: 2
- Parent experiment: 1
- Status: scored
- Hypothesis: Replace additive ID weights with a low-rank user-video dot-product scorer while retaining additive tab, hour, and weekday residuals.
- Validation GAUC: 0.6410553195276255
- Validation nDCG@5: 0.5240130284265571
- Validation primary: 0.5825341739770913
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
+"""Low-rank user-video interaction model with categorical residuals."""
 
 import numpy as np
 
@@ -9,48 +9,138 @@
 
 class Model:
     def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
-        self.weights = np.zeros(dimension, dtype=np.float32)
+        rng = np.random.default_rng(0)
+        self.user_embeddings = rng.normal(
+            0.0, 0.01, size=(dimension, 16)
+        ).astype(np.float32)
+        self.video_embeddings = rng.normal(
+            0.0, 0.01, size=(dimension, 16)
+        ).astype(np.float32)
+        self.residual_weights = np.zeros(dimension, dtype=np.float32)
         self.bias = np.float32(0.0)
         self.learning_rate = learning_rate
         self.l2 = l2
-        self.first_moment = np.zeros_like(self.weights)
-        self.second_moment = np.zeros_like(self.weights)
+
+        self.user_first_moment = np.zeros_like(self.user_embeddings)
+        self.user_second_moment = np.zeros_like(self.user_embeddings)
+        self.video_first_moment = np.zeros_like(self.video_embeddings)
+        self.video_second_moment = np.zeros_like(self.video_embeddings)
+        self.residual_first_moment = np.zeros_like(self.residual_weights)
+        self.residual_second_moment = np.zeros_like(self.residual_weights)
         self.step_number = 0
 
+    @staticmethod
+    def _normalized_interaction(user_vectors, video_vectors):
+        epsilon = np.float32(1e-8)
+        user_norms = np.sqrt((user_vectors * user_vectors).sum(axis=1) + epsilon)
+        video_norms = np.sqrt((video_vectors * video_vectors).sum(axis=1) + epsilon)
+        user_hat = user_vectors / user_norms[:, None]
+        video_hat = video_vectors / video_norms[:, None]
+        dots = (user_hat * video_hat).sum(axis=1)
+        return dots, user_hat, video_hat, user_norms, video_norms
+
     def logits(self, features):
-        return self.bias + self.weights[features].sum(1)
+        user_vectors = self.user_embeddings[features[:, 0]]
+        video_vectors = self.video_embeddings[features[:, 1]]
+        interaction, _, _, _, _ = self._normalized_interaction(
+            user_vectors, video_vectors
+        )
+        residual = self.residual_weights[features[:, 2:5]].sum(axis=1)
+        return self.bias + interaction + residual
+
+    def _adam_update(self, parameter, gradient, first_moment, second_moment):
+        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
+        first_moment *= beta1
+        first_moment += (1 - beta1) * gradient
+        second_moment *= beta2
+        second_moment += (1 - beta2) * (gradient * gradient)
+        first_hat = first_moment / (1 - beta1 ** self.step_number)
+        second_hat = second_moment / (1 - beta2 ** self.step_number)
+        parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
 
     def step(self, features, labels):
         size = len(labels)
-        logits = self.logits(features)
+        user_ids = features[:, 0]
+        video_ids = features[:, 1]
+        residual_ids = features[:, 2:5]
+
+        user_vectors = self.user_embeddings[user_ids]
+        video_vectors = self.video_embeddings[video_ids]
+        interaction, user_hat, video_hat, user_norms, video_norms = (
+            self._normalized_interaction(user_vectors, video_vectors)
+        )
+        residual = self.residual_weights[residual_ids].sum(axis=1)
+        logits = self.bias + interaction + residual
         probabilities = sigmoid(logits)
         gradient = ((probabilities - labels) / size).astype(np.float32)
-        grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
-        grad_weights += self.l2 * self.weights
+
+        grad_user_embeddings = np.zeros_like(self.user_embeddings)
+        grad_video_embeddings = np.zeros_like(self.video_embeddings)
+        grad_residual_weights = np.zeros_like(self.residual_weights)
+
+        user_gradient = gradient[:, None] * (
+            video_hat - interaction[:, None] * user_hat
+        ) / user_norms[:, None]
+        video_gradient = gradient[:, None] * (
+            user_hat - interaction[:, None] * video_hat
+        ) / video_norms[:, None]
+        np.add.at(grad_user_embeddings, user_ids, user_gradient)
+        np.add.at(grad_video_embeddings, video_ids, video_gradient)
+        np.add.at(
+            grad_residual_weights,
+            residual_ids.ravel(),
+            np.repeat(gradient, residual_ids.shape[1]),
+        )
+
+        grad_user_embeddings += self.l2 * self.user_embeddings
+        grad_video_embeddings += self.l2 * self.video_embeddings
+        grad_residual_weights += self.l2 * self.residual_weights
+
         self.step_number += 1
-        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
-        self.first_moment *= beta1
-        self.first_moment += (1 - beta1) * grad_weights
-        self.second_moment *= beta2
-        self.second_moment += (1 - beta2) * (grad_weights * grad_weights)
-        first_hat = self.first_moment / (1 - beta1 ** self.step_number)
-        second_hat = self.second_moment / (1 - beta2 ** self.step_number)
-        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
+        self._adam_update(
+            self.user_embeddings,
+            grad_user_embeddings,
+            self.user_first_moment,
+            self.user_second_moment,
+        )
+        self._adam_update(
+            self.video_embeddings,
+            grad_video_embeddings,
+            self.video_first_moment,
+            self.video_second_moment,
+        )
+        self._adam_update(
+            self.residual_weights,
+            grad_residual_weights,
+            self.residual_first_moment,
+            self.residual_second_moment,
+        )
         self.bias -= self.learning_rate * gradient.sum()
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
+        return (
+            self.user_embeddings.copy(),
+            self.video_embeddings.copy(),
+            self.residual_weights.copy(),
+            np.float32(self.bias),
+        )
 
     def load_state(self, state):
-        self.weights, self.bias = state
+        user_embeddings, video_embeddings, residual_weights, bias = state
+        self.user_embeddings = np.asarray(user_embeddings, dtype=np.float32).copy()
+        self.video_embeddings = np.asarray(video_embeddings, dtype=np.float32).copy()
+        self.residual_weights = np.asarray(residual_weights, dtype=np.float32).copy()
+        self.bias = np.float32(bias)
```

## Experiment 3

- Generation: 3
- Parent experiment: 1
- Status: scored
- Hypothesis: Add a train-fitted user-by-tab categorical cross to the additive experiment-1 scorer.
- Validation GAUC: 0.6699349275524258
- Validation nDCG@5: 0.5368127566631969
- Validation primary: 0.6033738421078114
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -14,7 +14,7 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday", "user_id_x_tab"]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -64,7 +64,7 @@
         raise ValueError("training split is empty")
 
     def raw(row):
-        return [row[1], row[2], row[3], row[4], row[5]]
+        return [row[1], row[2], row[3], row[4], row[5], (row[1], row[3])]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
```

## Experiment 4

- Generation: 4
- Parent experiment: 3
- Status: scored
- Hypothesis: Replace row-mean BCE with inverse-training-user-candidate-count weighted BCE to align training influence with within-user evaluation.
- Validation GAUC: 0.6644646681730271
- Validation nDCG@5: 0.5346652718799744
- Validation primary: 0.5995649700265007
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -20,11 +20,24 @@
     def logits(self, features):
         return self.bias + self.weights[features].sum(1)
 
-    def step(self, features, labels):
-        size = len(labels)
+    def step(self, features, labels, sample_weights):
+        sample_weights = np.asarray(sample_weights)
+        if sample_weights.ndim != 1 or len(sample_weights) != len(labels):
+            raise ValueError("sample_weights must be one-dimensional and aligned with labels")
+        try:
+            if not np.all(np.isfinite(sample_weights)) or not np.all(sample_weights > 0):
+                raise ValueError("sample_weights must contain finite strictly positive values")
+            sample_weight_sum = sample_weights.sum()
+            if not np.isfinite(sample_weight_sum) or sample_weight_sum <= 0:
+                raise ValueError("sample_weights must have a finite positive sum")
+        except TypeError as error:
+            raise ValueError("sample_weights must contain finite strictly positive values") from error
+
         logits = self.logits(features)
         probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+        gradient = (
+            (probabilities - labels) * sample_weights / sample_weights.sum()
+        ).astype(np.float32)
         grad_weights = np.zeros_like(self.weights)
         np.add.at(grad_weights, features, gradient[:, None])
         grad_weights += self.l2 * self.weights
@@ -38,10 +51,12 @@
         second_hat = self.second_moment / (1 - beta2 ** self.step_number)
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
         self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
-            labels * np.log(probabilities + 1e-9)
-            + (1 - labels) * np.log(1 - probabilities + 1e-9)
-        ))
+        return float(-np.sum(
+            sample_weights * (
+                labels * np.log(probabilities + 1e-9)
+                + (1 - labels) * np.log(1 - probabilities + 1e-9)
+            )
+        ) / sample_weights.sum())
 
     def predict(self, features, batch_size=200_000):
         return np.concatenate([
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -46,9 +46,24 @@
         config = json.load(handle)
     splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
     encoded, dimension = encode(splits)
-    train_features, train_labels, _ = encoded["train"]
+    train_features, train_labels, train_users = encoded["train"]
     valid_features, valid_labels, valid_users = encoded["valid"]
     test_features, _ = encoded["test"]
+    user_counts = {}
+    train_user_keys = [str(user) for user in train_users]
+    for user in train_user_keys:
+        user_counts[user] = user_counts.get(user, 0) + 1
+    train_weights = np.asarray(
+        [1.0 / user_counts[user] for user in train_user_keys], dtype=np.float32
+    )
+    if (
+        train_weights.ndim != 1
+        or len(train_weights) != len(train_features)
+        or len(train_weights) != len(train_labels)
+        or not np.all(np.isfinite(train_weights))
+        or not np.all(train_weights > 0)
+    ):
+        raise ValueError("training weights must be finite, positive, and aligned")
     model = Model(
         dimension,
         learning_rate=config["learning_rate"],
@@ -58,7 +73,11 @@
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
@@ -73,7 +92,11 @@
         losses = []
         for index in range(0, len(order), config["batch_size"]):
             batch = order[index:index + config["batch_size"]]
-            losses.append(model.step(train_features[batch], train_labels[batch]))
+            losses.append(
+                model.step(
+                    train_features[batch], train_labels[batch], train_weights[batch]
+                )
+            )
         predictions = model.predict(valid_features)
         proxy = within_user_auc(valid_users, valid_labels, predictions)
         print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_gauc_proxy={proxy:.6f}")
```
