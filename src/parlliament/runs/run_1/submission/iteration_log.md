# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add the complete impression-time request-context feature group to the neutral additive user/video scaffold: tab, hour, and weekday encoded with train-only vocabularies and explicit unknown handling.
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
@@ -1,7 +1,8 @@
-"""Neutral seed data contract using only the task's user and item identifiers."""
+"""Neutral additive data contract with impression-time request context features."""
 
 import csv
 import os
+from datetime import datetime
 
 import numpy as np
 
@@ -12,25 +13,43 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
 
 
 def load(data_dir, max_rows_per_split=None):
-    rows = []
+    result = {name: [] for name in SPLITS}
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
-    result = {}
-    for name, (low, high) in SPLITS.items():
-        selected = [row for row in rows if low <= row[0] <= high]
-        result[name] = selected if max_rows_per_split is None else selected[:max_rows_per_split]
+                date = int(row["date"])
+                split_name = next(
+                    (
+                        name
+                        for name, (low, high) in SPLITS.items()
+                        if low <= date <= high
+                    ),
+                    None,
+                )
+                if split_name is None:
+                    continue
+                base = (
+                    date,
+                    row["user_id"],
+                    row["video_id"],
+                    row["hourmin"],
+                    row["tab"],
+                )
+                if split_name == "test":
+                    result[split_name].append(base)
+                else:
+                    result[split_name].append(base + (1 if row[LABEL] != "0" else 0,))
+
+    if max_rows_per_split is not None:
+        for name in result:
+            result[name] = result[name][:max_rows_per_split]
     return result
 
 
@@ -38,28 +57,42 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
+
     def raw(row):
-        return [row[1], row[2]]
+        hour = int(float(row[3])) // 100
+        hour = max(0, min(23, hour))
+        weekday = datetime.strptime(str(int(row[0])), "%Y%m%d").weekday()
+        return [row[1], row[2], row[4], hour, weekday]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
         for index, value in enumerate(raw(row)):
             if value not in vocabs[index]:
                 vocabs[index][value] = len(vocabs[index])
+
     unknown = [len(vocab) for vocab in vocabs]
     dimensions = [len(vocab) + 1 for vocab in vocabs]
     offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)
     encoded = {}
+
     for name, rows in splits.items():
         features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
-        labels = np.empty(len(rows), dtype=np.float32)
         users = np.empty(len(rows), dtype="U32")
         for row_index, row in enumerate(rows):
-            for field_index, value in enumerate(raw(row)):
+            values = raw(row)
+            for field_index, value in enumerate(values):
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
+            labels = np.asarray(
+                [row[5] for row in rows], dtype=np.float32
+            )
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
@@ -88,12 +88,12 @@
     model.load_state(best_state)
     np.savez(
         args.output,
-        row_ids=np.arange(len(valid_labels), dtype=np.int64),
+        row_ids=np.arange(len(valid_features), dtype=np.int64),
         scores=model.predict(valid_features),
     )
     np.savez(
         Path(args.output).with_name("predictions_test.npz"),
-        row_ids=np.arange(len(test_labels), dtype=np.int64),
+        row_ids=np.arange(len(test_features), dtype=np.int64),
         scores=model.predict(test_features),
     )
```

### Error and recovery events

```json
[
  {
    "kind": "contract_fulfillment",
    "message": "invalid prediction artifact: row_ids must be consecutive and in canonical test order",
    "traceback": "ValueError('row_ids must be consecutive and in canonical test order')",
    "responsible_agents": [
      "trainer"
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
- Hypothesis: Add a one-hidden-layer nonlinear residual over the parent context fields while preserving its additive ID terms.
- Validation GAUC: 0.6691289473036826
- Validation nDCG@5: 0.5372061575740308
- Validation primary: 0.6031675524388567
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
+"""Additive ID model with a nonlinear request-context residual."""
 
 import numpy as np
 
@@ -17,40 +17,217 @@
         self.second_moment = np.zeros_like(self.weights)
         self.step_number = 0
 
+        self.context_embedding_dimension = 8
+        self.hidden_units = 32
+        self.context_ranges = None
+        self.context_embeddings = None
+        self.hidden_weights = None
+        self.hidden_bias = None
+        self.output_weights = None
+        self.output_bias = None
+        self.context_first_moment = None
+        self.context_second_moment = None
+        self.hidden_weights_first_moment = None
+        self.hidden_weights_second_moment = None
+        self.hidden_bias_first_moment = None
+        self.hidden_bias_second_moment = None
+        self.output_weights_first_moment = None
+        self.output_weights_second_moment = None
+        self.output_bias_first_moment = None
+        self.output_bias_second_moment = None
+
+    def _ensure_context_parameters(self, features):
+        features = np.asarray(features)
+        if self.context_embeddings is not None:
+            return
+        if features.ndim != 2 or features.shape[1] < 5:
+            raise ValueError("features must have shape (N, 5)")
+
+        ranges = []
+        for column in (2, 3, 4):
+            values = np.asarray(features[:, column], dtype=np.int64)
+            if values.size:
+                lower = int(values.min())
+                upper = int(values.max())
+            else:
+                lower, upper = 0, -1
+            ranges.append((lower, upper))
+        self.context_ranges = tuple(ranges)
+
+        rng = np.random
+        self.context_embeddings = []
+        for lower, upper in self.context_ranges:
+            size = max(1, upper - lower + 1) + 1
+            self.context_embeddings.append(
+                rng.normal(0.0, 0.02, (size, self.context_embedding_dimension))
+                .astype(np.float32)
+            )
+        self.hidden_weights = rng.normal(
+            0.0, np.sqrt(2.0 / (3 * self.context_embedding_dimension)),
+            (3 * self.context_embedding_dimension, self.hidden_units)
+        ).astype(np.float32)
+        self.hidden_bias = np.zeros(self.hidden_units, dtype=np.float32)
+        self.output_weights = rng.normal(
+            0.0, np.sqrt(2.0 / self.hidden_units), self.hidden_units
+        ).astype(np.float32)
+        self.output_bias = np.float32(0.0)
+
+        self.context_first_moment = [np.zeros_like(x) for x in self.context_embeddings]
+        self.context_second_moment = [np.zeros_like(x) for x in self.context_embeddings]
+        self.hidden_weights_first_moment = np.zeros_like(self.hidden_weights)
+        self.hidden_weights_second_moment = np.zeros_like(self.hidden_weights)
+        self.hidden_bias_first_moment = np.zeros_like(self.hidden_bias)
+        self.hidden_bias_second_moment = np.zeros_like(self.hidden_bias)
+        self.output_weights_first_moment = np.zeros_like(self.output_weights)
+        self.output_weights_second_moment = np.zeros_like(self.output_weights)
+        self.output_bias_first_moment = np.float32(0.0)
+        self.output_bias_second_moment = np.float32(0.0)
+
+    def _context_indices(self, features):
+        indices = []
+        for position, (lower, upper) in enumerate(self.context_ranges):
+            values = np.asarray(features[:, position + 2], dtype=np.int64)
+            valid = (values >= lower) & (values <= upper)
+            mapped = np.full(values.shape, len(self.context_embeddings[position]) - 1,
+                             dtype=np.int64)
+            mapped[valid] = values[valid] - lower
+            indices.append(mapped)
+        return indices
+
+    def _residual_forward(self, features):
+        self._ensure_context_parameters(features)
+        indices = self._context_indices(features)
+        embedded = [table[index] for table, index in zip(self.context_embeddings, indices)]
+        combined = np.concatenate(embedded, axis=1)
+        hidden_pre = combined.dot(self.hidden_weights) + self.hidden_bias
+        hidden = np.maximum(hidden_pre, 0.0)
+        residual = hidden.dot(self.output_weights) + self.output_bias
+        return residual.astype(np.float32), (indices, combined, hidden, hidden_pre)
+
     def logits(self, features):
-        return self.bias + self.weights[features].sum(1)
+        features = np.asarray(features)
+        if features.ndim != 2 or features.shape[1] < 5:
+            raise ValueError("features must have shape (N, 5)")
+        additive = self.bias + self.weights[features[:, :5]].sum(axis=1)
+        residual, _ = self._residual_forward(features)
+        return np.asarray(additive + residual, dtype=np.float32).reshape(-1)
+
+    def _adam_update(self, parameter, gradient, first, second, beta1, beta2, epsilon):
+        first *= beta1
+        first += (1.0 - beta1) * gradient
+        second *= beta2
+        second += (1.0 - beta2) * (gradient * gradient)
+        first_hat = first / (1.0 - beta1 ** self.step_number)
+        second_hat = second / (1.0 - beta2 ** self.step_number)
+        parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
 
     def step(self, features, labels):
+        features = np.asarray(features)
+        labels = np.asarray(labels, dtype=np.float32)
         size = len(labels)
-        logits = self.logits(features)
+        if size == 0:
+            return 0.0
+        additive = self.bias + self.weights[features[:, :5]].sum(axis=1)
+        residual, cache = self._residual_forward(features)
+        logits = np.asarray(additive + residual, dtype=np.float32)
         probabilities = sigmoid(logits)
         gradient = ((probabilities - labels) / size).astype(np.float32)
+
         grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
+        np.add.at(grad_weights, features[:, :5], gradient[:, None])
         grad_weights += self.l2 * self.weights
+        grad_bias = np.float32(gradient.sum())
+
+        indices, combined, hidden, hidden_pre = cache
+        grad_output_weights = hidden.T.dot(gradient) + self.l2 * self.output_weights
+        grad_output_bias = np.float32(gradient.sum())
+        hidden_gradient = gradient[:, None] * self.output_weights[None, :]
+        hidden_gradient *= (hidden_pre > 0.0)
+        grad_hidden_weights = combined.T.dot(hidden_gradient) + self.l2 * self.hidden_weights
+        grad_hidden_bias = hidden_gradient.sum(axis=0)
+        grad_combined = hidden_gradient.dot(self.hidden_weights.T)
+        grad_context = []
+        for position, index in enumerate(indices):
+            grad = np.zeros_like(self.context_embeddings[position])
+            np.add.at(grad, index, grad_combined[:, position * self.context_embedding_dimension:
+                                                   (position + 1) * self.context_embedding_dimension])
+            grad += self.l2 * self.context_embeddings[position]
+            grad_context.append(grad)
+
         self.step_number += 1
         beta1, beta2, epsilon = 0.9, 0.999, 1e-8
-        self.first_moment *= beta1
-        self.first_moment += (1 - beta1) * grad_weights
-        self.second_moment *= beta2
-        self.second_moment += (1 - beta2) * (grad_weights * grad_weights)
-        first_hat = self.first_moment / (1 - beta1 ** self.step_number)
-        second_hat = self.second_moment / (1 - beta2 ** self.step_number)
-        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
-        self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
-            labels * np.log(probabilities + 1e-9)
-            + (1 - labels) * np.log(1 - probabilities + 1e-9)
-        ))
+        self._adam_update(self.weights, grad_weights, self.first_moment,
+                          self.second_moment, beta1, beta2, epsilon)
+        self.bias -= self.learning_rate * grad_bias
+        for position in range(3):
+            self._adam_update(self.context_embeddings[position], grad_context[position],
+                              self.context_first_moment[position],
+                              self.context_second_moment[position], beta1, beta2, epsilon)
+        self._adam_update(self.hidden_weights, grad_hidden_weights,
+                          self.hidden_weights_first_moment, self.hidden_weights_second_moment,
+                          beta1, beta2, epsilon)
+        self._adam_update(self.hidden_bias, grad_hidden_bias,
+                          self.hidden_bias_first_moment, self.hidden_bias_second_moment,
+                          beta1, beta2, epsilon)
+        self._adam_update(self.output_weights, grad_output_weights,
+                          self.output_weights_first_moment, self.output_weights_second_moment,
+                          beta1, beta2, epsilon)
+        output_bias_gradient = np.asarray(grad_output_bias, dtype=np.float32)
+        self.output_bias_first_moment = beta1 * self.output_bias_first_moment + (1 - beta1) * output_bias_gradient
+        self.output_bias_second_moment = beta2 * self.output_bias_second_moment + (1 - beta2) * (output_bias_gradient ** 2)
+        output_first_hat = self.output_bias_first_moment / (1 - beta1 ** self.step_number)
+        output_second_hat = self.output_bias_second_moment / (1 - beta2 ** self.step_number)
+        self.output_bias -= self.learning_rate * output_first_hat / (np.sqrt(output_second_hat) + epsilon)
+
+        return float(-np.mean(labels * np.log(probabilities + 1e-9)
+                             + (1 - labels) * np.log(1 - probabilities + 1e-9)))
 
     def predict(self, features, batch_size=200_000):
+        if len(features) == 0:
+            return np.empty(0, dtype=np.float32)
         return np.concatenate([
             self.logits(features[index:index + batch_size])
             for index in range(0, len(features), batch_size)
-        ])
+        ]).astype(np.float32, copy=False)
 
     def state(self):
-        return self.weights.copy(), np.float32(self.bias)
+        return {
+            "weights": self.weights.copy(), "bias": np.float32(self.bias),
+            "first_moment": self.first_moment.copy(),
+            "second_moment": self.second_moment.copy(), "step_number": self.step_number,
+            "context_ranges": self.context_ranges,
+            "context_embeddings": None if self.context_embeddings is None else [x.copy() for x in self.context_embeddings],
+            "hidden_weights": None if self.hidden_weights is None else self.hidden_weights.copy(),
+            "hidden_bias": None if self.hidden_bias is None else self.hidden_bias.copy(),
+            "output_weights": None if self.output_weights is None else self.output_weights.copy(),
+            "output_bias": self.output_bias,
+            "context_first_moment": None if self.context_first_moment is None else [x.copy() for x in self.context_first_moment],
+            "context_second_moment": None if self.context_second_moment is None else [x.copy() for x in self.context_second_moment],
+            "hidden_weights_first_moment": None if self.hidden_weights_first_moment is None else self.hidden_weights_first_moment.copy(),
+            "hidden_weights_second_moment": None if self.hidden_weights_second_moment is None else self.hidden_weights_second_moment.copy(),
+            "hidden_bias_first_moment": None if self.hidden_bias_first_moment is None else self.hidden_bias_first_moment.copy(),
+            "hidden_bias_second_moment": None if self.hidden_bias_second_moment is None else self.hidden_bias_second_moment.copy(),
+            "output_weights_first_moment": None if self.output_weights_first_moment is None else self.output_weights_first_moment.copy(),
+            "output_weights_second_moment": None if self.output_weights_second_moment is None else self.output_weights_second_moment.copy(),
+            "output_bias_first_moment": self.output_bias_first_moment,
+            "output_bias_second_moment": self.output_bias_second_moment,
+        }
 
     def load_state(self, state):
-        self.weights, self.bias = state
+        if isinstance(state, tuple):
+            self.weights, self.bias = state
+            return
+        for name in ("weights", "bias", "first_moment", "second_moment", "step_number",
+                     "context_ranges", "context_embeddings", "hidden_weights", "hidden_bias",
+                     "output_weights", "output_bias", "context_first_moment",
+                     "context_second_moment", "hidden_weights_first_moment",
+                     "hidden_weights_second_moment", "hidden_bias_first_moment",
+                     "hidden_bias_second_moment", "output_weights_first_moment",
+                     "output_weights_second_moment", "output_bias_first_moment",
+                     "output_bias_second_moment"):
+            value = state[name]
+            if isinstance(value, list):
+                value = [x.copy() for x in value]
+            elif isinstance(value, np.ndarray):
+                value = value.copy()
+            setattr(self, name, value)
```

## Experiment 3

- Generation: 3
- Parent experiment: 2
- Status: scored
- Hypothesis: Apply stronger L2 regularization only to the nonlinear context residual parameters while leaving additive ID parameters at the current regularization.
- Validation GAUC: 0.6691888977921354
- Validation nDCG@5: 0.5369344217815536
- Validation primary: 0.6030616597868446
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -139,11 +139,12 @@
         grad_bias = np.float32(gradient.sum())
 
         indices, combined, hidden, hidden_pre = cache
-        grad_output_weights = hidden.T.dot(gradient) + self.l2 * self.output_weights
+        context_l2 = 10.0 * self.l2
+        grad_output_weights = hidden.T.dot(gradient) + context_l2 * self.output_weights
         grad_output_bias = np.float32(gradient.sum())
         hidden_gradient = gradient[:, None] * self.output_weights[None, :]
         hidden_gradient *= (hidden_pre > 0.0)
-        grad_hidden_weights = combined.T.dot(hidden_gradient) + self.l2 * self.hidden_weights
+        grad_hidden_weights = combined.T.dot(hidden_gradient) + context_l2 * self.hidden_weights
         grad_hidden_bias = hidden_gradient.sum(axis=0)
         grad_combined = hidden_gradient.dot(self.hidden_weights.T)
         grad_context = []
@@ -151,7 +152,7 @@
             grad = np.zeros_like(self.context_embeddings[position])
             np.add.at(grad, index, grad_combined[:, position * self.context_embedding_dimension:
                                                    (position + 1) * self.context_embedding_dimension])
-            grad += self.l2 * self.context_embeddings[position]
+            grad += context_l2 * self.context_embeddings[position]
             grad_context.append(grad)
 
         self.step_number += 1
```

## Experiment 4

- Generation: 4
- Parent experiment: 2
- Status: scored
- Hypothesis: Reduce the context residual capacity from 32 hidden units to 8 hidden units.
- Validation GAUC: 0.669028465960081
- Validation nDCG@5: 0.5368850842556074
- Validation primary: 0.6029567751078442
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -18,7 +18,7 @@
         self.step_number = 0
 
         self.context_embedding_dimension = 8
-        self.hidden_units = 32
+        self.hidden_units = 8
         self.context_ranges = None
         self.context_embeddings = None
         self.hidden_weights = None
```
