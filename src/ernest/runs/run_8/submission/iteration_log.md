# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add impression-time request-context features tab, hour, and weekday to the neutral user/video additive scorer while leaving model, loss, and training unchanged.
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
+"""Neutral seed data contract using user, item, and request-context identifiers."""
 
 import csv
+import math
 import os
+from datetime import datetime
 
 import numpy as np
 
@@ -12,7 +14,7 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -23,9 +25,17 @@
     ):
         with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
             for row in csv.DictReader(handle):
+                date = int(row["date"])
+                hour = int(max(0, min(23, math.floor(float(row["hourmin"]) / 100))))
+                weekday = datetime.strptime(str(date), "%Y%m%d").weekday()
                 rows.append((
-                    int(row["date"]), row["user_id"], row["video_id"],
+                    date,
+                    row["user_id"],
+                    row["video_id"],
                     1 if row[LABEL] != "0" else 0,
+                    int(row["tab"]),
+                    hour,
+                    weekday,
                 ))
     result = {}
     for name, (low, high) in SPLITS.items():
@@ -38,8 +48,9 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
+
     def raw(row):
-        return [row[1], row[2]]
+        return [row[1], row[2], row[4], row[5], row[6]]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
@@ -57,7 +68,8 @@
         for row_index, row in enumerate(rows):
             for field_index, value in enumerate(raw(row)):
                 features[row_index, field_index] = (
-                    vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
+                    vocabs[field_index].get(value, unknown[field_index])
+                    + offsets[field_index]
                 )
             labels[row_index] = row[3]
             users[row_index] = row[1]
```

## Experiment 2

- Generation: 2
- Parent experiment: 1
- Status: scored
- Hypothesis: Replace pointwise BCE with a within-user pairwise logistic ranking loss using the same rows and features.
- Validation GAUC: 0.6659210159749864
- Validation nDCG@5: 0.5349842836902958
- Validation primary: 0.6004526498326411
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -21,13 +21,45 @@
         return self.bias + self.weights[features].sum(1)
 
     def step(self, features, labels):
-        size = len(labels)
         logits = self.logits(features)
-        probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+        row_gradients = np.zeros(len(labels), dtype=np.float32)
+        total_pairs = 0
+        total_loss = 0.0
+
+        if len(labels):
+            _, group_ids = np.unique(features[:, 0], return_inverse=True)
+            for group_id in range(group_ids.max() + 1):
+                rows = np.flatnonzero(group_ids == group_id)
+                positive_rows = rows[labels[rows] == 1]
+                negative_rows = rows[labels[rows] == 0]
+                if not len(positive_rows) or not len(negative_rows):
+                    continue
+
+                positive_logits = logits[positive_rows]
+                negative_logits = logits[negative_rows]
+                differences = positive_logits[:, None] - negative_logits[None, :]
+                absolute_differences = np.abs(differences)
+                total_loss += float(np.sum(
+                    np.maximum(-differences, 0.0)
+                    + np.log1p(np.exp(-absolute_differences)),
+                    dtype=np.float64,
+                ))
+
+                pair_sigmoid = sigmoid(differences)
+                row_gradients[positive_rows] += np.sum(
+                    pair_sigmoid - 1.0, axis=1
+                )
+                row_gradients[negative_rows] += np.sum(
+                    1.0 - pair_sigmoid, axis=0
+                )
+                total_pairs += len(positive_rows) * len(negative_rows)
+
         grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
+        if total_pairs:
+            row_gradients /= float(total_pairs)
+            np.add.at(grad_weights, features, row_gradients[:, None])
         grad_weights += self.l2 * self.weights
+
         self.step_number += 1
         beta1, beta2, epsilon = 0.9, 0.999, 1e-8
         self.first_moment *= beta1
@@ -37,11 +69,10 @@
         first_hat = self.first_moment / (1 - beta1 ** self.step_number)
         second_hat = self.second_moment / (1 - beta2 ** self.step_number)
         self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
-        self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
-            labels * np.log(probabilities + 1e-9)
-            + (1 - labels) * np.log(1 - probabilities + 1e-9)
-        ))
+
+        if total_pairs:
+            return float(total_loss / total_pairs)
+        return 0.0
 
     def predict(self, features, batch_size=200_000):
         return np.concatenate([
```

## Experiment 3

- Generation: 3
- Parent experiment: 1
- Status: scored
- Hypothesis: Replace the additive user/video scorer with a user-video latent-factor interaction model using the existing request-context additive terms.
- Validation GAUC: 0.6445650287134015
- Validation nDCG@5: 0.5253955791277122
- Validation primary: 0.5849803039205568
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -1,6 +1,9 @@
-"""Fresh-start additive ID model with no interaction or baseline-derived architecture."""
+"""User-video latent-factor model with additive request-context terms."""
 
 import numpy as np
+
+
+EMBEDDING_DIMENSION = 16
 
 
 def sigmoid(value):
@@ -9,48 +12,120 @@
 
 class Model:
     def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
-        self.weights = np.zeros(dimension, dtype=np.float32)
-        self.bias = np.float32(0.0)
+        self.dimension = dimension
         self.learning_rate = learning_rate
         self.l2 = l2
-        self.first_moment = np.zeros_like(self.weights)
-        self.second_moment = np.zeros_like(self.weights)
+
+        values = (np.arange(dimension * EMBEDDING_DIMENSION, dtype=np.float32) % 31) + 1.0
+        self.user_embeddings = (values.reshape(dimension, EMBEDDING_DIMENSION) * 1e-4).astype(np.float32)
+        video_values = ((np.arange(dimension * EMBEDDING_DIMENSION, dtype=np.float32) + 11.0) % 37) + 1.0
+        self.video_embeddings = (video_values.reshape(dimension, EMBEDDING_DIMENSION) * 1e-4).astype(np.float32)
+        self.context_weights = np.zeros(dimension, dtype=np.float32)
+        self.bias = np.float32(0.0)
+
+        self.user_first_moment = np.zeros_like(self.user_embeddings)
+        self.user_second_moment = np.zeros_like(self.user_embeddings)
+        self.video_first_moment = np.zeros_like(self.video_embeddings)
+        self.video_second_moment = np.zeros_like(self.video_embeddings)
+        self.context_first_moment = np.zeros_like(self.context_weights)
+        self.context_second_moment = np.zeros_like(self.context_weights)
         self.step_number = 0
 
     def logits(self, features):
-        return self.bias + self.weights[features].sum(1)
+        users = features[:, 0]
+        videos = features[:, 1]
+        contexts = features[:, 2:]
+        interaction = np.sum(
+            self.user_embeddings[users] * self.video_embeddings[videos], axis=1
+        )
+        return self.bias + interaction + self.context_weights[contexts].sum(axis=1)
 
     def step(self, features, labels):
         size = len(labels)
         logits = self.logits(features)
         probabilities = sigmoid(logits)
         gradient = ((probabilities - labels) / size).astype(np.float32)
-        grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
-        grad_weights += self.l2 * self.weights
+
+        users = features[:, 0]
+        videos = features[:, 1]
+        contexts = features[:, 2:]
+        user_vectors = self.user_embeddings[users]
+        video_vectors = self.video_embeddings[videos]
+
+        grad_users = np.zeros_like(self.user_embeddings)
+        grad_videos = np.zeros_like(self.video_embeddings)
+        grad_context = np.zeros_like(self.context_weights)
+        np.add.at(grad_users, users, gradient[:, None] * video_vectors)
+        np.add.at(grad_videos, videos, gradient[:, None] * user_vectors)
+        np.add.at(grad_context, contexts, np.broadcast_to(gradient[:, None], contexts.shape))
+
+        grad_users += self.l2 * self.user_embeddings
+        grad_videos += self.l2 * self.video_embeddings
+        grad_context += self.l2 * self.context_weights
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
+
+        self.user_first_moment *= beta1
+        self.user_first_moment += (1 - beta1) * grad_users
+        self.user_second_moment *= beta2
+        self.user_second_moment += (1 - beta2) * (grad_users * grad_users)
+
+        self.video_first_moment *= beta1
+        self.video_first_moment += (1 - beta1) * grad_videos
+        self.video_second_moment *= beta2
+        self.video_second_moment += (1 - beta2) * (grad_videos * grad_videos)
+
+        self.context_first_moment *= beta1
+        self.context_first_moment += (1 - beta1) * grad_context
+        self.context_second_moment *= beta2
+        self.context_second_moment += (1 - beta2) * (grad_context * grad_context)
+
+        first_scale = 1 - beta1 ** self.step_number
+        second_scale = 1 - beta2 ** self.step_number
+
+        user_first_hat = self.user_first_moment / first_scale
+        user_second_hat = self.user_second_moment / second_scale
+        self.user_embeddings -= self.learning_rate * user_first_hat / (
+            np.sqrt(user_second_hat) + epsilon
+        )
+
+        video_first_hat = self.video_first_moment / first_scale
+        video_second_hat = self.video_second_moment / second_scale
+        self.video_embeddings -= self.learning_rate * video_first_hat / (
+            np.sqrt(video_second_hat) + epsilon
+        )
+
+        context_first_hat = self.context_first_moment / first_scale
+        context_second_hat = self.context_second_moment / second_scale
+        self.context_weights -= self.learning_rate * context_first_hat / (
+            np.sqrt(context_second_hat) + epsilon
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
-        ])
+        ]).astype(np.float32, copy=False)
 
     def state(self):
-        return self.weights.copy(), np.float32(self.bias)
+        return (
+            np.float32(self.bias),
+            self.user_embeddings.copy(),
+            self.video_embeddings.copy(),
+            self.context_weights.copy(),
+        )
 
     def load_state(self, state):
-        self.weights, self.bias = state
+        self.bias = np.float32(state[0])
+        self.user_embeddings = np.asarray(state[1], dtype=np.float32).copy()
+        self.video_embeddings = np.asarray(state[2], dtype=np.float32).copy()
+        self.context_weights = np.asarray(state[3], dtype=np.float32).copy()
```
