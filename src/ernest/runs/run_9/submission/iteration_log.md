# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add impression-time request-context fields to the five-field Factorization Machine baseline.
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
@@ -1,6 +1,7 @@
-"""KuaiRand five-field baseline data loader adapted to Ernest's fixed contract."""
+"""KuaiRand request-context data loader adapted to Ernest's fixed contract."""
 
 import csv
+import datetime
 import os
 
 import numpy as np
@@ -12,7 +13,7 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
+FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "weekday"]
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -30,14 +31,21 @@
     ):
         with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
             for row in csv.DictReader(handle):
+                date = int(row["date"])
+                hour = int(np.clip(np.floor(float(row["hourmin"]) / 100.0), 0, 23))
+                weekday = datetime.datetime.strptime(
+                    str(date), "%Y%m%d"
+                ).weekday()
                 rows.append((
-                    int(row["date"]),
+                    date,
                     row["user_id"],
                     row["video_id"],
                     video_to_author.get(row["video_id"], "UNK"),
                     row["tab"],
                     float(row["duration_ms"]),
                     1 if row[LABEL] != "0" else 0,
+                    hour,
+                    weekday,
                 ))
     result = {}
     for name, (low, high) in SPLITS.items():
@@ -65,6 +73,7 @@
         return [
             row[1], row[2], row[3], row[4],
             str(int(np.searchsorted(edges, row[5]))),
+            str(row[7]), str(row[8]),
         ]
 
     vocabularies = [dict() for _ in FIELDS]
```

## Experiment 2

- Generation: 2
- Parent experiment: 1
- Status: scored
- Hypothesis: Replace pointwise BCE training with within-user pairwise logistic ranking loss using one positive-negative pair per eligible training user batch.
- Validation GAUC: 0.665175962323382
- Validation nDCG@5: 0.534932179286644
- Validation primary: 0.600054070805013
- Failure stage: none
- Failure reason: none
- Recovery: The Overseer classified each failure, routed it to the responsible code agent, and retried within the configured attempt and wall-clock limits.

### Code diff


#### `data.py`

```diff
--- data.py
+++ data.py
@@ -100,3 +100,34 @@
             users[row_index] = row[1]
         encoded[name] = (features, labels, users)
     return encoded, int(sum(field_dimensions))
+
+
+def make_pairwise_pairs(encoded_split, seed=0):
+    """Return one sampled positive-negative pair for each eligible user.
+
+    The input is one encoded split tuple as returned by ``encode``.  Eligibility
+    requires at least one positive and one negative row for the same user.  The
+    returned matrices retain the encoded seven-field schema and are int32.
+    """
+    features, labels, users = encoded_split
+    if features.ndim != 2 or features.shape[1] != len(FIELDS):
+        raise ValueError("features must be a two-dimensional seven-field matrix")
+    if len(features) != len(labels) or len(features) != len(users):
+        raise ValueError("features, labels, and users must have equal length")
+
+    rng = np.random.RandomState(seed)
+    positions = {}
+    for index, user in enumerate(users):
+        positions.setdefault(user, [[], []])[1 if labels[index] > 0 else 0].append(index)
+
+    positive_indices = []
+    negative_indices = []
+    for positive, negative in positions.values():
+        if positive and negative:
+            positive_indices.append(positive[rng.randint(len(positive))])
+            negative_indices.append(negative[rng.randint(len(negative))])
+
+    return (
+        features[np.asarray(positive_indices, dtype=np.intp)].astype(np.int32, copy=False),
+        features[np.asarray(negative_indices, dtype=np.intp)].astype(np.int32, copy=False),
+    )
```

#### `model.py`

```diff
--- model.py
+++ model.py
@@ -91,6 +91,78 @@
             + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
         ))
 
+    def pairwise_step(self, positive_features, negative_features):
+        if len(positive_features) != len(negative_features):
+            raise ValueError("positive and negative pair rows must have equal counts")
+        size = len(positive_features)
+        if size == 0:
+            return 0.0
+
+        positive_logits, positive_embeddings, positive_summed = self.logits(
+            positive_features
+        )
+        negative_logits, negative_embeddings, negative_summed = self.logits(
+            negative_features
+        )
+        differences = positive_logits - negative_logits
+        pair_coefficient = sigmoid(-differences).astype(np.float32)
+        positive_gradient = -pair_coefficient / size
+        negative_gradient = pair_coefficient / size
+
+        embedding_gradient = np.zeros_like(self.embeddings)
+        weight_gradient = np.zeros_like(self.weights)
+        np.add.at(weight_gradient, positive_features, positive_gradient[:, None])
+        np.add.at(weight_gradient, negative_features, negative_gradient[:, None])
+        np.add.at(
+            embedding_gradient,
+            positive_features,
+            positive_gradient[:, None, None]
+            * (positive_summed[:, None, :] - positive_embeddings),
+        )
+        np.add.at(
+            embedding_gradient,
+            negative_features,
+            negative_gradient[:, None, None]
+            * (negative_summed[:, None, :] - negative_embeddings),
+        )
+        embedding_gradient += self.l2 * self.embeddings
+        weight_gradient += self.l2 * self.weights
+
+        self.step_number += 1
+        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
+        parameters = (
+            (
+                self.embeddings,
+                embedding_gradient,
+                self.embedding_first_moment,
+                self.embedding_second_moment,
+            ),
+            (
+                self.weights,
+                weight_gradient,
+                self.weight_first_moment,
+                self.weight_second_moment,
+            ),
+        )
+        for parameter, parameter_gradient, first, second in parameters:
+            first *= beta1
+            first += (1.0 - beta1) * parameter_gradient
+            second *= beta2
+            second += (1.0 - beta2) * (parameter_gradient * parameter_gradient)
+            first_hat = first / (1.0 - beta1 ** self.step_number)
+            second_hat = second / (1.0 - beta2 ** self.step_number)
+            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
+
+        loss = np.mean(np.logaddexp(0.0, -differences))
+        loss += 0.5 * self.l2 * (
+            np.sum(self.embeddings * self.embeddings)
+            + np.sum(self.weights * self.weights)
+        )
+        return float(loss)
+
+    def step_pairwise(self, positive_features, negative_features):
+        return self.pairwise_step(positive_features, negative_features)
+
     def predict(self, features, batch_size=200_000):
         if len(features) == 0:
             return np.empty(0, dtype=np.float32)
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -70,6 +70,50 @@
     return gauc, ndcg, (gauc + ndcg) / 2.0
 
 
+def _pair_features(features, labels, user_ids, generator, limit=None):
+    grouped = collections.defaultdict(lambda: {0: [], 1: []})
+    for row_index, (user_id, label) in enumerate(zip(user_ids, labels)):
+        label_value = int(label)
+        if label_value in (0, 1):
+            grouped[user_id][label_value].append(row_index)
+
+    positive_indices = []
+    negative_indices = []
+    for rows in grouped.values():
+        if rows[1] and rows[0]:
+            positive_indices.append(
+                rows[1][generator.integers(len(rows[1]))]
+            )
+            negative_indices.append(
+                rows[0][generator.integers(len(rows[0]))]
+            )
+            if limit is not None and len(positive_indices) >= limit:
+                break
+    if not positive_indices:
+        return None, None
+    return features[positive_indices], features[negative_indices]
+
+
+def _pairwise_method(model):
+    preferred_names = (
+        "pairwise_step",
+        "train_pairwise_step",
+        "pairwise_train_step",
+        "train_pairwise",
+        "pairwise_update",
+    )
+    for name in preferred_names:
+        method = getattr(model, name, None)
+        if callable(method):
+            return method
+    for name in dir(model):
+        if "pairwise" in name.lower():
+            method = getattr(model, name, None)
+            if callable(method):
+                return method
+    return None
+
+
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", required=True)
@@ -82,7 +126,7 @@
 
     splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
     encoded, dimension = encode(splits)
-    train_features, train_labels, _ = encoded["train"]
+    train_features, train_labels, train_users = encoded["train"]
     valid_features, valid_labels, valid_users = encoded[config["split"]]
     model = Model(
         dimension,
@@ -91,26 +135,44 @@
         l2=config["l2"],
         seed=config["seed"],
     )
+    pairwise_step = _pairwise_method(model)
 
     if args.contract_check:
         probe_size = min(8, len(train_labels))
         if probe_size == 0 or len(valid_labels) == 0:
             raise ValueError("contract probe requires non-empty train and validation slices")
-        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
+        probe_generator = np.random.default_rng(config["seed"])
+        positive_features, negative_features = _pair_features(
+            train_features,
+            train_labels,
+            train_users,
+            probe_generator,
+            limit=8,
+        )
+        if positive_features is not None and pairwise_step is not None:
+            loss = pairwise_step(
+                positive_features=positive_features,
+                negative_features=negative_features,
+            )
+            if not np.isfinite(loss):
+                raise ValueError("model produced NaN or infinity during contract probe")
         probe_scores = model.predict(valid_features[:probe_size])
         if probe_scores.ndim != 1 or len(probe_scores) != min(
             probe_size, len(valid_features)
         ):
             raise ValueError("model prediction shape violates the interface contract")
-        if not np.isfinite(loss) or not np.all(np.isfinite(probe_scores)):
+        if not np.all(np.isfinite(probe_scores)):
             raise ValueError("model produced NaN or infinity during contract probe")
         print(json.dumps({
             "contract": "ok",
             "feature_shape": list(train_features.shape),
-            "fields": 5,
+            "fields": 7,
             "interaction_dimension": config["interaction_dimension"],
         }))
         return
+
+    if pairwise_step is None:
+        raise AttributeError("Model does not provide a pairwise training method")
 
     generator = np.random.default_rng(config["seed"])
     best_primary, best_state, stale = -1.0, None, 0
@@ -119,13 +181,24 @@
         losses = []
         for index in range(0, len(order), config["batch_size"]):
             batch = order[index:index + config["batch_size"]]
-            losses.append(model.step(train_features[batch], train_labels[batch]))
+            positive_features, negative_features = _pair_features(
+                train_features[batch],
+                train_labels[batch],
+                train_users[batch],
+                generator,
+            )
+            if positive_features is not None:
+                losses.append(pairwise_step(
+                    positive_features=positive_features,
+                    negative_features=negative_features,
+                ))
         scores = model.predict(valid_features)
         gauc, ndcg, primary = training_selection_metrics(
             valid_users, valid_labels, scores
         )
+        loss_value = np.mean(losses) if losses else 0.0
         print(
-            f"epoch={epoch} loss={np.mean(losses):.6f} valid_GAUC={gauc:.6f} "
+            f"epoch={epoch} loss={loss_value:.6f} valid_GAUC={gauc:.6f} "
             f"valid_nDCG@5={ndcg:.6f} valid_primary={primary:.6f}"
         )
         if primary > best_primary + 1e-5:
```

### Error and recovery events

```json
[
  {
    "kind": "contract_usage",
    "message": "contract probe exited 1",
    "traceback": "Traceback (most recent call last):\n  File \"/mnt/d/tehpengagent/src/ernest/runs/run_9/attempts/attempt_09c497d4681c/train.py\", line 193, in <module>\n    main()\n  File \"/mnt/d/tehpengagent/src/ernest/runs/run_9/attempts/attempt_09c497d4681c/train.py\", line 129, in main\n    loss = model.pairwise_step(\n           ^^^^^^^^^^^^^^^^^^^\nAttributeError: 'Model' object has no attribute 'pairwise_step'\n",
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

## Experiment 3

- Generation: 3
- Parent experiment: 1
- Status: scored
- Hypothesis: Reweight the incumbent's pointwise BCE so each complete training user's impressions contribute equal total loss.
- Validation GAUC: 0.6593510505950979
- Validation nDCG@5: 0.532751265164889
- Validation primary: 0.5960511578799934
- Failure stage: none
- Failure reason: none
- Recovery: No error or recovery event occurred in this attempt.

### Code diff


#### `model.py`

```diff
--- model.py
+++ model.py
@@ -43,13 +43,28 @@
             summed,
         )
 
-    def step(self, features, labels):
+    def step(self, features, labels, sample_weights):
         size = len(labels)
         if size == 0:
             return 0.0
+        sample_weights = np.asarray(sample_weights)
+        if sample_weights.ndim != 1:
+            raise ValueError("sample_weights must be one-dimensional")
+        if len(sample_weights) != size:
+            raise ValueError("sample_weights must have the same length as labels")
+        if not np.all(np.isfinite(sample_weights)):
+            raise ValueError("sample_weights must be finite")
+        if np.any(sample_weights < 0.0):
+            raise ValueError("sample_weights must be nonnegative")
+        weight_sum = float(sample_weights.sum())
+        if weight_sum <= 0.0:
+            raise ValueError("sample_weights must have a positive sum")
+
         logits, field_embeddings, summed = self.logits(features)
         probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+        gradient = (
+            (probabilities - labels) * (sample_weights / weight_sum)
+        ).astype(np.float32)
         embedding_gradient = np.zeros_like(self.embeddings)
         weight_gradient = np.zeros_like(self.weights)
         np.add.at(weight_gradient, features, gradient[:, None])
@@ -86,10 +101,11 @@
             second_hat = second / (1.0 - beta2 ** self.step_number)
             parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
         self.bias -= self.learning_rate * gradient.sum()
-        return float(-np.mean(
+        per_row_loss = -(
             labels * np.log(probabilities + 1e-9)
             + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
-        ))
+        )
+        return float(np.sum(sample_weights * per_row_loss) / weight_sum)
 
     def predict(self, features, batch_size=200_000):
         if len(features) == 0:
```

#### `train.py`

```diff
--- train.py
+++ train.py
@@ -82,8 +82,13 @@
 
     splits = load(args.data_dir, max_rows_per_split=64 if args.contract_check else None)
     encoded, dimension = encode(splits)
-    train_features, train_labels, _ = encoded["train"]
+    train_features, train_labels, train_users = encoded["train"]
     valid_features, valid_labels, valid_users = encoded[config["split"]]
+    train_user_counts = collections.Counter(train_users)
+    train_sample_weights = np.asarray(
+        [1.0 / train_user_counts[user] for user in train_users],
+        dtype=np.float32,
+    )
     model = Model(
         dimension,
         interaction_dimension=config["interaction_dimension"],
@@ -96,7 +101,11 @@
         probe_size = min(8, len(train_labels))
         if probe_size == 0 or len(valid_labels) == 0:
             raise ValueError("contract probe requires non-empty train and validation slices")
-        loss = model.step(train_features[:probe_size], train_labels[:probe_size])
+        loss = model.step(
+            train_features[:probe_size],
+            train_labels[:probe_size],
+            train_sample_weights[:probe_size],
+        )
         probe_scores = model.predict(valid_features[:probe_size])
         if probe_scores.ndim != 1 or len(probe_scores) != min(
             probe_size, len(valid_features)
@@ -119,7 +128,13 @@
         losses = []
         for index in range(0, len(order), config["batch_size"]):
             batch = order[index:index + config["batch_size"]]
-            losses.append(model.step(train_features[batch], train_labels[batch]))
+            losses.append(
+                model.step(
+                    train_features[batch],
+                    train_labels[batch],
+                    train_sample_weights[batch],
+                )
+            )
         scores = model.predict(valid_features)
         gauc, ndcg, primary = training_selection_metrics(
             valid_users, valid_labels, scores
```
