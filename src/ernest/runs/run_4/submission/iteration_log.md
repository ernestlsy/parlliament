# Per-iteration run log

Manual interventions: **0 (none)**.

## Experiment 1

- Generation: 1
- Parent experiment: 0
- Status: scored
- Hypothesis: Add impression-time request-context features tab, hour, and weekday to the neutral user/item additive model.
- Validation GAUC: 0.6691805131784498
- Validation nDCG@5: 0.536436334841699
- Validation primary: 0.6028084240100744
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
+"""Neutral additive data contract with user, item, and request-context fields."""
 
 import csv
+import datetime
 import os
+import re
 
 import numpy as np
 
@@ -12,7 +14,37 @@
     "valid": (20220422, 20220428),
     "test": (20220429, 20220508),
 }
-FIELDS = ["user_id", "video_id"]
+FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
+
+
+def _weekday(value):
+    text = str(value)
+    for parser in (
+        lambda: datetime.datetime.strptime(text, "%Y%m%d").weekday(),
+        lambda: datetime.datetime.strptime(text, "%Y-%m-%d").weekday(),
+    ):
+        try:
+            return str(parser())
+        except ValueError:
+            pass
+    return ""
+
+
+def _hour(row):
+    for key in ("hour", "hour_min", "hourmin", "time", "timestamp"):
+        value = row.get(key)
+        if value is None or value == "":
+            continue
+        text = str(value).strip()
+        match = re.search(r"(?:T|\s|^)(\d{1,2})(?::|$)", text)
+        if match is not None:
+            return str(int(match.group(1)))
+        if key == "hour":
+            try:
+                return str(int(float(text)))
+            except ValueError:
+                return text
+    return ""
 
 
 def load(data_dir, max_rows_per_split=None):
@@ -23,8 +55,15 @@
     ):
         with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
             for row in csv.DictReader(handle):
+                date = int(row["date"])
+                tab = row.get("tab", row.get("tab_id", ""))
                 rows.append((
-                    int(row["date"]), row["user_id"], row["video_id"],
+                    date,
+                    row["user_id"],
+                    row["video_id"],
+                    tab,
+                    _hour(row),
+                    _weekday(date),
                     1 if row[LABEL] != "0" else 0,
                 ))
     result = {}
@@ -38,8 +77,9 @@
     train = splits["train"]
     if not train:
         raise ValueError("training split is empty")
+
     def raw(row):
-        return [row[1], row[2]]
+        return [row[1], row[2], row[3], row[4], row[5]]
 
     vocabs = [dict() for _ in FIELDS]
     for row in train:
@@ -59,7 +99,7 @@
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
- Hypothesis: Replace pointwise BCE with within-user pairwise BPR while retaining user ID, video ID, tab, hour, weekday, optimizer, and training schedule.
- Validation GAUC: 0.6664210072780321
- Validation nDCG@5: 0.5352482104970204
- Validation primary: 0.6008346088875263
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
+"""Fresh-start additive ID model trained with within-user BPR ranking loss."""
 
 import numpy as np
 
@@ -23,11 +23,36 @@
     def step(self, features, labels):
         size = len(labels)
         logits = self.logits(features)
-        probabilities = sigmoid(logits)
-        gradient = ((probabilities - labels) / size).astype(np.float32)
+        row_gradient = np.zeros(size, dtype=np.float32)
+        pair_count = 0
+        loss_total = 0.0
+
+        users, inverse = np.unique(features[:, 0], return_inverse=True)
+        for user_index in range(len(users)):
+            rows = np.flatnonzero(inverse == user_index)
+            positive = rows[labels[rows] > 0.5]
+            negative = rows[labels[rows] <= 0.5]
+            if not len(positive) or not len(negative):
+                continue
+
+            pair_count += len(positive) * len(negative)
+            negative_scores = logits[negative]
+            for positive_row in positive:
+                differences = logits[positive_row] - negative_scores
+                loss_total += np.logaddexp(0.0, -differences).sum()
+                pair_probability = sigmoid(-differences).astype(np.float32)
+                row_gradient[positive_row] -= pair_probability.sum()
+                np.add.at(row_gradient, negative, pair_probability)
+
         grad_weights = np.zeros_like(self.weights)
-        np.add.at(grad_weights, features, gradient[:, None])
+        if pair_count:
+            row_gradient /= np.float32(pair_count)
+            loss = loss_total / pair_count
+        else:
+            loss = 0.0
+        np.add.at(grad_weights, features, row_gradient[:, None])
         grad_weights += self.l2 * self.weights
+
         self.step_number += 1
         beta1, beta2, epsilon = 0.9, 0.999, 1e-8
         self.first_moment *= beta1
@@ -37,11 +62,8 @@
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
