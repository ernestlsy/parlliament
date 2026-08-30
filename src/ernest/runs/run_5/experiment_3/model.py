"""Additive categorical model trained with sampled within-user BPR pairs."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    """Additive model over user, item, and request-context categorical fields.

    Training uses sampled positive-negative pairs from the same encoded user,
    while prediction remains an unconditional per-row score computation.
    """

    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        self.weights = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.step_number = 0

    def logits(self, features):
        features = np.asarray(features, dtype=np.int64)
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        return self.bias + self.weights[features].sum(1)

    def _pairs(self, features, labels):
        """Return sampled within-user positive and negative row indices."""
        users = np.asarray(features, dtype=np.int64)[:, 0]
        labels = np.asarray(labels).reshape(-1)
        pair_positive = []
        pair_negative = []
        max_per_user = 64

        for user in np.unique(users):
            rows = np.flatnonzero(users == user)
            positives = rows[labels[rows] > 0.5]
            negatives = rows[labels[rows] <= 0.5]
            if len(positives) == 0 or len(negatives) == 0:
                continue

            pair_count = min(len(positives) * len(negatives), max_per_user)
            if len(positives) * len(negatives) <= max_per_user:
                positive_rows = np.repeat(positives, len(negatives))
                negative_rows = np.tile(negatives, len(positives))
            else:
                positive_rows = positives[np.random.randint(0, len(positives), pair_count)]
                negative_rows = negatives[np.random.randint(0, len(negatives), pair_count)]
            pair_positive.append(positive_rows)
            pair_negative.append(negative_rows)

        if not pair_positive:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        positive_rows = np.concatenate(pair_positive)
        negative_rows = np.concatenate(pair_negative)
        max_pairs = 4096
        if len(positive_rows) > max_pairs:
            selected = np.random.choice(len(positive_rows), max_pairs, replace=False)
            positive_rows = positive_rows[selected]
            negative_rows = negative_rows[selected]
        return positive_rows, negative_rows

    def step(self, features, labels):
        features = np.asarray(features, dtype=np.int64)
        labels = np.asarray(labels).reshape(-1)
        if len(features) == 0 or len(labels) == 0:
            return 0.0

        positive_rows, negative_rows = self._pairs(features, labels)
        if len(positive_rows) == 0:
            return 0.0

        positive_logits = self.logits(features[positive_rows])
        negative_logits = self.logits(features[negative_rows])
        differences = positive_logits - negative_logits
        pair_gradient = (sigmoid(differences) - 1.0).astype(np.float32)
        pair_gradient /= len(pair_gradient)

        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, features[positive_rows], pair_gradient[:, None])
        np.add.at(grad_weights, features[negative_rows], -pair_gradient[:, None])
        grad_weights += self.l2 * self.weights

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.first_moment *= beta1
        self.first_moment += (1 - beta1) * grad_weights
        self.second_moment *= beta2
        self.second_moment += (1 - beta2) * (grad_weights * grad_weights)
        first_hat = self.first_moment / (1 - beta1 ** self.step_number)
        second_hat = self.second_moment / (1 - beta2 ** self.step_number)
        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

        return float(np.mean(np.logaddexp(0.0, -differences)))

    def predict(self, features, batch_size=200_000):
        features = np.asarray(features, dtype=np.int64)
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.weights, self.bias = state
