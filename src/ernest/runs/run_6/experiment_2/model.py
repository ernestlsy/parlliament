"""Fresh-start additive ID and request-context model.

The model remains a neutral additive scorer: every encoded feature, including
impression-time request-context fields, contributes its own learned weight.
Training uses a within-user pairwise logistic ranking objective.
"""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        self.weights = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.step_number = 0

    def logits(self, features):
        features = np.asarray(features, dtype=np.intp)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional index matrix")
        return self.bias + np.sum(self.weights[features], axis=1, dtype=np.float32)

    def step(self, features, labels):
        features = np.asarray(features, dtype=np.intp)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        if features.ndim != 2 or features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must contain the same number of rows")

        logits = self.logits(features)
        example_gradient = np.zeros(len(labels), dtype=np.float32)
        pair_count = 0
        loss_total = 0.0
        user_ids = features[:, 0]

        for user_id in np.unique(user_ids):
            rows = np.flatnonzero(user_ids == user_id)
            positive_rows = rows[labels[rows] > 0.5]
            negative_rows = rows[labels[rows] <= 0.5]
            if len(positive_rows) == 0 or len(negative_rows) == 0:
                continue

            pair_positive = np.repeat(positive_rows, len(negative_rows))
            pair_negative = np.tile(negative_rows, len(positive_rows))
            differences = logits[pair_positive] - logits[pair_negative]
            pair_probability = sigmoid(-differences).astype(np.float32)
            np.add.at(example_gradient, pair_positive, -pair_probability)
            np.add.at(example_gradient, pair_negative, pair_probability)
            loss_total += float(np.logaddexp(0.0, -differences).sum())
            pair_count += len(differences)

        if pair_count:
            example_gradient /= pair_count
            loss = loss_total / pair_count
        else:
            loss = 0.0

        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, features, example_gradient[:, None])
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
        self.bias -= self.learning_rate * example_gradient.sum()
        return float(loss)

    def predict(self, features, batch_size=200_000):
        features = np.asarray(features, dtype=np.intp)
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
