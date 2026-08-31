"""Fresh-start additive ID model with within-user RankNet optimization."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        # The shared vector covers every encoded column supplied by data.py,
        # including the user, video, and request-context fields.
        self.weights = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.step_number = 0

    def logits(self, features):
        return self.bias + self.weights[features].sum(axis=1)

    def step(self, features, labels, user_ids):
        labels = np.asarray(labels)
        user_ids = np.asarray(user_ids)
        logits = self.logits(features)
        row_gradient = np.zeros(len(labels), dtype=np.float64)
        total_loss = 0.0
        pair_count = 0

        for user_id in np.unique(user_ids):
            group = np.flatnonzero(user_ids == user_id)
            positive = group[labels[group] == 1]
            negative = group[labels[group] == 0]
            if len(positive) == 0 or len(negative) == 0:
                continue

            differences = (
                logits[positive, None].astype(np.float64)
                - logits[negative][None, :].astype(np.float64)
            )
            pair_probabilities = sigmoid(differences)
            positive_gradient = pair_probabilities - 1.0
            negative_gradient = 1.0 - pair_probabilities

            row_gradient[positive] += positive_gradient.sum(axis=1)
            row_gradient[negative] += negative_gradient.sum(axis=0)
            total_loss += float(np.logaddexp(0.0, -differences).sum())
            pair_count += differences.size

        if pair_count == 0:
            return 0.0

        row_gradient /= pair_count
        grad_weights = np.zeros_like(self.weights)
        np.add.at(
            grad_weights,
            features,
            np.broadcast_to(row_gradient[:, None], features.shape),
        )
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
        self.bias -= self.learning_rate * row_gradient.sum()
        return float(total_loss / pair_count)

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.weights, self.bias = state
