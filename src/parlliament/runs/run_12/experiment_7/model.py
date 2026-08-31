"""Fresh-start additive ID model with no interaction or baseline-derived architecture."""

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
        return self.bias + self.weights[features].sum(1)

    def step(self, positive_features, negative_features):
        positive_features = np.asarray(positive_features)
        negative_features = np.asarray(negative_features)
        if positive_features.ndim != 2 or negative_features.ndim != 2:
            raise ValueError("positive_features and negative_features must be two-dimensional")
        if positive_features.shape != negative_features.shape:
            raise ValueError("positive_features and negative_features must have the same shape")
        if positive_features.shape[1] != 5:
            raise ValueError("pair feature matrices must have five columns")
        pair_count = positive_features.shape[0]
        if pair_count == 0:
            raise ValueError("pair batches must not be empty")

        positive_logits = self.logits(positive_features)
        negative_logits = self.logits(negative_features)
        delta = positive_logits - negative_logits
        pair_losses = np.logaddexp(np.float32(0.0), -delta)
        loss = float(np.mean(pair_losses, dtype=np.float64))
        gradient = (-sigmoid(-delta) / np.float32(pair_count)).astype(np.float32)

        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, positive_features, gradient[:, None])
        np.add.at(grad_weights, negative_features, -gradient[:, None])
        grad_weights += self.l2 * self.weights
        if not np.isfinite(loss) or not np.all(np.isfinite(grad_weights)):
            raise ValueError("pairwise loss and gradients must be finite")

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.first_moment *= beta1
        self.first_moment += (1 - beta1) * grad_weights
        self.second_moment *= beta2
        self.second_moment += (1 - beta2) * (grad_weights * grad_weights)
        first_hat = self.first_moment / (1 - beta1 ** self.step_number)
        second_hat = self.second_moment / (1 - beta2 ** self.step_number)
        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        return loss

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.weights, self.bias = state
