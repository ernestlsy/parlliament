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

    def step(self, features, labels, sample_weights):
        sample_weights = np.asarray(sample_weights)
        if sample_weights.ndim != 1 or len(sample_weights) != len(labels):
            raise ValueError("sample_weights must be one-dimensional and aligned with labels")
        try:
            if not np.all(np.isfinite(sample_weights)) or not np.all(sample_weights > 0):
                raise ValueError("sample_weights must contain finite strictly positive values")
            sample_weight_sum = sample_weights.sum()
            if not np.isfinite(sample_weight_sum) or sample_weight_sum <= 0:
                raise ValueError("sample_weights must have a finite positive sum")
        except TypeError as error:
            raise ValueError("sample_weights must contain finite strictly positive values") from error

        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = (
            (probabilities - labels) * sample_weights / sample_weights.sum()
        ).astype(np.float32)
        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, features, gradient[:, None])
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
        self.bias -= self.learning_rate * gradient.sum()
        return float(-np.sum(
            sample_weights * (
                labels * np.log(probabilities + 1e-9)
                + (1 - labels) * np.log(1 - probabilities + 1e-9)
            )
        ) / sample_weights.sum())

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.weights, self.bias = state
