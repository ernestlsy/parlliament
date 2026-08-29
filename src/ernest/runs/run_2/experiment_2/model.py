"""Additive ID effects with a compact, bounded FM interaction."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        self.weights = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.rank = 8
        self.factors = np.zeros((dimension, self.rank), dtype=np.float32)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.factor_first_moment = np.zeros_like(self.factors)
        self.factor_second_moment = np.zeros_like(self.factors)
        self.step_number = 0

    def logits(self, features):
        features = np.asarray(features, dtype=np.int64)
        additive = self.weights[features].sum(axis=1)
        bounded = np.tanh(self.factors)
        interaction = (bounded[features[:, 0]] * bounded[features[:, 1]]).sum(axis=1)
        return np.asarray(self.bias + additive + interaction, dtype=np.float32)

    def step(self, features, labels):
        features = np.asarray(features, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float32)
        size = len(labels)
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, features, gradient[:, None])
        grad_weights += self.l2 * self.weights

        bounded = np.tanh(self.factors)
        grad_factors = np.zeros_like(self.factors)
        left = bounded[features[:, 0]]
        right = bounded[features[:, 1]]
        np.add.at(grad_factors, features[:, 0], gradient[:, None] * right)
        np.add.at(grad_factors, features[:, 1], gradient[:, None] * left)
        grad_factors *= 1.0 - bounded * bounded
        grad_factors += self.l2 * self.factors

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.first_moment *= beta1
        self.first_moment += (1 - beta1) * grad_weights
        self.second_moment *= beta2
        self.second_moment += (1 - beta2) * (grad_weights * grad_weights)
        self.factor_first_moment *= beta1
        self.factor_first_moment += (1 - beta1) * grad_factors
        self.factor_second_moment *= beta2
        self.factor_second_moment += (1 - beta2) * (grad_factors * grad_factors)
        first_hat = self.first_moment / (1 - beta1 ** self.step_number)
        second_hat = self.second_moment / (1 - beta2 ** self.step_number)
        factor_first_hat = self.factor_first_moment / (1 - beta1 ** self.step_number)
        factor_second_hat = self.factor_second_moment / (1 - beta2 ** self.step_number)
        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.factors -= self.learning_rate * factor_first_hat / (np.sqrt(factor_second_hat) + epsilon)
        self.bias -= self.learning_rate * gradient.sum()
        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1 - labels) * np.log(1 - probabilities + 1e-9)
        ))

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias), self.factors.copy()

    def load_state(self, state):
        self.weights, self.bias, self.factors = state
