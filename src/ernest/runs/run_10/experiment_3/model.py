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
        pair_count = len(positive_features)
        positive_logits = self.logits(positive_features)
        negative_logits = self.logits(negative_features)
        margins = positive_logits - negative_logits
        loss = np.mean(np.logaddexp(0.0, -margins))
        pair_gradient = ((sigmoid(margins) - 1.0) / pair_count).astype(np.float32)

        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, positive_features, pair_gradient[:, None])
        np.add.at(grad_weights, negative_features, -pair_gradient[:, None])
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

        return float(loss)

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.weights, self.bias = state
