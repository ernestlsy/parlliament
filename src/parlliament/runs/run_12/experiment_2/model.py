"""Second-order factorization-machine model over encoded categorical IDs."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6, factor_rank=None):
        if factor_rank is None:
            raise ValueError("factor_rank must be provided")
        if factor_rank <= 0:
            raise ValueError("factor_rank must be positive")

        self.weights = np.zeros(dimension, dtype=np.float32)
        rng = np.random.RandomState(0)
        self.factors = rng.normal(
            loc=0.0, scale=0.01, size=(dimension, factor_rank)
        ).astype(np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2

        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.factor_first_moment = np.zeros_like(self.factors)
        self.factor_second_moment = np.zeros_like(self.factors)
        self.bias_first_moment = np.float32(0.0)
        self.bias_second_moment = np.float32(0.0)
        self.step_number = 0

    def logits(self, features):
        selected_weights = self.weights[features]
        selected_factors = self.factors[features]
        factor_sums = selected_factors.sum(axis=1)
        interactions = np.float32(0.5) * (
            (factor_sums * factor_sums).sum(axis=1)
            - (selected_factors * selected_factors).sum(axis=(1, 2))
        )
        return self.bias + selected_weights.sum(axis=1) + interactions

    def step(self, features, labels):
        size = len(labels)
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)

        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, features, gradient[:, None])
        grad_weights += self.l2 * self.weights

        selected_factors = self.factors[features]
        factor_sums = selected_factors.sum(axis=1)
        selected_factor_gradients = (
            gradient[:, None, None]
            * (factor_sums[:, None, :] - selected_factors)
        )
        grad_factors = np.zeros_like(self.factors)
        np.add.at(grad_factors, features, selected_factor_gradients)
        grad_factors += self.l2 * self.factors
        grad_bias = np.float32(gradient.sum())

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

        self.bias_first_moment = np.float32(
            beta1 * self.bias_first_moment + (1 - beta1) * grad_bias
        )
        self.bias_second_moment = np.float32(
            beta2 * self.bias_second_moment + (1 - beta2) * grad_bias * grad_bias
        )

        correction1 = 1 - beta1 ** self.step_number
        correction2 = 1 - beta2 ** self.step_number
        first_hat = self.first_moment / correction1
        second_hat = self.second_moment / correction2
        factor_first_hat = self.factor_first_moment / correction1
        factor_second_hat = self.factor_second_moment / correction2
        bias_first_hat = self.bias_first_moment / correction1
        bias_second_hat = self.bias_second_moment / correction2

        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.factors -= (
            self.learning_rate
            * factor_first_hat
            / (np.sqrt(factor_second_hat) + epsilon)
        )
        self.bias = np.float32(
            self.bias
            - self.learning_rate
            * bias_first_hat
            / (np.sqrt(bias_second_hat) + epsilon)
        )

        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1 - labels) * np.log(1 - probabilities + 1e-9)
        ))

    def predict(self, features, batch_size=200_000):
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), self.factors.copy(), np.float32(self.bias)

    def load_state(self, state):
        weights, factors, bias = state
        self.weights = np.asarray(weights, dtype=np.float32).copy()
        self.factors = np.asarray(factors, dtype=np.float32).copy()
        self.bias = np.float32(bias)
