"""Official-style NumPy Factorization Machine seed model."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, embedding_dim=16, learning_rate=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.embeddings = rng.normal(0, 0.01, (dimension, embedding_dim)).astype(np.float32)
        self.linear = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.first_moment_embeddings = np.zeros_like(self.embeddings)
        self.second_moment_embeddings = np.zeros_like(self.embeddings)
        self.first_moment_linear = np.zeros_like(self.linear)
        self.second_moment_linear = np.zeros_like(self.linear)
        self.step_number = 0

    def logits(self, features):
        embedded = self.embeddings[features]
        summed = embedded.sum(1)
        interaction = 0.5 * ((summed ** 2).sum(1) - (embedded ** 2).sum((1, 2)))
        return self.bias + self.linear[features].sum(1) + interaction, embedded, summed

    def step(self, features, labels):
        size = len(labels)
        logits, embedded, summed = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        grad_embeddings = np.zeros_like(self.embeddings)
        grad_linear = np.zeros_like(self.linear)
        np.add.at(grad_linear, features, gradient[:, None])
        np.add.at(
            grad_embeddings,
            features,
            gradient[:, None, None] * (summed[:, None, :] - embedded),
        )
        grad_embeddings += self.l2 * self.embeddings
        grad_linear += self.l2 * self.linear
        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        parameters = (
            (self.embeddings, grad_embeddings, self.first_moment_embeddings, self.second_moment_embeddings),
            (self.linear, grad_linear, self.first_moment_linear, self.second_moment_linear),
        )
        for parameter, grad, first, second in parameters:
            first *= beta1
            first += (1 - beta1) * grad
            second *= beta2
            second += (1 - beta2) * (grad * grad)
            first_hat = first / (1 - beta1 ** self.step_number)
            second_hat = second / (1 - beta2 ** self.step_number)
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.bias -= self.learning_rate * gradient.sum()
        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1 - labels) * np.log(1 - probabilities + 1e-9)
        ))

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])[0]
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.embeddings.copy(), self.linear.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.embeddings, self.linear, self.bias = state

