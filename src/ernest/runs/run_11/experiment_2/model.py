"""KuaiRand five-field Factorization Machine baseline implemented with NumPy."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


class Model:
    def __init__(
        self,
        dimension,
        interaction_dimension=16,
        learning_rate=0.001,
        l2=1e-6,
        seed=0,
    ):
        generator = np.random.default_rng(seed)
        self.embeddings = generator.normal(
            0.0, 0.01, (int(dimension), int(interaction_dimension))
        ).astype(np.float32)
        self.weights = np.zeros(int(dimension), dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.embedding_first_moment = np.zeros_like(self.embeddings)
        self.embedding_second_moment = np.zeros_like(self.embeddings)
        self.weight_first_moment = np.zeros_like(self.weights)
        self.weight_second_moment = np.zeros_like(self.weights)
        self.step_number = 0

    def logits(self, features):
        field_embeddings = self.embeddings[features]
        summed = field_embeddings.sum(axis=1)
        interaction = 0.5 * (
            (summed ** 2).sum(axis=1)
            - (field_embeddings ** 2).sum(axis=(1, 2))
        )
        return (
            self.bias + self.weights[features].sum(axis=1) + interaction,
            field_embeddings,
            summed,
        )

    def step(self, features, labels):
        size = len(labels)
        if size == 0:
            return 0.0
        logits, field_embeddings, summed = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        embedding_gradient = np.zeros_like(self.embeddings)
        weight_gradient = np.zeros_like(self.weights)
        np.add.at(weight_gradient, features, gradient[:, None])
        np.add.at(
            embedding_gradient,
            features,
            gradient[:, None, None] * (summed[:, None, :] - field_embeddings),
        )
        embedding_gradient += self.l2 * self.embeddings
        weight_gradient += self.l2 * self.weights

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        parameters = (
            (
                self.embeddings,
                embedding_gradient,
                self.embedding_first_moment,
                self.embedding_second_moment,
            ),
            (
                self.weights,
                weight_gradient,
                self.weight_first_moment,
                self.weight_second_moment,
            ),
        )
        for parameter, parameter_gradient, first, second in parameters:
            first *= beta1
            first += (1.0 - beta1) * parameter_gradient
            second *= beta2
            second += (1.0 - beta2) * (parameter_gradient * parameter_gradient)
            first_hat = first / (1.0 - beta1 ** self.step_number)
            second_hat = second / (1.0 - beta2 ** self.step_number)
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.bias -= self.learning_rate * gradient.sum()
        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
        ))

    def predict(self, features, batch_size=200_000):
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([
            self.logits(features[index:index + batch_size])[0]
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return (
            self.embeddings.copy(), self.weights.copy(), np.float32(self.bias)
        )

    def load_state(self, state):
        embeddings, weights, bias = state
        self.embeddings = np.asarray(embeddings, dtype=np.float32).copy()
        self.weights = np.asarray(weights, dtype=np.float32).copy()
        self.bias = np.float32(bias)
