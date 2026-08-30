"""Pointwise factorization-machine model for encoded user/item features."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        self.dimension = int(dimension)
        self.interaction_dimension = 16
        self.weights = np.zeros(self.dimension, dtype=np.float32)
        self.embeddings = np.zeros(
            (self.dimension, self.interaction_dimension), dtype=np.float32
        )
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.embedding_first_moment = np.zeros_like(self.embeddings)
        self.embedding_second_moment = np.zeros_like(self.embeddings)
        self.step_number = 0

    def logits(self, features):
        features = np.asarray(features, dtype=np.intp)
        users = features[:, 0]
        items = features[:, 1]
        interaction = (self.embeddings[users] * self.embeddings[items]).sum(axis=1)
        result = self.bias + self.weights[users] + self.weights[items] + interaction
        return np.nan_to_num(result, nan=0.0, posinf=30.0, neginf=-30.0)

    def step(self, features, labels):
        size = len(labels)
        if size == 0:
            return 0.0
        features = np.asarray(features, dtype=np.intp)
        labels = np.asarray(labels, dtype=np.float32)
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        grad_weights = np.zeros_like(self.weights)
        users = features[:, 0]
        items = features[:, 1]
        user_embeddings = self.embeddings[users].copy()
        item_embeddings = self.embeddings[items].copy()
        grad_embeddings = np.zeros_like(self.embeddings)
        np.add.at(grad_weights, users, gradient)
        np.add.at(grad_weights, items, gradient)
        np.add.at(grad_embeddings, users, gradient[:, None] * item_embeddings)
        np.add.at(grad_embeddings, items, gradient[:, None] * user_embeddings)
        grad_weights += self.l2 * self.weights
        grad_embeddings += self.l2 * self.embeddings
        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.first_moment *= beta1
        self.first_moment += (1 - beta1) * grad_weights
        self.second_moment *= beta2
        self.second_moment += (1 - beta2) * (grad_weights * grad_weights)
        self.embedding_first_moment *= beta1
        self.embedding_first_moment += (1 - beta1) * grad_embeddings
        self.embedding_second_moment *= beta2
        self.embedding_second_moment += (1 - beta2) * (grad_embeddings * grad_embeddings)
        first_hat = self.first_moment / (1 - beta1 ** self.step_number)
        second_hat = self.second_moment / (1 - beta2 ** self.step_number)
        embedding_first_hat = self.embedding_first_moment / (1 - beta1 ** self.step_number)
        embedding_second_hat = self.embedding_second_moment / (1 - beta2 ** self.step_number)
        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.embeddings -= self.learning_rate * embedding_first_hat / (
            np.sqrt(embedding_second_hat) + epsilon
        )
        self.weights = np.nan_to_num(self.weights, nan=0.0, posinf=10.0, neginf=-10.0)
        self.embeddings = np.nan_to_num(
            self.embeddings, nan=0.0, posinf=10.0, neginf=-10.0
        )
        self.bias -= self.learning_rate * gradient.sum()
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
        return self.weights.copy(), np.float32(self.bias), self.embeddings.copy()

    def load_state(self, state):
        if len(state) == 2:
            self.weights, self.bias = state
            self.embeddings.fill(0.0)
        else:
            self.weights, self.bias, self.embeddings = state
        self.weights = np.asarray(self.weights, dtype=np.float32).copy()
        self.bias = np.float32(self.bias)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32).copy()
