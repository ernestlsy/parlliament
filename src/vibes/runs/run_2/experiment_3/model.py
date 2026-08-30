"""Additive user/item effects with low-rank and hashed pair interactions."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    CROSS_BUCKETS = 262144

    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        if np.isscalar(dimension):
            user_dimension = item_dimension = int(dimension)
        else:
            user_dimension, item_dimension = map(int, dimension[:2])
        self.user_weights = np.zeros(user_dimension, dtype=np.float32)
        self.item_weights = np.zeros(item_dimension, dtype=np.float32)
        self.user_embeddings = np.zeros((user_dimension, 16), dtype=np.float32)
        self.item_embeddings = np.zeros((item_dimension, 16), dtype=np.float32)
        self.cross_weights = np.zeros(self.CROSS_BUCKETS, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.first_moment = [np.zeros_like(value) for value in self._parameters()]
        self.second_moment = [np.zeros_like(value) for value in self._parameters()]
        self.step_number = 0

    def _parameters(self):
        return (self.user_weights, self.item_weights,
                self.user_embeddings, self.item_embeddings,
                self.cross_weights)

    def _buckets(self, features):
        users = np.asarray(features)[:, 0].astype(np.uint64)
        items = np.asarray(features)[:, 1].astype(np.uint64)
        hashed = (users * np.uint64(1000003)) ^ (items * np.uint64(1000033))
        hashed ^= hashed >> np.uint64(16)
        return (hashed % np.uint64(self.CROSS_BUCKETS)).astype(np.intp)

    def logits(self, features):
        features = np.asarray(features)
        users = features[:, 0]
        items = features[:, 1]
        buckets = self._buckets(features)
        return (self.bias + self.user_weights[users] + self.item_weights[items]
                + np.sum(self.user_embeddings[users] * self.item_embeddings[items], axis=1)
                + self.cross_weights[buckets])

    def step(self, features, labels):
        size = len(labels)
        if size == 0:
            return 0.0
        features = np.asarray(features)
        users = features[:, 0]
        items = features[:, 1]
        buckets = self._buckets(features)
        labels = np.asarray(labels, dtype=np.float32)
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        grad_user_weights = np.zeros_like(self.user_weights)
        grad_item_weights = np.zeros_like(self.item_weights)
        grad_user_embeddings = np.zeros_like(self.user_embeddings)
        grad_item_embeddings = np.zeros_like(self.item_embeddings)
        grad_cross_weights = np.zeros_like(self.cross_weights)
        np.add.at(grad_user_weights, users, gradient)
        np.add.at(grad_item_weights, items, gradient)
        np.add.at(grad_user_embeddings, users,
                  gradient[:, None] * self.item_embeddings[items])
        np.add.at(grad_item_embeddings, items,
                  gradient[:, None] * self.user_embeddings[users])
        np.add.at(grad_cross_weights, buckets, gradient)
        gradients = (grad_user_weights, grad_item_weights,
                     grad_user_embeddings, grad_item_embeddings,
                     grad_cross_weights)
        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, first, second, gradient_value in zip(
                self._parameters(), self.first_moment, self.second_moment, gradients):
            gradient_value = gradient_value + self.l2 * parameter
            first *= beta1
            first += (1 - beta1) * gradient_value
            second *= beta2
            second += (1 - beta2) * (gradient_value * gradient_value)
            first_hat = first / (1 - beta1 ** self.step_number)
            second_hat = second / (1 - beta2 ** self.step_number)
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.bias -= self.learning_rate * gradient.sum()
        return float(-np.mean(labels * np.log(probabilities + 1e-9)
                              + (1 - labels) * np.log(1 - probabilities + 1e-9)))

    def predict(self, features, batch_size=200_000):
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return (self.user_weights.copy(), self.item_weights.copy(),
                self.user_embeddings.copy(), self.item_embeddings.copy(),
                self.cross_weights.copy(), np.float32(self.bias),
                tuple(value.copy() for value in self.first_moment),
                tuple(value.copy() for value in self.second_moment),
                int(self.step_number))

    def load_state(self, state):
        (self.user_weights, self.item_weights, self.user_embeddings,
         self.item_embeddings, self.cross_weights, self.bias, first_moment,
         second_moment, self.step_number) = state
        self.user_weights = np.asarray(self.user_weights, dtype=np.float32)
        self.item_weights = np.asarray(self.item_weights, dtype=np.float32)
        self.user_embeddings = np.asarray(self.user_embeddings, dtype=np.float32)
        self.item_embeddings = np.asarray(self.item_embeddings, dtype=np.float32)
        self.cross_weights = np.asarray(self.cross_weights, dtype=np.float32)
        self.first_moment = [np.asarray(value, dtype=np.float32) for value in first_moment]
        self.second_moment = [np.asarray(value, dtype=np.float32) for value in second_moment]
        self.bias = np.float32(self.bias)
        self.step_number = int(self.step_number)
