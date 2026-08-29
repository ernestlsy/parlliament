"""Additive user/item effects with a low-rank interaction and censored watch head."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6,
                 watch_time_loss_weight=0.1):
        if np.isscalar(dimension):
            user_dimension = item_dimension = int(dimension)
        else:
            user_dimension, item_dimension = map(int, dimension[:2])
        self.user_weights = np.zeros(user_dimension, dtype=np.float32)
        self.item_weights = np.zeros(item_dimension, dtype=np.float32)
        self.user_embeddings = np.zeros((user_dimension, 16), dtype=np.float32)
        self.item_embeddings = np.zeros((item_dimension, 16), dtype=np.float32)
        self.watch_user_weights = np.zeros(user_dimension, dtype=np.float32)
        self.watch_item_weights = np.zeros(item_dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.watch_bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.watch_time_loss_weight = float(watch_time_loss_weight)
        self.first_moment = [np.zeros_like(value) for value in self._parameters()]
        self.second_moment = [np.zeros_like(value) for value in self._parameters()]
        self.step_number = 0

    def _parameters(self):
        return (self.user_weights, self.item_weights, self.user_embeddings,
                self.item_embeddings, self.watch_user_weights,
                self.watch_item_weights)

    def logits(self, features):
        features = np.asarray(features)
        users, items = features[:, 0].astype(np.int64), features[:, 1].astype(np.int64)
        return (self.bias + self.user_weights[users] + self.item_weights[items]
                + np.sum(self.user_embeddings[users] * self.item_embeddings[items], axis=1))

    def watch_logits(self, features):
        features = np.asarray(features)
        users, items = features[:, 0].astype(np.int64), features[:, 1].astype(np.int64)
        return (self.watch_bias + self.watch_user_weights[users] +
                self.watch_item_weights[items] +
                np.sum(self.user_embeddings[users] * self.item_embeddings[items], axis=1))

    def step(self, features, labels, watch_times=None, watch_time=None):
        if watch_times is None:
            watch_times = watch_time
        size = len(labels)
        if size == 0:
            return 0.0
        features = np.asarray(features)
        users = features[:, 0].astype(np.int64)
        items = features[:, 1].astype(np.int64)
        labels = np.asarray(labels, dtype=np.float32)
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        grads = [np.zeros_like(value) for value in self._parameters()]
        np.add.at(grads[0], users, gradient)
        np.add.at(grads[1], items, gradient)
        np.add.at(grads[2], users, gradient[:, None] * self.item_embeddings[items])
        np.add.at(grads[3], items, gradient[:, None] * self.user_embeddings[users])

        auxiliary_loss = 0.0
        if watch_times is not None:
            observed = np.asarray(watch_times, dtype=np.float32).reshape(-1)
            eligible = (labels > 0.5) & np.isfinite(observed)
            if np.any(eligible):
                prediction = self.watch_logits(features)
                hinge = np.maximum(0.0, observed - prediction)
                active = eligible & (observed > prediction)
                count = float(np.sum(eligible))
                scale = self.watch_time_loss_weight / count
                watch_gradient = (-scale * active.astype(np.float32))
                np.add.at(grads[4], users, watch_gradient)
                np.add.at(grads[5], items, watch_gradient)
                np.add.at(grads[2], users, watch_gradient[:, None] * self.item_embeddings[items])
                np.add.at(grads[3], items, watch_gradient[:, None] * self.user_embeddings[users])
                self.watch_bias -= self.learning_rate * watch_gradient.sum()
                auxiliary_loss = self.watch_time_loss_weight * float(np.mean(hinge[eligible]))

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, first, second, gradient_value in zip(
                self._parameters(), self.first_moment, self.second_moment, grads):
            gradient_value = gradient_value + self.l2 * parameter
            first *= beta1
            first += (1 - beta1) * gradient_value
            second *= beta2
            second += (1 - beta2) * (gradient_value * gradient_value)
            first_hat = first / (1 - beta1 ** self.step_number)
            second_hat = second / (1 - beta2 ** self.step_number)
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.bias -= self.learning_rate * gradient.sum()
        classification_loss = -np.mean(labels * np.log(probabilities + 1e-9) +
                                         (1 - labels) * np.log(1 - probabilities + 1e-9))
        return float(classification_loss + auxiliary_loss)

    def predict(self, features, batch_size=200_000):
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([self.logits(features[index:index + batch_size])
                               for index in range(0, len(features), batch_size)])

    def state(self):
        return (self.user_weights.copy(), self.item_weights.copy(),
                self.user_embeddings.copy(), self.item_embeddings.copy(),
                self.watch_user_weights.copy(), self.watch_item_weights.copy(),
                np.float32(self.bias), np.float32(self.watch_bias),
                tuple(value.copy() for value in self.first_moment),
                tuple(value.copy() for value in self.second_moment), int(self.step_number))

    def load_state(self, state):
        if len(state) == 8:
            (self.user_weights, self.item_weights, self.user_embeddings,
             self.item_embeddings, self.bias, first_moment, second_moment,
             self.step_number) = state
            self.watch_user_weights = np.zeros_like(self.user_weights)
            self.watch_item_weights = np.zeros_like(self.item_weights)
            self.watch_bias = np.float32(0.0)
        else:
            (self.user_weights, self.item_weights, self.user_embeddings,
             self.item_embeddings, self.watch_user_weights, self.watch_item_weights,
             self.bias, self.watch_bias, first_moment, second_moment,
             self.step_number) = state
        self.user_weights = np.asarray(self.user_weights, dtype=np.float32)
        self.item_weights = np.asarray(self.item_weights, dtype=np.float32)
        self.user_embeddings = np.asarray(self.user_embeddings, dtype=np.float32)
        self.item_embeddings = np.asarray(self.item_embeddings, dtype=np.float32)
        self.watch_user_weights = np.asarray(self.watch_user_weights, dtype=np.float32)
        self.watch_item_weights = np.asarray(self.watch_item_weights, dtype=np.float32)
        self.first_moment = [np.asarray(value, dtype=np.float32) for value in first_moment]
        self.second_moment = [np.asarray(value, dtype=np.float32) for value in second_moment]
        if len(self.first_moment) != len(self._parameters()):
            self.first_moment = [np.zeros_like(value) for value in self._parameters()]
            self.second_moment = [np.zeros_like(value) for value in self._parameters()]
        self.bias = np.float32(self.bias)
        self.watch_bias = np.float32(self.watch_bias)
        self.step_number = int(self.step_number)
