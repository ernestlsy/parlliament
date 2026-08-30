"""Additive user/item effects with shared interaction and click supervision."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6,
                 auxiliary_click_loss_weight=0.2):
        if np.isscalar(dimension):
            user_dimension = item_dimension = int(dimension)
        else:
            user_dimension, item_dimension = map(int, dimension[:2])
        self.user_weights = np.zeros(user_dimension, dtype=np.float32)
        self.item_weights = np.zeros(item_dimension, dtype=np.float32)
        self.user_embeddings = np.zeros((user_dimension, 16), dtype=np.float32)
        self.item_embeddings = np.zeros((item_dimension, 16), dtype=np.float32)
        self.click_weights = np.zeros(16, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.click_bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.auxiliary_click_loss_weight = float(auxiliary_click_loss_weight)
        self.first_moment = [np.zeros_like(value) for value in self._parameters()]
        self.second_moment = [np.zeros_like(value) for value in self._parameters()]
        self.step_number = 0

    def _parameters(self):
        return (self.user_weights, self.item_weights,
                self.user_embeddings, self.item_embeddings, self.click_weights)

    def _shared(self, features):
        features = np.asarray(features)
        users = features[:, 0].astype(np.int64)
        items = features[:, 1].astype(np.int64)
        return users, items, self.user_embeddings[users] * self.item_embeddings[items]

    def logits(self, features):
        users, items, shared = self._shared(features)
        return (self.bias + self.user_weights[users] + self.item_weights[items]
                + np.sum(shared, axis=1))

    def _targets(self, labels, is_click):
        if isinstance(labels, dict):
            long_view = labels.get("long_view", labels.get("label"))
            if is_click is None:
                is_click = labels.get("is_click")
            return np.asarray(long_view, dtype=np.float32), is_click
        values = np.asarray(labels)
        if is_click is None and values.ndim == 2 and values.shape[1] >= 2:
            return values[:, 0].astype(np.float32), values[:, 1].astype(np.float32)
        return values.astype(np.float32), is_click

    def step(self, features, labels, is_click=None):
        labels, is_click = self._targets(labels, is_click)
        size = len(labels)
        if size == 0:
            return 0.0
        users, items, shared = self._shared(features)
        long_logits = self.logits(features)
        long_probabilities = sigmoid(long_logits)
        long_gradient = ((long_probabilities - labels) / size).astype(np.float32)
        grad_user_weights = np.zeros_like(self.user_weights)
        grad_item_weights = np.zeros_like(self.item_weights)
        grad_user_embeddings = np.zeros_like(self.user_embeddings)
        grad_item_embeddings = np.zeros_like(self.item_embeddings)
        np.add.at(grad_user_weights, users, long_gradient)
        np.add.at(grad_item_weights, items, long_gradient)
        shared_gradient = long_gradient[:, None]
        click_loss = 0.0
        grad_click_weights = np.zeros_like(self.click_weights)
        click_bias_gradient = 0.0
        if is_click is not None:
            is_click = np.asarray(is_click, dtype=np.float32).reshape(-1)
            click_logits = self.click_bias + shared @ self.click_weights
            click_probabilities = sigmoid(click_logits)
            click_gradient = (self.auxiliary_click_loss_weight *
                              (click_probabilities - is_click) / size).astype(np.float32)
            shared_gradient += click_gradient[:, None] * self.click_weights
            grad_click_weights = click_gradient @ shared
            click_bias_gradient = float(click_gradient.sum())
            click_loss = float(-np.mean(is_click * np.log(click_probabilities + 1e-9)
                                        + (1 - is_click) * np.log(1 - click_probabilities + 1e-9)))
        np.add.at(grad_user_embeddings, users, shared_gradient * self.item_embeddings[items])
        np.add.at(grad_item_embeddings, items, shared_gradient * self.user_embeddings[users])
        gradients = (grad_user_weights, grad_item_weights, grad_user_embeddings,
                     grad_item_embeddings, grad_click_weights)
        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, first, second, gradient_value in zip(
                self._parameters(), self.first_moment, self.second_moment, gradients):
            gradient_value = gradient_value + self.l2 * parameter
            first *= beta1
            first += (1 - beta1) * gradient_value
            second *= beta2
            second += (1 - beta2) * gradient_value * gradient_value
            first_hat = first / (1 - beta1 ** self.step_number)
            second_hat = second / (1 - beta2 ** self.step_number)
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.bias -= self.learning_rate * long_gradient.sum()
        self.click_bias -= self.learning_rate * click_bias_gradient
        long_loss = float(-np.mean(labels * np.log(long_probabilities + 1e-9)
                                   + (1 - labels) * np.log(1 - long_probabilities + 1e-9)))
        return long_loss + self.auxiliary_click_loss_weight * click_loss

    def predict(self, features, batch_size=200_000):
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        result = np.concatenate([self.logits(features[index:index + batch_size])
                                 for index in range(0, len(features), batch_size)])
        return np.nan_to_num(result, nan=0.0, posinf=30.0, neginf=-30.0).astype(np.float32)

    def state(self):
        return (self.user_weights.copy(), self.item_weights.copy(),
                self.user_embeddings.copy(), self.item_embeddings.copy(),
                self.click_weights.copy(), np.float32(self.bias),
                np.float32(self.click_bias),
                tuple(value.copy() for value in self.first_moment),
                tuple(value.copy() for value in self.second_moment), int(self.step_number))

    def load_state(self, state):
        if len(state) == 8:
            (self.user_weights, self.item_weights, self.user_embeddings,
             self.item_embeddings, self.bias, first_moment, second_moment,
             self.step_number) = state
            self.click_weights = np.zeros(16, dtype=np.float32)
            self.click_bias = np.float32(0.0)
            self.first_moment = [np.asarray(value, dtype=np.float32) for value in first_moment]
            self.second_moment = [np.asarray(value, dtype=np.float32) for value in second_moment]
            self.first_moment.append(np.zeros_like(self.click_weights))
            self.second_moment.append(np.zeros_like(self.click_weights))
        else:
            (self.user_weights, self.item_weights, self.user_embeddings,
             self.item_embeddings, self.click_weights, self.bias, self.click_bias,
             first_moment, second_moment, self.step_number) = state
            self.first_moment = [np.asarray(value, dtype=np.float32) for value in first_moment]
            self.second_moment = [np.asarray(value, dtype=np.float32) for value in second_moment]
        self.user_weights = np.asarray(self.user_weights, dtype=np.float32)
        self.item_weights = np.asarray(self.item_weights, dtype=np.float32)
        self.user_embeddings = np.asarray(self.user_embeddings, dtype=np.float32)
        self.item_embeddings = np.asarray(self.item_embeddings, dtype=np.float32)
        self.click_weights = np.asarray(self.click_weights, dtype=np.float32)
        self.bias = np.float32(self.bias)
        self.click_bias = np.float32(self.click_bias)
        self.step_number = int(self.step_number)
