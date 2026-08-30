"""Low-rank user/video factorization model."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        self.user_bias = np.zeros(dimension, dtype=np.float32)
        self.video_bias = np.zeros(dimension, dtype=np.float32)
        rng = np.random.default_rng(0)
        self.user_embeddings = rng.normal(0.0, 0.01, (dimension, 32)).astype(np.float32)
        self.video_embeddings = rng.normal(0.0, 0.01, (dimension, 32)).astype(np.float32)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.user_bias_first = np.zeros_like(self.user_bias)
        self.user_bias_second = np.zeros_like(self.user_bias)
        self.video_bias_first = np.zeros_like(self.video_bias)
        self.video_bias_second = np.zeros_like(self.video_bias)
        self.user_embeddings_first = np.zeros_like(self.user_embeddings)
        self.user_embeddings_second = np.zeros_like(self.user_embeddings)
        self.video_embeddings_first = np.zeros_like(self.video_embeddings)
        self.video_embeddings_second = np.zeros_like(self.video_embeddings)
        self.step_number = 0

    def logits(self, features):
        users = features[:, 0]
        videos = features[:, 1]
        return (self.user_bias[users] + self.video_bias[videos]
                + np.sum(self.user_embeddings[users] * self.video_embeddings[videos], axis=1))

    def _adam(self, parameter, gradient, first, second):
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        gradient = gradient + self.l2 * parameter
        first *= beta1
        first += (1 - beta1) * gradient
        second *= beta2
        second += (1 - beta2) * (gradient * gradient)
        first_hat = first / (1 - beta1 ** self.step_number)
        second_hat = second / (1 - beta2 ** self.step_number)
        parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        return parameter

    def step(self, features, labels):
        size = len(labels)
        users = features[:, 0]
        videos = features[:, 1]
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        grad_user_bias = np.zeros_like(self.user_bias)
        grad_video_bias = np.zeros_like(self.video_bias)
        grad_user_embeddings = np.zeros_like(self.user_embeddings)
        grad_video_embeddings = np.zeros_like(self.video_embeddings)
        np.add.at(grad_user_bias, users, gradient)
        np.add.at(grad_video_bias, videos, gradient)
        np.add.at(grad_user_embeddings, users, gradient[:, None] * self.video_embeddings[videos])
        np.add.at(grad_video_embeddings, videos, gradient[:, None] * self.user_embeddings[users])
        self.step_number += 1
        self.user_bias = self._adam(self.user_bias, grad_user_bias, self.user_bias_first, self.user_bias_second)
        self.video_bias = self._adam(self.video_bias, grad_video_bias, self.video_bias_first, self.video_bias_second)
        self.user_embeddings = self._adam(self.user_embeddings, grad_user_embeddings, self.user_embeddings_first, self.user_embeddings_second)
        self.video_embeddings = self._adam(self.video_embeddings, grad_video_embeddings, self.video_embeddings_first, self.video_embeddings_second)
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
        return (self.user_bias.copy(), self.video_bias.copy(),
                self.user_embeddings.copy(), self.video_embeddings.copy())

    def load_state(self, state):
        (self.user_bias, self.video_bias,
         self.user_embeddings, self.video_embeddings) = state
