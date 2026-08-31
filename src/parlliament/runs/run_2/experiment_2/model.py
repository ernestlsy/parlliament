"""Low-rank user-video interaction model with categorical residuals."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        rng = np.random.default_rng(0)
        self.user_embeddings = rng.normal(
            0.0, 0.01, size=(dimension, 16)
        ).astype(np.float32)
        self.video_embeddings = rng.normal(
            0.0, 0.01, size=(dimension, 16)
        ).astype(np.float32)
        self.residual_weights = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2

        self.user_first_moment = np.zeros_like(self.user_embeddings)
        self.user_second_moment = np.zeros_like(self.user_embeddings)
        self.video_first_moment = np.zeros_like(self.video_embeddings)
        self.video_second_moment = np.zeros_like(self.video_embeddings)
        self.residual_first_moment = np.zeros_like(self.residual_weights)
        self.residual_second_moment = np.zeros_like(self.residual_weights)
        self.step_number = 0

    @staticmethod
    def _normalized_interaction(user_vectors, video_vectors):
        epsilon = np.float32(1e-8)
        user_norms = np.sqrt((user_vectors * user_vectors).sum(axis=1) + epsilon)
        video_norms = np.sqrt((video_vectors * video_vectors).sum(axis=1) + epsilon)
        user_hat = user_vectors / user_norms[:, None]
        video_hat = video_vectors / video_norms[:, None]
        dots = (user_hat * video_hat).sum(axis=1)
        return dots, user_hat, video_hat, user_norms, video_norms

    def logits(self, features):
        user_vectors = self.user_embeddings[features[:, 0]]
        video_vectors = self.video_embeddings[features[:, 1]]
        interaction, _, _, _, _ = self._normalized_interaction(
            user_vectors, video_vectors
        )
        residual = self.residual_weights[features[:, 2:5]].sum(axis=1)
        return self.bias + interaction + residual

    def _adam_update(self, parameter, gradient, first_moment, second_moment):
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        first_moment *= beta1
        first_moment += (1 - beta1) * gradient
        second_moment *= beta2
        second_moment += (1 - beta2) * (gradient * gradient)
        first_hat = first_moment / (1 - beta1 ** self.step_number)
        second_hat = second_moment / (1 - beta2 ** self.step_number)
        parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

    def step(self, features, labels):
        size = len(labels)
        user_ids = features[:, 0]
        video_ids = features[:, 1]
        residual_ids = features[:, 2:5]

        user_vectors = self.user_embeddings[user_ids]
        video_vectors = self.video_embeddings[video_ids]
        interaction, user_hat, video_hat, user_norms, video_norms = (
            self._normalized_interaction(user_vectors, video_vectors)
        )
        residual = self.residual_weights[residual_ids].sum(axis=1)
        logits = self.bias + interaction + residual
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)

        grad_user_embeddings = np.zeros_like(self.user_embeddings)
        grad_video_embeddings = np.zeros_like(self.video_embeddings)
        grad_residual_weights = np.zeros_like(self.residual_weights)

        user_gradient = gradient[:, None] * (
            video_hat - interaction[:, None] * user_hat
        ) / user_norms[:, None]
        video_gradient = gradient[:, None] * (
            user_hat - interaction[:, None] * video_hat
        ) / video_norms[:, None]
        np.add.at(grad_user_embeddings, user_ids, user_gradient)
        np.add.at(grad_video_embeddings, video_ids, video_gradient)
        np.add.at(
            grad_residual_weights,
            residual_ids.ravel(),
            np.repeat(gradient, residual_ids.shape[1]),
        )

        grad_user_embeddings += self.l2 * self.user_embeddings
        grad_video_embeddings += self.l2 * self.video_embeddings
        grad_residual_weights += self.l2 * self.residual_weights

        self.step_number += 1
        self._adam_update(
            self.user_embeddings,
            grad_user_embeddings,
            self.user_first_moment,
            self.user_second_moment,
        )
        self._adam_update(
            self.video_embeddings,
            grad_video_embeddings,
            self.video_first_moment,
            self.video_second_moment,
        )
        self._adam_update(
            self.residual_weights,
            grad_residual_weights,
            self.residual_first_moment,
            self.residual_second_moment,
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
        return (
            self.user_embeddings.copy(),
            self.video_embeddings.copy(),
            self.residual_weights.copy(),
            np.float32(self.bias),
        )

    def load_state(self, state):
        user_embeddings, video_embeddings, residual_weights, bias = state
        self.user_embeddings = np.asarray(user_embeddings, dtype=np.float32).copy()
        self.video_embeddings = np.asarray(video_embeddings, dtype=np.float32).copy()
        self.residual_weights = np.asarray(residual_weights, dtype=np.float32).copy()
        self.bias = np.float32(bias)
