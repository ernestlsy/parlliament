"""User-video latent-factor model with additive request-context terms."""

import numpy as np


EMBEDDING_DIMENSION = 16


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.l2 = l2

        values = (np.arange(dimension * EMBEDDING_DIMENSION, dtype=np.float32) % 31) + 1.0
        self.user_embeddings = (values.reshape(dimension, EMBEDDING_DIMENSION) * 1e-4).astype(np.float32)
        video_values = ((np.arange(dimension * EMBEDDING_DIMENSION, dtype=np.float32) + 11.0) % 37) + 1.0
        self.video_embeddings = (video_values.reshape(dimension, EMBEDDING_DIMENSION) * 1e-4).astype(np.float32)
        self.context_weights = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)

        self.user_first_moment = np.zeros_like(self.user_embeddings)
        self.user_second_moment = np.zeros_like(self.user_embeddings)
        self.video_first_moment = np.zeros_like(self.video_embeddings)
        self.video_second_moment = np.zeros_like(self.video_embeddings)
        self.context_first_moment = np.zeros_like(self.context_weights)
        self.context_second_moment = np.zeros_like(self.context_weights)
        self.step_number = 0

    def logits(self, features):
        users = features[:, 0]
        videos = features[:, 1]
        contexts = features[:, 2:]
        interaction = np.sum(
            self.user_embeddings[users] * self.video_embeddings[videos], axis=1
        )
        return self.bias + interaction + self.context_weights[contexts].sum(axis=1)

    def step(self, features, labels):
        size = len(labels)
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)

        users = features[:, 0]
        videos = features[:, 1]
        contexts = features[:, 2:]
        user_vectors = self.user_embeddings[users]
        video_vectors = self.video_embeddings[videos]

        grad_users = np.zeros_like(self.user_embeddings)
        grad_videos = np.zeros_like(self.video_embeddings)
        grad_context = np.zeros_like(self.context_weights)
        np.add.at(grad_users, users, gradient[:, None] * video_vectors)
        np.add.at(grad_videos, videos, gradient[:, None] * user_vectors)
        np.add.at(grad_context, contexts, np.broadcast_to(gradient[:, None], contexts.shape))

        grad_users += self.l2 * self.user_embeddings
        grad_videos += self.l2 * self.video_embeddings
        grad_context += self.l2 * self.context_weights

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        self.user_first_moment *= beta1
        self.user_first_moment += (1 - beta1) * grad_users
        self.user_second_moment *= beta2
        self.user_second_moment += (1 - beta2) * (grad_users * grad_users)

        self.video_first_moment *= beta1
        self.video_first_moment += (1 - beta1) * grad_videos
        self.video_second_moment *= beta2
        self.video_second_moment += (1 - beta2) * (grad_videos * grad_videos)

        self.context_first_moment *= beta1
        self.context_first_moment += (1 - beta1) * grad_context
        self.context_second_moment *= beta2
        self.context_second_moment += (1 - beta2) * (grad_context * grad_context)

        first_scale = 1 - beta1 ** self.step_number
        second_scale = 1 - beta2 ** self.step_number

        user_first_hat = self.user_first_moment / first_scale
        user_second_hat = self.user_second_moment / second_scale
        self.user_embeddings -= self.learning_rate * user_first_hat / (
            np.sqrt(user_second_hat) + epsilon
        )

        video_first_hat = self.video_first_moment / first_scale
        video_second_hat = self.video_second_moment / second_scale
        self.video_embeddings -= self.learning_rate * video_first_hat / (
            np.sqrt(video_second_hat) + epsilon
        )

        context_first_hat = self.context_first_moment / first_scale
        context_second_hat = self.context_second_moment / second_scale
        self.context_weights -= self.learning_rate * context_first_hat / (
            np.sqrt(context_second_hat) + epsilon
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
        ]).astype(np.float32, copy=False)

    def state(self):
        return (
            np.float32(self.bias),
            self.user_embeddings.copy(),
            self.video_embeddings.copy(),
            self.context_weights.copy(),
        )

    def load_state(self, state):
        self.bias = np.float32(state[0])
        self.user_embeddings = np.asarray(state[1], dtype=np.float32).copy()
        self.video_embeddings = np.asarray(state[2], dtype=np.float32).copy()
        self.context_weights = np.asarray(state[3], dtype=np.float32).copy()
