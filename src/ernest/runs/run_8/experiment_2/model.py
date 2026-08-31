"""Fresh-start additive ID model with no interaction or baseline-derived architecture."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        self.weights = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.step_number = 0

    def logits(self, features):
        return self.bias + self.weights[features].sum(1)

    def step(self, features, labels):
        logits = self.logits(features)
        row_gradients = np.zeros(len(labels), dtype=np.float32)
        total_pairs = 0
        total_loss = 0.0

        if len(labels):
            _, group_ids = np.unique(features[:, 0], return_inverse=True)
            for group_id in range(group_ids.max() + 1):
                rows = np.flatnonzero(group_ids == group_id)
                positive_rows = rows[labels[rows] == 1]
                negative_rows = rows[labels[rows] == 0]
                if not len(positive_rows) or not len(negative_rows):
                    continue

                positive_logits = logits[positive_rows]
                negative_logits = logits[negative_rows]
                differences = positive_logits[:, None] - negative_logits[None, :]
                absolute_differences = np.abs(differences)
                total_loss += float(np.sum(
                    np.maximum(-differences, 0.0)
                    + np.log1p(np.exp(-absolute_differences)),
                    dtype=np.float64,
                ))

                pair_sigmoid = sigmoid(differences)
                row_gradients[positive_rows] += np.sum(
                    pair_sigmoid - 1.0, axis=1
                )
                row_gradients[negative_rows] += np.sum(
                    1.0 - pair_sigmoid, axis=0
                )
                total_pairs += len(positive_rows) * len(negative_rows)

        grad_weights = np.zeros_like(self.weights)
        if total_pairs:
            row_gradients /= float(total_pairs)
            np.add.at(grad_weights, features, row_gradients[:, None])
        grad_weights += self.l2 * self.weights

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.first_moment *= beta1
        self.first_moment += (1 - beta1) * grad_weights
        self.second_moment *= beta2
        self.second_moment += (1 - beta2) * (grad_weights * grad_weights)
        first_hat = self.first_moment / (1 - beta1 ** self.step_number)
        second_hat = self.second_moment / (1 - beta2 ** self.step_number)
        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

        if total_pairs:
            return float(total_loss / total_pairs)
        return 0.0

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.weights, self.bias = state
