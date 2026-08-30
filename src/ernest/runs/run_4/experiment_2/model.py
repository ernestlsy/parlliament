"""Fresh-start additive ID model trained with within-user BPR ranking loss."""

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
        size = len(labels)
        logits = self.logits(features)
        row_gradient = np.zeros(size, dtype=np.float32)
        pair_count = 0
        loss_total = 0.0

        users, inverse = np.unique(features[:, 0], return_inverse=True)
        for user_index in range(len(users)):
            rows = np.flatnonzero(inverse == user_index)
            positive = rows[labels[rows] > 0.5]
            negative = rows[labels[rows] <= 0.5]
            if not len(positive) or not len(negative):
                continue

            pair_count += len(positive) * len(negative)
            negative_scores = logits[negative]
            for positive_row in positive:
                differences = logits[positive_row] - negative_scores
                loss_total += np.logaddexp(0.0, -differences).sum()
                pair_probability = sigmoid(-differences).astype(np.float32)
                row_gradient[positive_row] -= pair_probability.sum()
                np.add.at(row_gradient, negative, pair_probability)

        grad_weights = np.zeros_like(self.weights)
        if pair_count:
            row_gradient /= np.float32(pair_count)
            loss = loss_total / pair_count
        else:
            loss = 0.0
        np.add.at(grad_weights, features, row_gradient[:, None])
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

        return float(loss)

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.weights, self.bias = state
