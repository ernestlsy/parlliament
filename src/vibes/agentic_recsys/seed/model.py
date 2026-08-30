"""Fresh-start additive ID model with no interaction or baseline-derived architecture."""

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class Model:
    """Additive per-field weights over encoded ID features.

    Parameters are initialised from a small random normal, never from zeros. Zeros are
    harmless for this purely additive scoring function, but any descendant that adds a
    multiplicative interaction term inherits a dead gradient from them: with an
    interaction u . v, grad(u) is proportional to v and grad(v) is proportional to u, so
    two zero blocks stay zero forever and the interaction never learns. Descendants must
    keep a non-zero init for every block that appears in a product.
    """

    def __init__(self, dimension, learning_rate=0.01, l2=1e-6, seed=0, init_scale=0.01):
        generator = np.random.default_rng(seed)
        self.weights = (
            generator.normal(0.0, init_scale, size=dimension).astype(np.float32)
        )
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.init_scale = init_scale
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.step_number = 0

    def parameter_blocks(self):
        """Every trainable array, by name, for the dead-gradient check in train.py.

        A descendant that adds parameters (embeddings, interaction factors, per-task
        heads) must list them here. Training fails loudly if a listed block does not
        change, so an inert term cannot silently report its parent's score.
        Optimizer state (moments, step counters) is not a parameter and stays out.
        """
        return {"weights": self.weights, "bias": np.asarray(self.bias)}

    def logits(self, features):
        return self.bias + self.weights[features].sum(1)

    def step(self, features, labels):
        size = len(labels)
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, features, gradient[:, None])
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
        self.bias -= self.learning_rate * gradient.sum()
        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1 - labels) * np.log(1 - probabilities + 1e-9)
        ))

    def predict(self, features, batch_size=200_000):
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ])

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        self.weights, self.bias = state
