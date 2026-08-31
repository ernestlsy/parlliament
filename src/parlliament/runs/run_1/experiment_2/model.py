"""Additive ID model with a nonlinear request-context residual."""

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

        self.context_embedding_dimension = 8
        self.hidden_units = 32
        self.context_ranges = None
        self.context_embeddings = None
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_weights = None
        self.output_bias = None
        self.context_first_moment = None
        self.context_second_moment = None
        self.hidden_weights_first_moment = None
        self.hidden_weights_second_moment = None
        self.hidden_bias_first_moment = None
        self.hidden_bias_second_moment = None
        self.output_weights_first_moment = None
        self.output_weights_second_moment = None
        self.output_bias_first_moment = None
        self.output_bias_second_moment = None

    def _ensure_context_parameters(self, features):
        features = np.asarray(features)
        if self.context_embeddings is not None:
            return
        if features.ndim != 2 or features.shape[1] < 5:
            raise ValueError("features must have shape (N, 5)")

        ranges = []
        for column in (2, 3, 4):
            values = np.asarray(features[:, column], dtype=np.int64)
            if values.size:
                lower = int(values.min())
                upper = int(values.max())
            else:
                lower, upper = 0, -1
            ranges.append((lower, upper))
        self.context_ranges = tuple(ranges)

        rng = np.random
        self.context_embeddings = []
        for lower, upper in self.context_ranges:
            size = max(1, upper - lower + 1) + 1
            self.context_embeddings.append(
                rng.normal(0.0, 0.02, (size, self.context_embedding_dimension))
                .astype(np.float32)
            )
        self.hidden_weights = rng.normal(
            0.0, np.sqrt(2.0 / (3 * self.context_embedding_dimension)),
            (3 * self.context_embedding_dimension, self.hidden_units)
        ).astype(np.float32)
        self.hidden_bias = np.zeros(self.hidden_units, dtype=np.float32)
        self.output_weights = rng.normal(
            0.0, np.sqrt(2.0 / self.hidden_units), self.hidden_units
        ).astype(np.float32)
        self.output_bias = np.float32(0.0)

        self.context_first_moment = [np.zeros_like(x) for x in self.context_embeddings]
        self.context_second_moment = [np.zeros_like(x) for x in self.context_embeddings]
        self.hidden_weights_first_moment = np.zeros_like(self.hidden_weights)
        self.hidden_weights_second_moment = np.zeros_like(self.hidden_weights)
        self.hidden_bias_first_moment = np.zeros_like(self.hidden_bias)
        self.hidden_bias_second_moment = np.zeros_like(self.hidden_bias)
        self.output_weights_first_moment = np.zeros_like(self.output_weights)
        self.output_weights_second_moment = np.zeros_like(self.output_weights)
        self.output_bias_first_moment = np.float32(0.0)
        self.output_bias_second_moment = np.float32(0.0)

    def _context_indices(self, features):
        indices = []
        for position, (lower, upper) in enumerate(self.context_ranges):
            values = np.asarray(features[:, position + 2], dtype=np.int64)
            valid = (values >= lower) & (values <= upper)
            mapped = np.full(values.shape, len(self.context_embeddings[position]) - 1,
                             dtype=np.int64)
            mapped[valid] = values[valid] - lower
            indices.append(mapped)
        return indices

    def _residual_forward(self, features):
        self._ensure_context_parameters(features)
        indices = self._context_indices(features)
        embedded = [table[index] for table, index in zip(self.context_embeddings, indices)]
        combined = np.concatenate(embedded, axis=1)
        hidden_pre = combined.dot(self.hidden_weights) + self.hidden_bias
        hidden = np.maximum(hidden_pre, 0.0)
        residual = hidden.dot(self.output_weights) + self.output_bias
        return residual.astype(np.float32), (indices, combined, hidden, hidden_pre)

    def logits(self, features):
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] < 5:
            raise ValueError("features must have shape (N, 5)")
        additive = self.bias + self.weights[features[:, :5]].sum(axis=1)
        residual, _ = self._residual_forward(features)
        return np.asarray(additive + residual, dtype=np.float32).reshape(-1)

    def _adam_update(self, parameter, gradient, first, second, beta1, beta2, epsilon):
        first *= beta1
        first += (1.0 - beta1) * gradient
        second *= beta2
        second += (1.0 - beta2) * (gradient * gradient)
        first_hat = first / (1.0 - beta1 ** self.step_number)
        second_hat = second / (1.0 - beta2 ** self.step_number)
        parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

    def step(self, features, labels):
        features = np.asarray(features)
        labels = np.asarray(labels, dtype=np.float32)
        size = len(labels)
        if size == 0:
            return 0.0
        additive = self.bias + self.weights[features[:, :5]].sum(axis=1)
        residual, cache = self._residual_forward(features)
        logits = np.asarray(additive + residual, dtype=np.float32)
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)

        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, features[:, :5], gradient[:, None])
        grad_weights += self.l2 * self.weights
        grad_bias = np.float32(gradient.sum())

        indices, combined, hidden, hidden_pre = cache
        grad_output_weights = hidden.T.dot(gradient) + self.l2 * self.output_weights
        grad_output_bias = np.float32(gradient.sum())
        hidden_gradient = gradient[:, None] * self.output_weights[None, :]
        hidden_gradient *= (hidden_pre > 0.0)
        grad_hidden_weights = combined.T.dot(hidden_gradient) + self.l2 * self.hidden_weights
        grad_hidden_bias = hidden_gradient.sum(axis=0)
        grad_combined = hidden_gradient.dot(self.hidden_weights.T)
        grad_context = []
        for position, index in enumerate(indices):
            grad = np.zeros_like(self.context_embeddings[position])
            np.add.at(grad, index, grad_combined[:, position * self.context_embedding_dimension:
                                                   (position + 1) * self.context_embedding_dimension])
            grad += self.l2 * self.context_embeddings[position]
            grad_context.append(grad)

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self._adam_update(self.weights, grad_weights, self.first_moment,
                          self.second_moment, beta1, beta2, epsilon)
        self.bias -= self.learning_rate * grad_bias
        for position in range(3):
            self._adam_update(self.context_embeddings[position], grad_context[position],
                              self.context_first_moment[position],
                              self.context_second_moment[position], beta1, beta2, epsilon)
        self._adam_update(self.hidden_weights, grad_hidden_weights,
                          self.hidden_weights_first_moment, self.hidden_weights_second_moment,
                          beta1, beta2, epsilon)
        self._adam_update(self.hidden_bias, grad_hidden_bias,
                          self.hidden_bias_first_moment, self.hidden_bias_second_moment,
                          beta1, beta2, epsilon)
        self._adam_update(self.output_weights, grad_output_weights,
                          self.output_weights_first_moment, self.output_weights_second_moment,
                          beta1, beta2, epsilon)
        output_bias_gradient = np.asarray(grad_output_bias, dtype=np.float32)
        self.output_bias_first_moment = beta1 * self.output_bias_first_moment + (1 - beta1) * output_bias_gradient
        self.output_bias_second_moment = beta2 * self.output_bias_second_moment + (1 - beta2) * (output_bias_gradient ** 2)
        output_first_hat = self.output_bias_first_moment / (1 - beta1 ** self.step_number)
        output_second_hat = self.output_bias_second_moment / (1 - beta2 ** self.step_number)
        self.output_bias -= self.learning_rate * output_first_hat / (np.sqrt(output_second_hat) + epsilon)

        return float(-np.mean(labels * np.log(probabilities + 1e-9)
                             + (1 - labels) * np.log(1 - probabilities + 1e-9)))

    def predict(self, features, batch_size=200_000):
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([
            self.logits(features[index:index + batch_size])
            for index in range(0, len(features), batch_size)
        ]).astype(np.float32, copy=False)

    def state(self):
        return {
            "weights": self.weights.copy(), "bias": np.float32(self.bias),
            "first_moment": self.first_moment.copy(),
            "second_moment": self.second_moment.copy(), "step_number": self.step_number,
            "context_ranges": self.context_ranges,
            "context_embeddings": None if self.context_embeddings is None else [x.copy() for x in self.context_embeddings],
            "hidden_weights": None if self.hidden_weights is None else self.hidden_weights.copy(),
            "hidden_bias": None if self.hidden_bias is None else self.hidden_bias.copy(),
            "output_weights": None if self.output_weights is None else self.output_weights.copy(),
            "output_bias": self.output_bias,
            "context_first_moment": None if self.context_first_moment is None else [x.copy() for x in self.context_first_moment],
            "context_second_moment": None if self.context_second_moment is None else [x.copy() for x in self.context_second_moment],
            "hidden_weights_first_moment": None if self.hidden_weights_first_moment is None else self.hidden_weights_first_moment.copy(),
            "hidden_weights_second_moment": None if self.hidden_weights_second_moment is None else self.hidden_weights_second_moment.copy(),
            "hidden_bias_first_moment": None if self.hidden_bias_first_moment is None else self.hidden_bias_first_moment.copy(),
            "hidden_bias_second_moment": None if self.hidden_bias_second_moment is None else self.hidden_bias_second_moment.copy(),
            "output_weights_first_moment": None if self.output_weights_first_moment is None else self.output_weights_first_moment.copy(),
            "output_weights_second_moment": None if self.output_weights_second_moment is None else self.output_weights_second_moment.copy(),
            "output_bias_first_moment": self.output_bias_first_moment,
            "output_bias_second_moment": self.output_bias_second_moment,
        }

    def load_state(self, state):
        if isinstance(state, tuple):
            self.weights, self.bias = state
            return
        for name in ("weights", "bias", "first_moment", "second_moment", "step_number",
                     "context_ranges", "context_embeddings", "hidden_weights", "hidden_bias",
                     "output_weights", "output_bias", "context_first_moment",
                     "context_second_moment", "hidden_weights_first_moment",
                     "hidden_weights_second_moment", "hidden_bias_first_moment",
                     "hidden_bias_second_moment", "output_weights_first_moment",
                     "output_weights_second_moment", "output_bias_first_moment",
                     "output_bias_second_moment"):
            value = state[name]
            if isinstance(value, list):
                value = [x.copy() for x in value]
            elif isinstance(value, np.ndarray):
                value = value.copy()
            setattr(self, name, value)
