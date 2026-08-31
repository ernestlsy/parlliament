"""Additive categorical and cyclic-hour model trained with BCE and Adam."""

import numpy as np


CYCLIC_FEATURES = 2


def sigmoid(value):
    """Return numerically stable sigmoid values."""
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


class Model:
    def __init__(self, dimension, learning_rate=0.01, l2=1e-6):
        self.categorical_dimension = int(dimension)
        if self.categorical_dimension < 0:
            raise ValueError("dimension must be non-negative")
        self.weights = np.zeros(
            self.categorical_dimension + CYCLIC_FEATURES, dtype=np.float32
        )
        self.bias = np.float32(0.0)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.step_number = 0

    def _split_features(self, features):
        """Return categorical indices and exactly two cyclic numeric values."""
        if isinstance(features, dict):
            categorical = features.get("categorical")
            if categorical is None:
                categorical = features.get("categorical_features")
            if categorical is None:
                categorical = features.get("indices")

            cyclic = features.get("numeric")
            if cyclic is None:
                cyclic = features.get("cyclic")
            if cyclic is None:
                cyclic = features.get("cyclic_features")
            if cyclic is None and "hour_sin" in features and "hour_cos" in features:
                cyclic = np.column_stack((features["hour_sin"], features["hour_cos"]))

            if categorical is None and cyclic is None:
                categorical = np.empty((0, 0), dtype=np.int64)
                cyclic = np.empty((0, CYCLIC_FEATURES), dtype=np.float32)
            elif categorical is None:
                cyclic_array = np.asarray(cyclic)
                row_count = 1 if cyclic_array.ndim == 1 else len(cyclic_array)
                categorical = np.empty((row_count, 0), dtype=np.int64)
            elif cyclic is None:
                categorical_array = np.asarray(categorical)
                row_count = 1 if categorical_array.ndim == 1 else len(categorical_array)
                cyclic = np.zeros((row_count, CYCLIC_FEATURES), dtype=np.float32)
        elif isinstance(features, (tuple, list)) and len(features) == 2:
            categorical, cyclic = features
        else:
            array = np.asarray(features)
            if array.ndim != 2:
                raise ValueError("features must be a two-dimensional array or a pair")
            if array.shape[1] < CYCLIC_FEATURES:
                categorical = array
                cyclic = np.zeros((len(array), CYCLIC_FEATURES), dtype=np.float32)
            elif np.issubdtype(array.dtype, np.floating):
                categorical = array[:, :-CYCLIC_FEATURES]
                cyclic = array[:, -CYCLIC_FEATURES:]
            else:
                categorical = array
                cyclic = np.zeros((len(array), CYCLIC_FEATURES), dtype=np.float32)

        categorical = np.asarray(categorical)
        if categorical.ndim == 0:
            categorical = categorical.reshape(1, 1)
        elif categorical.ndim == 1:
            categorical = categorical.reshape(-1, 1)
        if categorical.ndim != 2:
            raise ValueError("categorical features must be one- or two-dimensional")
        if categorical.size and self.categorical_dimension:
            categorical = np.asarray(categorical, dtype=np.int64)
            categorical = np.clip(
                categorical, 0, self.categorical_dimension - 1
            )
        else:
            categorical = np.empty((len(categorical), 0), dtype=np.int64)

        cyclic = np.asarray(cyclic, dtype=np.float32)
        if cyclic.ndim == 1:
            if cyclic.size == CYCLIC_FEATURES and len(categorical) == 1:
                cyclic = cyclic.reshape(1, CYCLIC_FEATURES)
            elif cyclic.size % CYCLIC_FEATURES == 0:
                cyclic = cyclic.reshape(-1, CYCLIC_FEATURES)
            else:
                raise ValueError("exactly two cyclic numeric features are required")
        if cyclic.ndim != 2 or cyclic.shape[1] != CYCLIC_FEATURES:
            raise ValueError("exactly two cyclic numeric features are required")
        cyclic = np.nan_to_num(cyclic, nan=0.0, posinf=0.0, neginf=0.0)
        if len(categorical) != len(cyclic):
            raise ValueError("categorical and cyclic feature lengths differ")
        return categorical, cyclic

    def logits(self, features):
        categorical, cyclic = self._split_features(features)
        result = np.full(len(cyclic), self.bias, dtype=np.float32)
        if categorical.shape[1]:
            result += self.weights[categorical].sum(axis=1)
        result += cyclic @ self.weights[self.categorical_dimension:]
        return np.nan_to_num(
            np.asarray(result, dtype=np.float32), nan=0.0, posinf=30.0, neginf=-30.0
        )

    def step(self, features, labels):
        categorical, cyclic = self._split_features(features)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        if len(cyclic) > len(labels):
            categorical = categorical[:len(labels)]
            cyclic = cyclic[:len(labels)]
        if len(labels) != len(cyclic):
            raise ValueError("feature and label lengths differ")
        size = len(labels)
        if size == 0:
            return 0.0

        logits = self.logits((categorical, cyclic))
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size).astype(np.float32)
        grad_weights = np.zeros_like(self.weights)
        if categorical.shape[1]:
            np.add.at(grad_weights, categorical, gradient[:, None])
        grad_weights[self.categorical_dimension:] = gradient @ cyclic
        grad_weights += self.l2 * self.weights

        self.step_number += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.first_moment *= beta1
        self.first_moment += (1.0 - beta1) * grad_weights
        self.second_moment *= beta2
        self.second_moment += (1.0 - beta2) * (grad_weights * grad_weights)
        first_hat = self.first_moment / (1.0 - beta1 ** self.step_number)
        second_hat = self.second_moment / (1.0 - beta2 ** self.step_number)
        self.weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        self.bias -= self.learning_rate * gradient.sum()

        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
        ))

    def predict(self, features, batch_size=200_000):
        categorical, cyclic = self._split_features(features)
        length = len(cyclic)
        if length == 0:
            return np.empty(0, dtype=np.float32)
        outputs = [
            self.logits((categorical[index:index + batch_size], cyclic[index:index + batch_size]))
            for index in range(0, length, batch_size)
        ]
        return np.nan_to_num(
            np.concatenate(outputs).astype(np.float32),
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )

    def _slice_features(self, features, start, stop):
        categorical, cyclic = self._split_features(features)
        return categorical[start:stop], cyclic[start:stop]

    def state(self):
        return self.weights.copy(), np.float32(self.bias)

    def load_state(self, state):
        weights, bias = state
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        expected = self.categorical_dimension + CYCLIC_FEATURES
        if len(weights) == self.categorical_dimension:
            expanded = np.zeros(expected, dtype=np.float32)
            expanded[:self.categorical_dimension] = weights
            weights = expanded
        if len(weights) != expected:
            raise ValueError("state has an incompatible weight dimension")
        self.weights = weights.copy()
        self.bias = np.float32(bias)
