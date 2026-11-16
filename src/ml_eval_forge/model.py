"""A small deterministic logistic model implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    logits = np.asarray(logits, dtype=np.float64)
    positive = logits >= 0
    result = np.empty_like(logits)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)))


@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    seed: int

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) - self.mean) / self.scale

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self._standardize(features) @ self.weights + self.bias

    def predict_proba(self, features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if temperature <= 0 or not np.isfinite(temperature):
            raise ValueError("temperature must be finite and positive")
        return sigmoid(self.logits(features) / temperature)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "seed": self.seed,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def train_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 2026,
    epochs: int = 220,
    learning_rate: float = 0.18,
    l2: float = 1e-2,
) -> tuple[LogisticModel, list[float]]:
    """Fit a regularized logistic model with full-batch gradient descent."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be 2-D and labels must match its row count")
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("epochs must be positive; learning_rate and l2 must be valid")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (x - mean) / scale
    weights = np.zeros(standardized.shape[1], dtype=np.float64)
    bias = 0.0
    history: list[float] = []
    for _ in range(epochs):
        probabilities = sigmoid(standardized @ weights + bias)
        error = probabilities - y
        weights -= learning_rate * (standardized.T @ error / x.shape[0] + l2 * weights)
        bias -= learning_rate * float(error.mean())
        history.append(log_loss(y, probabilities) + 0.5 * l2 * float(weights @ weights))

    model = LogisticModel(mean, scale, weights, bias, seed)
    if not np.all(np.isfinite(history)):
        raise FloatingPointError("training produced a non-finite loss")
    return model, history
