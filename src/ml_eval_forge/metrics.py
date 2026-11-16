"""Metrics and validation-only temperature calibration."""

from __future__ import annotations

import numpy as np

from .model import log_loss, sigmoid


def _check_inputs(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError("y_true and probabilities must be equally sized vectors")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return y, p


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    y, p = _check_inputs(y_true, probabilities)
    if bins < 2:
        raise ValueError("bins must be >= 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        mask = (p >= left) & ((p < right) if index < bins - 1 else (p <= right))
        if np.any(mask):
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y, p = _check_inputs(y_true, probabilities)
    predictions = (p >= 0.5).astype(np.int64)
    true_positive = int(np.sum((predictions == 1) & (y == 1)))
    false_positive = int(np.sum((predictions == 1) & (y == 0)))
    false_negative = int(np.sum((predictions == 0) & (y == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float(np.mean(predictions == y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "log_loss": log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
    }


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a scalar temperature using validation logits only."""

    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    if logits.ndim != 1 or y.shape != logits.shape:
        raise ValueError("logits and labels must be equally sized vectors")
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
    losses = np.array([log_loss(y, sigmoid(logits / value)) for value in candidates])
    return float(candidates[int(np.argmin(losses))])


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int = 2026,
    samples: int = 400,
) -> tuple[float, float]:
    y, p = _check_inputs(y_true, probabilities)
    if samples < 20:
        raise ValueError("samples must be >= 20")
    rng = np.random.default_rng(seed)
    predictions = (p >= 0.5).astype(np.int64)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, y.size, size=y.size)
        values[index] = np.mean(predictions[selected] == y[selected])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
