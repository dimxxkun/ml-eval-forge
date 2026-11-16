"""Deterministic synthetic data and split provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    """A fully materialized train/validation/test split."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    fingerprint: str
    seed: int


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def make_dataset(
    n_samples: int = 2400,
    n_features: int = 8,
    seed: int = 2026,
) -> DatasetBundle:
    """Create a reproducible binary task with a nonlinear held-out signal.

    The feature names are intentionally generic, but the workload resembles a
    small multispectral classification task: correlated channels, a linear
    decision component, and a weak interaction term.
    """

    if n_samples < 60 or n_features < 3:
        raise ValueError("n_samples must be >= 60 and n_features must be >= 3")

    rng = np.random.default_rng(seed)
    covariance = np.full((n_features, n_features), 0.12, dtype=np.float64)
    np.fill_diagonal(covariance, 1.0)
    x = rng.multivariate_normal(np.zeros(n_features), covariance, size=n_samples)
    weights = np.linspace(-1.2, 1.0, n_features)
    latent = x @ weights + 0.45 * np.sin(x[:, 0]) - 0.25 * x[:, 1] * x[:, 2]
    probabilities = 1.0 / (1.0 + np.exp(-latent))
    y = rng.binomial(1, probabilities).astype(np.int64)

    order = rng.permutation(n_samples)
    train_end = int(n_samples * 0.60)
    valid_end = int(n_samples * 0.80)
    train_idx, valid_idx, test_idx = (
        order[:train_end],
        order[train_end:valid_end],
        order[valid_end:],
    )
    splits = (x[train_idx], y[train_idx], x[valid_idx], y[valid_idx], x[test_idx], y[test_idx])
    return DatasetBundle(*splits, _fingerprint(*splits), seed)


def assert_disjoint(bundle: DatasetBundle) -> None:
    """Reject accidental row overlap across the three split arrays."""

    split_hashes: list[set[bytes]] = []
    for features in (bundle.x_train, bundle.x_valid, bundle.x_test):
        split_hashes.append({np.ascontiguousarray(row).tobytes() for row in features})
    for first in range(3):
        for second in range(first + 1, 3):
            overlap = split_hashes[first].intersection(split_hashes[second])
            if overlap:
                raise ValueError(f"split overlap detected between split {first} and {second}")
