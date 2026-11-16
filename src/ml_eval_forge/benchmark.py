"""Reference-vs-vectorized inference benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .data import make_dataset
from .model import sigmoid, train_logistic


def _rowwise_predict(model, features: np.ndarray) -> np.ndarray:
    values = []
    for row in features:
        values.append(float(sigmoid(model.logits(row[None, :])[0])))
    return np.asarray(values, dtype=np.float64)


def run_benchmark(output: str | Path, *, seed: int = 2026, rows: int = 5000) -> dict[str, float | int | str]:
    if rows < 100:
        raise ValueError("rows must be >= 100")
    bundle = make_dataset(seed=seed)
    model, _ = train_logistic(bundle.x_train, bundle.y_train, seed=seed)
    rng = np.random.default_rng(seed + 1)
    features = rng.normal(size=(rows, bundle.x_train.shape[1]))

    vectorized_start = time.perf_counter()
    vectorized = model.predict_proba(features)
    vectorized_seconds = time.perf_counter() - vectorized_start

    reference_start = time.perf_counter()
    reference = _rowwise_predict(model, features)
    reference_seconds = time.perf_counter() - reference_start

    maximum_error = float(np.max(np.abs(vectorized - reference)))
    if maximum_error > 1e-12:
        raise AssertionError(f"inference parity failed: max error={maximum_error}")
    report: dict[str, float | int | str] = {
        "seed": seed,
        "rows": rows,
        "features": int(features.shape[1]),
        "vectorized_seconds": vectorized_seconds,
        "rowwise_seconds": reference_seconds,
        "speedup": reference_seconds / max(vectorized_seconds, 1e-12),
        "max_absolute_error": maximum_error,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report
