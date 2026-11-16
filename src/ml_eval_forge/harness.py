"""Experiment orchestration and evidence validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import DatasetBundle, assert_disjoint, make_dataset
from .metrics import bootstrap_accuracy_ci, classification_metrics, fit_temperature
from .model import LogisticModel, train_logistic


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def validate_contract(bundle: DatasetBundle, model: LogisticModel, metrics: dict[str, object]) -> None:
    """Fail fast when a run violates its evaluation contract."""

    assert_disjoint(bundle)
    test_probabilities = np.asarray(metrics["test_calibrated_probabilities"], dtype=np.float64)
    if test_probabilities.shape != bundle.y_test.shape:
        raise ValueError("test prediction shape does not match the held-out labels")
    if not np.all(np.isfinite(test_probabilities)):
        raise ValueError("test predictions contain non-finite values")
    if np.any((test_probabilities < 0) | (test_probabilities > 1)):
        raise ValueError("test predictions are outside [0, 1]")
    scalar_metrics = metrics["test_calibrated"]
    if not isinstance(scalar_metrics, dict) or not all(
        np.isfinite(float(value)) for value in scalar_metrics.values()
    ):
        raise ValueError("scalar metrics contain a non-finite value")


def run_experiment(output: str | Path, *, seed: int = 2026) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    bundle = make_dataset(seed=seed)
    model, history = train_logistic(bundle.x_train, bundle.y_train, seed=seed)

    valid_logits = model.logits(bundle.x_valid)
    temperature = fit_temperature(valid_logits, bundle.y_valid)
    raw_probabilities = model.predict_proba(bundle.x_test)
    calibrated_probabilities = model.predict_proba(bundle.x_test, temperature=temperature)
    metrics: dict[str, object] = {
        "seed": seed,
        "dataset_fingerprint": bundle.fingerprint,
        "train_examples": int(bundle.y_train.size),
        "valid_examples": int(bundle.y_valid.size),
        "test_examples": int(bundle.y_test.size),
        "temperature": temperature,
        "training": {
            "epochs": len(history),
            "initial_regularized_loss": history[0],
            "final_regularized_loss": history[-1],
        },
        "test_raw": classification_metrics(bundle.y_test, raw_probabilities),
        "test_calibrated": classification_metrics(bundle.y_test, calibrated_probabilities),
        "accuracy_bootstrap_95": bootstrap_accuracy_ci(
            bundle.y_test, calibrated_probabilities, seed=seed
        ),
        "test_calibrated_probabilities": calibrated_probabilities.tolist(),
    }
    validate_contract(bundle, model, metrics)

    _write_json(
        output_path / "data_manifest.json",
        {
            "seed": seed,
            "fingerprint": bundle.fingerprint,
            "sizes": {
                "train": int(bundle.y_train.size),
                "valid": int(bundle.y_valid.size),
                "test": int(bundle.y_test.size),
            },
        },
    )
    _write_json(output_path / "metrics.json", metrics)
    _write_json(output_path / "training_history.json", {"regularized_loss": history})
    model.save(str(output_path / "model.json"))
    return metrics
