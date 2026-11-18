"""Experiment orchestration and evidence validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .data import assert_disjoint, make_dataset
from .metrics import bootstrap_accuracy_ci, classification_metrics, fit_temperature
from .model import train_logistic


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def experiment_artifacts(seed: int) -> dict[str, object]:
    """The fixed experiment owns generation, training and report semantics."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    bundle = make_dataset(seed=seed)
    assert_disjoint(bundle)
    model, history = train_logistic(bundle.x_train, bundle.y_train, seed=seed)

    valid_logits = model.logits(bundle.x_valid)
    temperature = fit_temperature(valid_logits, bundle.y_valid)
    raw_probabilities = model.predict_proba(bundle.x_test)
    calibrated_probabilities = model.predict_proba(
        bundle.x_test, temperature=temperature
    )
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
        "test_calibrated": classification_metrics(
            bundle.y_test, calibrated_probabilities
        ),
        "accuracy_bootstrap_95": bootstrap_accuracy_ci(
            bundle.y_test, calibrated_probabilities, seed=seed
        ),
        "test_calibrated_probabilities": calibrated_probabilities.tolist(),
    }
    return {
        "data_manifest.json": {
            "seed": seed,
            "fingerprint": bundle.fingerprint,
            "sizes": {
                "train": int(bundle.y_train.size),
                "valid": int(bundle.y_valid.size),
                "test": int(bundle.y_test.size),
            },
        },
        "metrics.json": metrics,
        "training_history.json": {"regularized_loss": history},
        "model.json": model.to_dict(),
    }


def run_experiment(output: str | Path, *, seed: int = 2026) -> dict[str, object]:
    artifacts = experiment_artifacts(seed)
    for filename, payload in artifacts.items():
        _write_json(Path(output) / filename, payload)
    return artifacts["metrics.json"]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _compare(actual, expected, location):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{location}: missing or unexpected fields")
        for key, value in expected.items():
            _compare(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{location}: incorrect array shape")
        for index, value in enumerate(expected):
            _compare(actual[index], value, f"{location}[{index}]")
    elif isinstance(expected, float):
        if (
            type(actual) not in (int, float)
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
        ):
            raise ValueError(f"{location}: differs from recomputed value")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{location}: differs from recomputed value")


def validate_run(directory: str | Path) -> dict[str, object]:
    """Read-only replay of this repository's fixed synthetic experiment.

    This checks consistency, not authorship or independent scientific validity.
    Run in the same numerical environment; RNG/BLAS changes can change fingerprints.
    """
    directory = Path(directory)
    payloads = {}
    for filename in (
        "data_manifest.json",
        "model.json",
        "metrics.json",
        "training_history.json",
    ):
        try:
            payloads[filename] = json.loads(
                (directory / filename).read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"{filename}: {error}") from error
    manifest = payloads["data_manifest.json"]
    if not isinstance(manifest, dict):
        raise ValueError("data_manifest.json must contain an object")
    expected = experiment_artifacts(manifest.get("seed"))
    for filename, reference in expected.items():
        _compare(payloads[filename], reference, filename)
    return {"validated": True, "seed": manifest["seed"], "artifacts": list(expected)}
