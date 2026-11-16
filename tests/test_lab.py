import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ml_eval_forge.benchmark import run_benchmark
from ml_eval_forge.data import assert_disjoint, make_dataset
from ml_eval_forge.harness import run_experiment
from ml_eval_forge.metrics import classification_metrics, fit_temperature
from ml_eval_forge.model import train_logistic


class EvalForgeTests(unittest.TestCase):
    def test_dataset_is_reproducible_and_disjoint(self):
        first = make_dataset(seed=17)
        second = make_dataset(seed=17)
        self.assertEqual(first.fingerprint, second.fingerprint)
        assert_disjoint(first)

    def test_training_is_reproducible(self):
        bundle = make_dataset(seed=19)
        first, first_history = train_logistic(bundle.x_train, bundle.y_train, seed=19)
        second, second_history = train_logistic(bundle.x_train, bundle.y_train, seed=19)
        np.testing.assert_allclose(first.weights, second.weights)
        np.testing.assert_allclose(first_history, second_history)
        self.assertLess(first_history[-1], first_history[0])

    def test_calibration_metrics_are_finite(self):
        bundle = make_dataset(seed=23)
        model, _ = train_logistic(bundle.x_train, bundle.y_train, seed=23)
        temperature = fit_temperature(model.logits(bundle.x_valid), bundle.y_valid)
        metrics = classification_metrics(
            bundle.y_test, model.predict_proba(bundle.x_test, temperature=temperature)
        )
        self.assertGreater(temperature, 0.0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_experiment_writes_a_reviewable_artifact_set(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_experiment(directory, seed=29)
            self.assertIn("test_calibrated", report)
            self.assertTrue((Path(directory) / "metrics.json").exists())
            loaded = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(loaded["seed"], 29)

    def test_benchmark_enforces_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_benchmark(Path(directory) / "benchmark.json", rows=150)
            self.assertLessEqual(report["max_absolute_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
