import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml_eval_forge.harness import run_experiment, validate_run


class ArtifactValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        run_experiment(self.directory, seed=41)
        self.originals = {
            path.name: path.read_bytes() for path in self.directory.iterdir()
        }

    def test_valid_run_is_replayed_without_modifying_artifacts(self):
        self.assertTrue(validate_run(self.directory)["validated"])
        self.assertEqual(
            self.originals,
            {path.name: path.read_bytes() for path in self.directory.iterdir()},
        )

    def test_independent_artifact_mutations_are_rejected(self):
        cases = [
            ("data_manifest.json", ["fingerprint"], "0" * 64),
            ("data_manifest.json", ["sizes", "test"], 1),
            ("data_manifest.json", ["seed"], True),
            ("model.json", ["weights", 0], 100.0),
            ("model.json", ["scale", 0], 0.0),
            ("model.json", ["bias"], "not a number"),
            ("metrics.json", ["test_raw", "accuracy"], 1.0),
            ("metrics.json", ["temperature"], 100.0),
            ("metrics.json", ["test_calibrated_probabilities", 0], 0.123456),
            ("metrics.json", ["test_calibrated_probabilities"], []),
            ("metrics.json", ["accuracy_bootstrap_95"], [1.0, 1.0]),
            ("training_history.json", ["regularized_loss", 0], 0.0),
        ]
        for filename, keys, replacement in cases:
            with self.subTest(filename=filename, field=keys):
                for name, content in self.originals.items():
                    (self.directory / name).write_bytes(content)
                value = json.loads(self.originals[filename])
                target = value
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = replacement
                (self.directory / filename).write_text(
                    json.dumps(value), encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    validate_run(self.directory)

    def test_missing_and_ambiguous_json_fail(self):
        target = self.directory / "model.json"
        target.unlink()
        with self.assertRaisesRegex(ValueError, "model.json"):
            validate_run(self.directory)
        for value in ('{"bias": NaN}', '{"bias": 1, "bias": 2}', "{}"):
            with self.subTest(value=value):
                target.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_run(self.directory)


class ValidationCliTests(unittest.TestCase):
    def test_existing_but_invalid_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "data_manifest.json",
                "model.json",
                "metrics.json",
                "training_history.json",
            ):
                (Path(directory) / name).write_text("not JSON", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "ml_eval_forge", "validate", "--run", directory],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("validated artifact set", result.stdout)


if __name__ == "__main__":
    unittest.main()
