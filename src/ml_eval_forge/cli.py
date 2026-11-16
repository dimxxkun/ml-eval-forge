"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .harness import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible ML evaluation lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="train, calibrate, evaluate, and write evidence")
    run.add_argument("--output", default="results/latest")
    run.add_argument("--seed", type=int, default=2026)

    benchmark = subparsers.add_parser("benchmark", help="measure reference/vectorized parity")
    benchmark.add_argument("--output", default="results/latest/benchmark.json")
    benchmark.add_argument("--seed", type=int, default=2026)
    benchmark.add_argument("--rows", type=int, default=5000)

    validate = subparsers.add_parser("validate", help="validate an existing experiment artifact")
    validate.add_argument("--run", dest="run_path", default="results/latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = run_experiment(args.output, seed=args.seed)
        print(json.dumps({"test_raw": report["test_raw"], "test_calibrated": report["test_calibrated"]}, indent=2))
        return 0
    if args.command == "benchmark":
        print(json.dumps(run_benchmark(args.output, seed=args.seed, rows=args.rows), indent=2))
        return 0
    if args.command == "validate":
        required = ["data_manifest.json", "metrics.json", "model.json"]
        missing = [name for name in required if not (Path(args.run_path) / name).exists()]
        if missing:
            raise SystemExit(f"missing artifact(s): {', '.join(missing)}")
        print(f"validated artifact set: {Path(args.run_path).resolve()}")
        return 0
    raise SystemExit("unknown command")
