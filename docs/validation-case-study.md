# A validator that accepted invalid evidence

## Observed failure

Before this change, the public `validate` command checked only whether three artifact paths existed. A regression test created four files containing the text `not JSON`, invoked `python -m ml_eval_forge validate --run <directory>`, and observed exit code 0. The new rejection assertion failed against that implementation. A successful exit therefore did not establish that the experiment had produced a usable result.

## Fix and ownership

The experiment harness now owns the complete four-file artifact contract. Generation and validation use the same experiment definition instead of keeping a second, incomplete list of plausibility checks in the CLI. Validation strictly parses JSON, reconstructs the seeded dataset and train-only normalization, retrains the fixed logistic model, fits temperature on validation data, and recomputes held-out predictions and reports. It compares the complete saved structure with that replay and never rewrites the evidence under inspection.

Sharing the experiment definition prevents producer/consumer drift. It also means this check can agree with a bug shared by the producer and replay; independent metric and loss checks still matter. The validator establishes internal consistency, not independent scientific validity or authorship.

Inspecting the producing model code also revealed inconsistent loss accounting: data loss used predictions from before the parameter update while regularization used weights after it. The history now evaluates both terms at the updated model, with the stable logistic objective `logaddexp(0, z) - y*z`. A one-step test independently computes cross-entropy plus regularization from the returned model and checks the recorded objective. This correction changes reported loss history; it does not change the optimizer update rule.

## Verification performed

- Eleven unittest tests passed locally, including the public CLI rejection regression and a successful read-only replay.
- Negative controls independently alter a fingerprint, split size, seed type, model weight, scale, bias type, accuracy, temperature, probability, prediction-array length, bootstrap interval, and loss-history value. Every mutation must fail.
- Missing files, non-finite JSON numbers, duplicate object keys, and missing model fields are rejected.
- Invalid training data is rejected. Repeated training remains deterministic in the tested environment.
- The regenerated `results/latest` artifact set passes the public validation command.

Run the evidence yourself:

```bash
python -m unittest discover -s tests -v
python -m ml_eval_forge run --output runs/review
python -m ml_eval_forge validate --run runs/review
```

The tests and commands ran on Python 3.12 with NumPy 2.5.5 on Windows. CI repeats generation and validation within its own environment; committed byte fingerprints are not advertised as portable across different numerical builds.

## Scope and development workflow

This repair was developed with a coding agent: inspect the CLI and producer, demonstrate a false pass through the public command, move validation into the owning harness, add negative controls, inspect numerical accounting, and rerun the complete experiment. The repository retains the code and evidence so a reviewer can inspect the work. These are new portfolio exercises, not claims of previous employment or production deployment.
