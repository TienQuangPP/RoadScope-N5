# Contributing

Contributions that improve correctness, portability, documentation, tests, or
measured deployment behavior are welcome.

## Development Setup

```bash
python -m venv .venv
python -m pip install -e ".[train,dev]"
python -m unittest discover -s tests -v
```

Run source compilation before submitting a change:

```bash
python -m compileall -q src run.py
```

## Change Requirements

- Keep code and comments in English.
- Add focused tests for behavior changes.
- Do not commit credentials, local absolute paths, datasets, private images,
  generated reports, or training checkpoints.
- Preserve the public ONNX input/output contract unless the change is clearly
  documented as breaking.
- Report measured results with the exact data split, threshold, hardware,
  runtime, and precision.
- Do not describe dense-target agreement as independent ground-truth accuracy.
- Do not present relative depth as metric distance.
- Do not add autonomous-control or safety-certification claims.

## Pull Requests

Explain the problem, implementation, tests, and compatibility impact. Include
before/after measurements only when collected under the same protocol.

By contributing, you agree that your contribution is licensed under the same
GNU AGPL-3.0 terms as this repository.
