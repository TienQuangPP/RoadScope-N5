"""Benchmark the released RoadScope-N5 ONNX model with a selected provider."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from roadscope_n5.inference import DEFAULT_MODEL, create_session


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("batch and iterations must be positive; warmup must be non-negative")
    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")

    session = create_session(args.model, args.provider)
    model_input = session.get_inputs()[0]
    height, width = int(model_input.shape[-2]), int(model_input.shape[-1])
    tensor = np.zeros((args.batch, 3, height, width), dtype=np.float32)
    for _ in range(args.warmup):
        session.run(None, {model_input.name: tensor})

    timings: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        session.run(None, {model_input.name: tensor})
        timings.append((time.perf_counter() - started) * 1000.0)

    median_ms = statistics.median(timings)
    report = {
        "model": args.model.name,
        "provider": session.get_providers()[0],
        "batch": args.batch,
        "input_shape": [args.batch, 3, height, width],
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "mean_ms": statistics.fmean(timings),
        "median_ms": median_ms,
        "p95_ms": percentile(timings, 95.0),
        "throughput_images_per_second": args.batch * 1000.0 / max(median_ms, 1e-9),
        "model_bytes": args.model.stat().st_size,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
