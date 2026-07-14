"""Validate numerical parity between a PyTorch checkpoint and its ONNX export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from roadscope_n5.training.checkpoint_io import load_checkpoint
from roadscope_n5.training.export import ExportWrapper
from roadscope_n5.training.model import MODEL_VERSION, RoadScopeN5Model


OUTPUT_NAMES = ("detections", "road", "lane", "relative_depth", "confidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        raise ValueError("imgsz must be positive and divisible by 32")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Checkpoint model version {checkpoint.get('model_version')!r} does not match {MODEL_VERSION!r}"
        )

    model = RoadScopeN5Model(args.weights).eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, args.imgsz, args.imgsz))
    model.load_state_dict(checkpoint["model"])
    wrapper = ExportWrapper(model.eval())

    generator = torch.Generator().manual_seed(391)
    sample = torch.rand(1, 3, args.imgsz, args.imgsz, generator=generator)
    with torch.inference_mode():
        torch_outputs = [value.detach().cpu().numpy() for value in wrapper(sample)]

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(list(OUTPUT_NAMES), {"images": sample.numpy()})

    output_reports = []
    passed = True
    for name, expected, actual in zip(OUTPUT_NAMES, torch_outputs, onnx_outputs):
        if expected.shape != actual.shape:
            output_reports.append({
                "name": name,
                "pytorch_shape": list(expected.shape),
                "onnx_shape": list(actual.shape),
                "passed": False,
                "reason": "shape_mismatch",
            })
            passed = False
            continue
        difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        current_passed = bool(np.allclose(expected, actual, atol=args.atol, rtol=args.rtol))
        output_reports.append({
            "name": name,
            "shape": list(expected.shape),
            "max_abs_error": float(difference.max(initial=0.0)),
            "mean_abs_error": float(difference.mean()),
            "passed": current_passed,
        })
        passed &= current_passed

    report = {
        "model_version": MODEL_VERSION,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "onnx": args.onnx.name,
        "provider": "CPUExecutionProvider",
        "input_shape": list(sample.shape),
        "atol": args.atol,
        "rtol": args.rtol,
        "passed": passed,
        "outputs": output_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
