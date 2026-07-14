"""Convert a RoadScope-N5 FP32 ONNX model to mixed-precision FP16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


EXPECTED_OUTPUTS = [
    "detections",
    "road",
    "lane",
    "relative_depth",
    "confidence",
]


def sort_graph_nodes(model: onnx.ModelProto) -> None:
    """Topologically sort top-level nodes after float16 inserts I/O Casts."""
    available = {value.name for value in model.graph.input}
    available.update(initializer.name for initializer in model.graph.initializer)
    remaining = list(model.graph.node)
    ordered = []
    while remaining:
        next_remaining = []
        progressed = False
        for node in remaining:
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(node.output)
                progressed = True
            else:
                next_remaining.append(node)
        if not progressed:
            unresolved = [node.name or node.op_type for node in next_remaining[:10]]
            raise RuntimeError(f"Cannot topologically sort converted graph: {unresolved}")
        remaining = next_remaining
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Source ONNX model not found: {args.model}")
    if args.output.suffix.lower() != ".onnx":
        raise ValueError("output must use the .onnx extension")
    if args.output.resolve() == args.model.resolve():
        raise ValueError("output must not overwrite the source model")

    model = onnx.load(args.model)
    outputs = [output.name for output in model.graph.output]
    if outputs != EXPECTED_OUTPUTS:
        raise RuntimeError(f"Unexpected source outputs: {outputs}")
    converted = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
        min_positive_val=1e-7,
        max_finite_val=65504.0,
    )
    converted_outputs = [output.name for output in converted.graph.output]
    if converted_outputs != EXPECTED_OUTPUTS:
        raise RuntimeError(f"FP16 outputs changed: {converted_outputs}")
    sort_graph_nodes(converted)
    onnx.checker.check_model(converted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, args.output)
    report = {
        "format": "ONNX mixed precision FP16",
        "io_types": "float32 input and outputs; eligible internal tensors FP16",
        "source_model": args.model.name,
        "output_model": args.output.name,
        "source_bytes": args.model.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "size_ratio": args.output.stat().st_size / args.model.stat().st_size,
        "outputs": converted_outputs,
        "onnx_checker": "PASS",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
