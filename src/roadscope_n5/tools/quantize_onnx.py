"""Calibrate and statically quantize RoadScope-N5 ONNX to QDQ INT8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evenly_spaced(paths: list[Path], count: int, offset: float = 0.0) -> list[Path]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= len(paths):
        return paths
    positions = (np.arange(count, dtype=np.float64) + offset) * len(paths) / count
    indices = np.clip(positions.astype(np.int64), 0, len(paths) - 1)
    return [paths[int(index)] for index in indices]


def preprocess(image_path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unreadable calibration image: {image_path}")
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    boxed = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    boxed[top : top + resized_height, left : left + resized_width] = resized
    return (
        cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / 255.0
    )


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], input_name: str, input_size: int) -> None:
        self.paths = paths
        self.input_name = input_name
        self.input_size = input_size
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.paths):
            return None
        path = self.paths[self.index]
        self.index += 1
        if self.index % 50 == 0 or self.index == len(self.paths):
            print(f"CALIBRATION={self.index}/{len(self.paths)}", flush=True)
        return {self.input_name: preprocess(path, self.input_size)}

    def rewind(self) -> None:
        self.index = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument(
        "--activation-type", choices=("qint8", "quint8"), default="qint8"
    )
    parser.add_argument("--preprocessed-model", type=Path, default=None)
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Keep every ONNX node whose name starts with this prefix in FP32",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Source ONNX model not found: {args.model}")
    if not args.images.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {args.images}")
    if args.samples <= 0:
        raise ValueError("samples must be greater than zero")
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        raise ValueError("imgsz must be positive and divisible by 32")
    if args.output.suffix.lower() != ".onnx":
        raise ValueError("output must use the .onnx extension")
    if args.output.resolve() == args.model.resolve():
        raise ValueError("output must not overwrite the source model")
    if args.preprocessed_model is not None:
        if args.preprocessed_model.suffix.lower() != ".onnx":
            raise ValueError("preprocessed-model must use the .onnx extension")
        if args.preprocessed_model.resolve() in {args.model.resolve(), args.output.resolve()}:
            raise ValueError("preprocessed-model must use a separate path")

    paths = sorted(
        path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No calibration images under {args.images}")
    selected = evenly_spaced(paths, min(args.samples, len(paths)), offset=0.25)
    quantization_input = args.model
    if args.preprocessed_model is not None:
        args.preprocessed_model.parent.mkdir(parents=True, exist_ok=True)
        quant_pre_process(
            str(args.model),
            str(args.preprocessed_model),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            guess_output_rank=True,
            verbose=0,
        )
        quantization_input = args.preprocessed_model
        print(f"PREPROCESSED_MODEL={quantization_input}", flush=True)
    model = onnx.load(quantization_input, load_external_data=False)
    input_name = model.graph.input[0].name
    expected_outputs = [
        "detections",
        "road",
        "lane",
        "relative_depth",
        "confidence",
    ]
    output_names = [output.name for output in model.graph.output]
    if output_names != expected_outputs:
        raise RuntimeError(f"Unexpected model outputs: {output_names}")
    excluded_nodes = sorted(
        node.name
        for node in model.graph.node
        if any(node.name.startswith(prefix) for prefix in args.exclude_prefix)
    )
    if args.exclude_prefix and not excluded_nodes:
        raise RuntimeError(
            f"No nodes matched exclusion prefixes: {args.exclude_prefix}"
        )
    if excluded_nodes:
        print(
            f"EXCLUDED_FROM_QUANTIZATION={len(excluded_nodes)} "
            f"prefixes={args.exclude_prefix}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageCalibrationReader(selected, input_name, args.imgsz)
    quantize_static(
        model_input=str(quantization_input),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=(
            QuantType.QInt8 if args.activation_type == "qint8" else QuantType.QUInt8
        ),
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=excluded_nodes,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": args.activation_type == "qint8",
            "WeightSymmetric": True,
            "DedicatedQDQPair": False,
        },
    )
    quantized = onnx.load(args.output)
    onnx.checker.check_model(quantized)
    quantized_outputs = [output.name for output in quantized.graph.output]
    if quantized_outputs != expected_outputs:
        raise RuntimeError(f"Quantized outputs changed: {quantized_outputs}")
    report = {
        "format": "ONNX QDQ static INT8",
        "calibration_method": "MinMax",
        "activation_type": (
            "QInt8 symmetric"
            if args.activation_type == "qint8"
            else "QUInt8 asymmetric"
        ),
        "weight_type": "QInt8 symmetric per-channel",
        "calibration_root": args.images.name,
        "calibration_images": len(selected),
        "selection": "evenly spaced over sorted validation paths with offset 0.25",
        "input_size": args.imgsz,
        "source_model": args.model.name,
        "quantization_input": quantization_input.name,
        "output_model": args.output.name,
        "source_bytes": args.model.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "size_ratio": args.output.stat().st_size / args.model.stat().st_size,
        "outputs": quantized_outputs,
        "excluded_node_prefixes": args.exclude_prefix,
        "excluded_nodes": len(excluded_nodes),
        "onnx_checker": "PASS",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
