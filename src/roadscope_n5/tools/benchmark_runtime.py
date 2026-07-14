"""Reproducible network-only latency and parameter audit for RoadScope-N5."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from roadscope_n5.training.checkpoint_io import load_checkpoint
from roadscope_n5.training.model import MODEL_VERSION, RoadScopeN5Model


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def benchmark(
    model: RoadScopeN5Model,
    device: torch.device,
    imgsz: int,
    precision: str,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | str]:
    model = model.to(device).eval()
    image = torch.rand(1, 3, imgsz, imgsz, device=device)
    use_amp = device.type == "cuda" and precision == "fp16"
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples: list[float] = []
        if device.type == "cuda":
            events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(iterations)
            ]
            for start_event, end_event in events:
                start_event.record()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    model(image)
                end_event.record()
            torch.cuda.synchronize(device)
            samples = [start.elapsed_time(end) for start, end in events]
        else:
            for _ in range(iterations):
                start = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    model(image)
                samples.append((time.perf_counter() - start) * 1000.0)
    median = statistics.median(samples)
    result: dict[str, float | int | str] = {
        "device": str(device),
        "precision": precision,
        "input": f"1x3x{imgsz}x{imgsz}",
        "warmup": warmup,
        "iterations": iterations,
        "timing_method": "CUDA events" if device.type == "cuda" else "perf_counter wall clock",
        "latency_median_ms": round(median, 3),
        "latency_p95_ms": round(percentile(samples, 0.95), 3),
        "fps_from_median": round(1000.0 / max(median, 1e-9), 2),
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
    return result


def module_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/runtime_benchmark.json"))
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--gpu-iters", type=int, default=200)
    parser.add_argument("--cpu-iters", type=int, default=20)
    parser.add_argument("--skip-cpu", action="store_true")
    args = parser.parse_args()
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        raise ValueError("imgsz must be positive and divisible by 32")
    if args.gpu_iters <= 0 or args.cpu_iters <= 0:
        raise ValueError("gpu-iters and cpu-iters must be greater than zero")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = RoadScopeN5Model(args.weights)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, args.imgsz, args.imgsz))
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        if checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError(
                f"Checkpoint model version {checkpoint.get('model_version')!r} does not match {MODEL_VERSION!r}"
            )
        model.load_state_dict(checkpoint["model"])
    report: dict[str, object] = {
        "scope": "network forward only; excludes decoding, NMS, image I/O, and preprocessing",
        "model_version": MODEL_VERSION,
        "weights_status": (
            "trained checkpoint"
            if args.checkpoint
            else "COCO detector initialization with untrained auxiliary heads"
        ),
        "system": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "parameters": {
            "detector": module_parameters(model.detector),
            "projection": module_parameters(model.project),
            "feature_fusion_weights": model.feature_logits.numel(),
            "fusion": module_parameters(model.fuse),
            "road_head": module_parameters(model.road),
            "lane_head": module_parameters(model.lane),
            "depth_head": module_parameters(model.depth),
            "confidence_head": module_parameters(model.confidence),
            "task_log_variances": model.task_log_vars.numel(),
            "total": module_parameters(model),
        },
        "feature_shapes": [list(feature.shape) for feature in model.features or ()],
        "benchmarks": [],
    }
    if torch.cuda.is_available():
        report["benchmarks"].append(benchmark(model, torch.device("cuda"), args.imgsz, "fp32", 30, args.gpu_iters))
        report["benchmarks"].append(
            benchmark(
                model,
                torch.device("cuda"),
                args.imgsz,
                "fp16_autocast",
                30,
                args.gpu_iters,
            )
        )
        model = model.cpu()
        torch.cuda.empty_cache()
    if not args.skip_cpu:
        report["benchmarks"].append(benchmark(model, torch.device("cpu"), args.imgsz, "fp32", 3, args.cpu_iters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
