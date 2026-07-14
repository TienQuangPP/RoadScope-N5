"""Compare a reduced-precision ONNX candidate with validated FP32 ONNX."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from roadscope_n5.inference import box_iou, decode_detections
from roadscope_n5.tools.quantize_onnx import IMAGE_SUFFIXES, evenly_spaced, preprocess


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def update_binary(
    accumulator: dict[str, int], reference: np.ndarray, candidate: np.ndarray
) -> None:
    reference = reference >= 0.5
    candidate = candidate >= 0.5
    accumulator["intersection"] += int((reference & candidate).sum())
    accumulator["union"] += int((reference | candidate).sum())
    accumulator["equal"] += int((reference == candidate).sum())
    accumulator["pixels"] += int(reference.size)


def finish_binary(accumulator: dict[str, int]) -> dict[str, float]:
    return {
        "positive_mask_iou": accumulator["intersection"] / max(accumulator["union"], 1),
        "pixel_agreement": accumulator["equal"] / max(accumulator["pixels"], 1),
    }


def match_detections(
    reference_boxes: np.ndarray,
    reference_scores: np.ndarray,
    reference_classes: np.ndarray,
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
) -> tuple[int, list[float], list[float]]:
    used: set[int] = set()
    matched_ious: list[float] = []
    score_differences: list[float] = []
    for box, score, class_id in zip(reference_boxes, reference_scores, reference_classes):
        indices = np.asarray(
            [
                index
                for index, candidate_class in enumerate(candidate_classes)
                if index not in used and int(candidate_class) == int(class_id)
            ],
            dtype=np.int64,
        )
        if indices.size == 0:
            continue
        ious = box_iou(box, candidate_boxes[indices])
        best_local = int(ious.argmax())
        if float(ious[best_local]) < 0.5:
            continue
        best = int(indices[best_local])
        used.add(best)
        matched_ious.append(float(ious[best_local]))
        score_differences.append(abs(float(score) - float(candidate_scores[best])))
    return len(used), matched_ious, score_differences


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run(session: ort.InferenceSession, tensor: np.ndarray) -> tuple[list[np.ndarray], float]:
    started = time.perf_counter()
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    milliseconds = (time.perf_counter() - started) * 1000.0
    return outputs, milliseconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--candidate", "--int8", dest="candidate", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    for name, path in (("fp32", args.fp32), ("candidate", args.candidate)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} ONNX model not found: {path}")
    if args.fp32.resolve() == args.candidate.resolve():
        raise ValueError("candidate must be different from the FP32 reference")
    if not args.images.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {args.images}")
    if args.samples <= 0:
        raise ValueError("samples must be greater than zero")
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        raise ValueError("imgsz must be positive and divisible by 32")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("iou must be in [0, 1]")

    paths = sorted(
        path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No validation images under {args.images}")
    selected = evenly_spaced(paths, min(args.samples, len(paths)), offset=0.75)
    fp32_session = create_session(args.fp32)
    candidate_session = create_session(args.candidate)
    expected_outputs = [
        "detections",
        "road",
        "lane",
        "relative_depth",
        "confidence",
    ]
    for name, session in (("fp32", fp32_session), ("candidate", candidate_session)):
        names = [output.name for output in session.get_outputs()]
        if names != expected_outputs:
            raise RuntimeError(f"Unexpected {name} outputs: {names}")

    warmup = preprocess(selected[0], args.imgsz)
    for _ in range(5):
        fp32_session.run(None, {fp32_session.get_inputs()[0].name: warmup})
        candidate_session.run(None, {candidate_session.get_inputs()[0].name: warmup})

    fp32_times: list[float] = []
    int8_times: list[float] = []
    reference_detection_count = candidate_detection_count = matched_count = 0
    matched_ious: list[float] = []
    matched_score_differences: list[float] = []
    road = {"intersection": 0, "union": 0, "equal": 0, "pixels": 0}
    lane = {"intersection": 0, "union": 0, "equal": 0, "pixels": 0}
    depth_absolute_sum = depth_squared_sum = 0.0
    depth_values = 0
    quality_differences: list[float] = []
    nonfinite_outputs = 0

    for index, path in enumerate(selected, 1):
        tensor = preprocess(path, args.imgsz)
        # Alternate execution order to reduce systematic thermal/order bias.
        if index % 2:
            reference, fp32_ms = run(fp32_session, tensor)
            candidate, int8_ms = run(candidate_session, tensor)
        else:
            candidate, int8_ms = run(candidate_session, tensor)
            reference, fp32_ms = run(fp32_session, tensor)
        fp32_times.append(fp32_ms)
        int8_times.append(int8_ms)
        if any(not np.isfinite(output).all() for output in reference + candidate):
            nonfinite_outputs += 1
            continue

        ref_boxes, ref_scores, ref_classes = decode_detections(
            reference[0], args.confidence, args.iou, 300
        )
        int_boxes, int_scores, int_classes = decode_detections(
            candidate[0], args.confidence, args.iou, 300
        )
        matches, ious, score_differences = match_detections(
            ref_boxes,
            ref_scores,
            ref_classes,
            int_boxes,
            int_scores,
            int_classes,
        )
        reference_detection_count += len(ref_boxes)
        candidate_detection_count += len(int_boxes)
        matched_count += matches
        matched_ious.extend(ious)
        matched_score_differences.extend(score_differences)
        update_binary(road, reference[1], candidate[1])
        update_binary(lane, reference[2], candidate[2])
        depth_difference = candidate[3].astype(np.float32) - reference[3].astype(np.float32)
        depth_absolute_sum += float(np.abs(depth_difference).sum())
        depth_squared_sum += float(np.square(depth_difference).sum())
        depth_values += int(depth_difference.size)
        quality_differences.extend(
            np.abs(candidate[4].astype(np.float32) - reference[4].astype(np.float32))
            .reshape(-1)
            .tolist()
        )
        if index % 25 == 0 or index == len(selected):
            print(f"VALIDATED={index}/{len(selected)}", flush=True)

    detection_precision = matched_count / max(candidate_detection_count, 1)
    detection_recall = matched_count / max(reference_detection_count, 1)
    road_report = finish_binary(road)
    lane_report = finish_binary(lane)
    depth_mae = depth_absolute_sum / max(depth_values, 1)
    quality_mae = float(np.mean(quality_differences)) if quality_differences else 0.0
    checks = {
        "all_outputs_finite": nonfinite_outputs == 0,
        "detection_agreement_precision_gte_0.95": detection_precision >= 0.95,
        "detection_agreement_recall_gte_0.95": detection_recall >= 0.95,
        "matched_box_iou_gte_0.95": bool(matched_ious) and float(np.mean(matched_ious)) >= 0.95,
        "road_mask_iou_gte_0.98": road_report["positive_mask_iou"] >= 0.98,
        "lane_mask_iou_gte_0.95": lane_report["positive_mask_iou"] >= 0.95,
        "relative_depth_mae_lte_0.02": depth_mae <= 0.02,
        "quality_mae_lte_0.02": quality_mae <= 0.02,
    }
    report: dict[str, Any] = {
        "scope": "Reduced-precision candidate agreement with validated FP32 ONNX; not ground-truth accuracy",
        "images_root": args.images.name,
        "images": len(selected),
        "selection": "evenly spaced over sorted paths with offset 0.75",
        "confidence_threshold": args.confidence,
        "iou_threshold": args.iou,
        "nonfinite_images": nonfinite_outputs,
        "detection_agreement": {
            "fp32_detections": reference_detection_count,
            "candidate_detections": candidate_detection_count,
            "matched_iou_0.5": matched_count,
            "precision_vs_fp32": detection_precision,
            "recall_vs_fp32": detection_recall,
            "matched_box_mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "matched_confidence_mae": float(np.mean(matched_score_differences))
            if matched_score_differences
            else 0.0,
        },
        "road_agreement": road_report,
        "lane_agreement": lane_report,
        "relative_depth_agreement": {
            "mae": depth_mae,
            "rmse": float(np.sqrt(depth_squared_sum / max(depth_values, 1))),
        },
        "quality_agreement": {"mae": quality_mae},
        "cpu_network_latency": {
            "fp32_median_ms": percentile(fp32_times, 50),
            "fp32_p95_ms": percentile(fp32_times, 95),
            "candidate_median_ms": percentile(int8_times, 50),
            "candidate_p95_ms": percentile(int8_times, 95),
            "median_speedup": percentile(fp32_times, 50) / max(percentile(int8_times, 50), 1e-9),
        },
        "model_size": {
            "fp32_bytes": args.fp32.stat().st_size,
            "candidate_bytes": args.candidate.stat().st_size,
            "ratio": args.candidate.stat().st_size / args.fp32.stat().st_size,
        },
        "acceptance_checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
