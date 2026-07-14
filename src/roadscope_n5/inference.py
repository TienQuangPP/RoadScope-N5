"""Run the exported RoadScope-N5 ONNX model on images.

The depth output is relative depth. It is deliberately not converted to metres
or an arbitrary distance unit.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .contracts import CLASS_NAMES, MODEL_OUTPUT_NAMES


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_MODEL = next(
    (
        candidate
        for candidate in (Path.cwd() / "roadscope_n5.onnx", PROJECT_ROOT / "roadscope_n5.onnx")
        if candidate.is_file()
    ),
    PROJECT_ROOT / "roadscope_n5.onnx",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = (
    (255, 170, 40),
    (255, 120, 40),
    (30, 190, 255),
    (0, 140, 255),
    (50, 80, 255),
    (180, 80, 255),
    (255, 80, 80),
    (255, 200, 80),
    (80, 220, 80),
    (220, 220, 80),
)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    output = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output, scale, left, top, resized_width, resized_height


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0.0).prod(axis=1)
    box_area = np.maximum(box[2:] - box[:2], 0.0).prod()
    boxes_area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0).prod(axis=1)
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float, max_det: int) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    # Class offsets make suppression class-aware without one loop per class.
    shifted = boxes + classes[:, None].astype(np.float32) * 4096.0
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(shifted[current], shifted[remaining]) <= threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_detections(
    raw: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(raw)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"Unexpected detection shape: {raw.shape}")
    if prediction.shape[0] == 4 + len(CLASS_NAMES):
        prediction = prediction.T
    if prediction.shape[1] != 4 + len(CLASS_NAMES):
        raise ValueError(f"Unexpected detection shape after transpose: {prediction.shape}")
    class_scores = prediction[:, 4:]
    classes = class_scores.argmax(axis=1).astype(np.int64)
    scores = class_scores[np.arange(class_scores.shape[0]), classes]
    selected = scores >= confidence_threshold
    prediction, scores, classes = prediction[selected], scores[selected], classes[selected]
    if prediction.shape[0] == 0:
        return np.empty((0, 4), np.float32), scores, classes
    centers = prediction[:, :2]
    sizes = prediction[:, 2:4]
    boxes = np.concatenate((centers - sizes / 2.0, centers + sizes / 2.0), axis=1)
    keep = nms(boxes, scores, classes, iou_threshold, max_det)
    return boxes[keep], scores[keep], classes[keep]


def restore_boxes(
    boxes: np.ndarray,
    scale: float,
    left: int,
    top: int,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - left) / scale
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - top) / scale
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, original_width - 1)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, original_height - 1)
    return boxes


def restore_dense(
    dense: np.ndarray,
    input_size: int,
    left: int,
    top: int,
    resized_width: int,
    resized_height: int,
    original_width: int,
    original_height: int,
    interpolation: int,
) -> np.ndarray:
    dense = np.asarray(dense).squeeze().astype(np.float32)
    full = cv2.resize(dense, (input_size, input_size), interpolation=interpolation)
    content = full[top : top + resized_height, left : left + resized_width]
    return cv2.resize(content, (original_width, original_height), interpolation=interpolation)


def create_session(model: Path, provider: str) -> ort.InferenceSession:
    if provider not in {"auto", "cuda", "cpu"}:
        raise ValueError("provider must be one of: auto, cuda, cpu")
    available = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider is unavailable; providers={available}")
    if provider == "auto":
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
    elif provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model), providers=providers)


def draw_result(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    object_depths: np.ndarray,
    road: np.ndarray,
    lane: np.ndarray,
    road_threshold: float,
    lane_threshold: float,
) -> np.ndarray:
    result = image.copy()
    road_mask = road >= road_threshold
    lane_mask = lane >= lane_threshold
    color_layer = result.copy()
    color_layer[road_mask] = (40, 180, 70)
    result = cv2.addWeighted(result, 0.78, color_layer, 0.22, 0.0)
    result[lane_mask] = (0, 220, 255)
    thickness = max(1, round(min(image.shape[:2]) / 360))
    font_scale = max(0.42, min(image.shape[:2]) / 1200.0)
    placed_labels: list[tuple[int, int, int, int]] = []
    for box, score, class_id, object_depth in zip(boxes, scores, classes, object_depths):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = COLORS[int(class_id)]
        label = f"{CLASS_NAMES[int(class_id)]} {score:.2f} rel{object_depth:.2f}"
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_width = text_width + 6
        label_height = text_height + baseline + 6
        label_left = int(np.clip(x1, 0, max(result.shape[1] - label_width, 0)))
        label_positions = (
            y1 - label_height,
            y1,
            y2,
            y2 - label_height,
            y1 - 2 * label_height,
            y2 + label_height,
        )
        label_top = int(np.clip(label_positions[0], 0, max(result.shape[0] - label_height, 0)))
        for position in label_positions:
            position = int(np.clip(position, 0, max(result.shape[0] - label_height, 0)))
            rectangle = (label_left, position, label_left + label_width, position + label_height)
            overlaps = any(
                rectangle[0] < placed[2]
                and rectangle[2] > placed[0]
                and rectangle[1] < placed[3]
                and rectangle[3] > placed[1]
                for placed in placed_labels
            )
            if not overlaps:
                label_top = position
                break
        label_rectangle = (label_left, label_top, label_left + label_width, label_top + label_height)
        placed_labels.append(label_rectangle)
        cv2.rectangle(result, label_rectangle[:2], label_rectangle[2:], color, -1)
        text_y = label_top + text_height + 3
        cv2.putText(
            result,
            label,
            (label_left + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )
    return result


def object_relative_depths(depth: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Estimate robust object depth from its central-lower image region."""
    values = np.zeros(boxes.shape[0], dtype=np.float32)
    height, width = depth.shape
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        # Avoid box borders, which commonly contain background depth.
        sample_x1 = int(np.clip(x1 + 0.25 * box_width, 0, width - 1))
        sample_x2 = int(np.clip(x2 - 0.25 * box_width, sample_x1 + 1, width))
        sample_y1 = int(np.clip(y1 + 0.45 * box_height, 0, height - 1))
        sample_y2 = int(np.clip(y2 - 0.10 * box_height, sample_y1 + 1, height))
        region = depth[sample_y1:sample_y2, sample_x1:sample_x2]
        values[index] = float(np.median(region)) if region.size else 0.0
    return values


def fit_panel(image: np.ndarray, height: int) -> np.ndarray:
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def write_image(path: Path, image: np.ndarray, parameters: list[int] | None = None) -> None:
    if not cv2.imwrite(str(path), image, parameters or []):
        raise OSError(f"Failed to write image: {path}")


def safe_source_label(image_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return image_path.name


def collect_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"No supported image source: {source}")


def reject_duplicate_stems(images: list[Path]) -> None:
    by_stem: dict[str, list[Path]] = {}
    for image in images:
        by_stem.setdefault(image.stem.casefold(), []).append(image)
    duplicates = [paths for paths in by_stem.values() if len(paths) > 1]
    if duplicates:
        names = "; ".join(", ".join(path.as_posix() for path in paths) for paths in duplicates)
        raise RuntimeError(f"Duplicate input stems would overwrite output files: {names}")


@dataclass(frozen=True)
class Prediction:
    """One source-resolution RoadScope-N5 prediction."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    road: np.ndarray
    lane: np.ndarray
    relative_depth: np.ndarray
    quality: float
    object_depths: np.ndarray
    proximity_ranks: np.ndarray
    inference_ms: float


class RoadScopeRuntime:
    """Reusable ONNX Runtime interface for one RoadScope-N5 model."""

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        provider: str = "auto",
        confidence: float = 0.25,
        iou: float = 0.50,
        max_detections: int = 300,
    ) -> None:
        self.model = Path(model)
        if not self.model.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model}")
        if provider not in {"auto", "cuda", "cpu"}:
            raise ValueError("provider must be one of: auto, cuda, cpu")
        for name, value in {"confidence": confidence, "iou": iou}.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if max_detections <= 0:
            raise ValueError("max_detections must be greater than zero")
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections
        self.session = create_session(self.model, provider)
        self.input_metadata = self.session.get_inputs()[0]
        if len(self.input_metadata.shape) != 4 or self.input_metadata.shape[1] != 3:
            raise RuntimeError(f"Unexpected ONNX input shape: {self.input_metadata.shape}")
        try:
            input_height = int(self.input_metadata.shape[-2])
            input_width = int(self.input_metadata.shape[-1])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"ONNX input spatial dimensions must be fixed: {self.input_metadata.shape}"
            ) from error
        if input_height != input_width:
            raise RuntimeError(f"ONNX input must be square: {self.input_metadata.shape}")
        self.input_size = input_width
        output_names = tuple(output.name for output in self.session.get_outputs())
        if output_names != MODEL_OUTPUT_NAMES:
            raise RuntimeError(f"Unexpected ONNX outputs: {output_names}")

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def predict(self, image: np.ndarray) -> Prediction:
        """Run inference and restore all dense outputs to source resolution."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR uint8 array with shape HxWx3")
        if image.dtype != np.uint8:
            raise ValueError(f"image must use uint8 pixels, got {image.dtype}")
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image must not be empty")
        original_height, original_width = image.shape[:2]
        boxed, scale, left, top, resized_width, resized_height = letterbox(image, self.input_size)
        tensor = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        started = time.perf_counter()
        raw_outputs = self.session.run(None, {self.input_metadata.name: tensor})
        inference_ms = (time.perf_counter() - started) * 1000.0
        for name, output in zip(MODEL_OUTPUT_NAMES, raw_outputs):
            if not np.isfinite(output).all():
                raise RuntimeError(f"Model output contains NaN or Inf: {name}")
        raw_detection, raw_road, raw_lane, raw_depth, raw_quality = raw_outputs
        boxes, scores, classes = decode_detections(
            raw_detection,
            self.confidence,
            self.iou,
            self.max_detections,
        )
        boxes = restore_boxes(boxes, scale, left, top, original_width, original_height)
        dense_arguments = (
            self.input_size,
            left,
            top,
            resized_width,
            resized_height,
            original_width,
            original_height,
            cv2.INTER_LINEAR,
        )
        road = restore_dense(raw_road, *dense_arguments)
        lane = restore_dense(raw_lane, *dense_arguments)
        relative_depth = restore_dense(raw_depth, *dense_arguments)
        quality = float(np.asarray(raw_quality).reshape(-1)[0])
        object_depths = object_relative_depths(relative_depth, boxes)
        proximity_order = np.argsort(-object_depths)
        proximity_ranks = np.empty(object_depths.shape[0], dtype=np.int64)
        proximity_ranks[proximity_order] = np.arange(1, object_depths.shape[0] + 1)
        return Prediction(
            boxes=boxes,
            scores=scores,
            classes=classes,
            road=road,
            lane=lane,
            relative_depth=relative_depth,
            quality=quality,
            object_depths=object_depths,
            proximity_ranks=proximity_ranks,
            inference_ms=inference_ms,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path.cwd() / "output")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--road-threshold", type=float, default=0.50)
    parser.add_argument("--lane-threshold", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    thresholds = {
        "road-threshold": args.road_threshold,
        "lane-threshold": args.lane_threshold,
    }
    for name, value in thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be greater than zero")
    images = collect_images(args.source)
    output_root = args.output.resolve()
    images = [image for image in images if not image.resolve().is_relative_to(output_root)]
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError("No images selected")
    reject_duplicate_stems(images)
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = RoadScopeRuntime(
        model=args.model,
        provider=args.provider,
        confidence=args.confidence,
        iou=args.iou,
        max_detections=args.max_det,
    )
    total_inference_ms = 0.0
    processed_count = 0
    skipped_count = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"SKIP unreadable={image_path}", flush=True)
            skipped_count += 1
            continue
        original_height, original_width = image.shape[:2]
        prediction = runtime.predict(image)
        total_inference_ms += prediction.inference_ms
        annotated = draw_result(
            image,
            prediction.boxes,
            prediction.scores,
            prediction.classes,
            prediction.object_depths,
            prediction.road,
            prediction.lane,
            args.road_threshold,
            args.lane_threshold,
        )
        depth_u8 = np.clip(prediction.relative_depth * 255.0, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        panel_height = min(480, original_height)
        comparison = cv2.hconcat(
            (
                fit_panel(image, panel_height),
                fit_panel(annotated, panel_height),
                fit_panel(depth_color, panel_height),
            )
        )
        stem = image_path.stem
        write_image(args.output / f"{stem}_annotated.jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_image(args.output / f"{stem}_relative_depth.png", depth_color)
        write_image(args.output / f"{stem}_comparison.jpg", comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
        detections: list[dict[str, Any]] = []
        for box, score, class_id, object_depth, proximity_rank in zip(
            prediction.boxes,
            prediction.scores,
            prediction.classes,
            prediction.object_depths,
            prediction.proximity_ranks,
        ):
            detections.append({
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "confidence": round(float(score), 6),
                "box_xyxy": [round(float(value), 2) for value in box],
                "relative_depth_median": round(float(object_depth), 6),
                "relative_proximity_rank": int(proximity_rank),
            })
        report = {
            "model": args.model.name,
            "source": safe_source_label(image_path),
            "input_size": runtime.input_size,
            "original_size": [original_width, original_height],
            "provider": runtime.provider,
            "network_inference_ms": round(prediction.inference_ms, 3),
            "quality_heuristic": round(prediction.quality, 6),
            "quality_warning": "Model scene-quality heuristic; not a safety probability.",
            "depth_semantics": (
                "Relative depth only: larger values indicate nearer structure within this image. "
                "Values are not metric distance and are not directly comparable across images."
            ),
            "detections": detections,
        }
        (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        processed_count += 1
        print(
            f"PROCESSED={processed_count}/{len(images)} detections={len(detections)} "
            f"quality={prediction.quality:.3f} network_ms={prediction.inference_ms:.2f} "
            f"image={image_path.name}",
            flush=True,
        )

    if processed_count == 0:
        raise RuntimeError("No readable images were processed")

    print(
        f"DONE images={processed_count} skipped={skipped_count} "
        f"provider={runtime.provider} "
        f"mean_network_ms={total_inference_ms / processed_count:.2f} "
        f"output={args.output.as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
