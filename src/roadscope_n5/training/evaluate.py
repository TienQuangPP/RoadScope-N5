"""Evaluate RoadScope-N5 detection and dense-target agreement.

Detection metrics use official BDD100K YOLO boxes. Dense metrics measure
agreement with supplied dense targets and must not be described as ground-truth
road, lane, depth, uncertainty, or safety accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression

from roadscope_n5.contracts import CLASS_NAMES as CLASS_NAME_SEQUENCE

from .checkpoint_io import load_checkpoint
from .dataset import BDDMultiTaskDataset, collate_multitask
from .model import MODEL_VERSION, RoadScopeN5Model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASS_NAMES = dict(enumerate(CLASS_NAME_SEQUENCE))
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def match_predictions(prediction: Tensor, target_classes: Tensor, target_boxes: Tensor) -> np.ndarray:
    """Match predictions to labels independently at each COCO IoU threshold."""
    correct = np.zeros((prediction.shape[0], len(IOU_THRESHOLDS)), dtype=bool)
    if prediction.shape[0] == 0 or target_classes.shape[0] == 0:
        return correct
    iou = box_iou(target_boxes, prediction[:, :4])
    class_match = target_classes[:, None] == prediction[:, 5]
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS.to(iou.device)):
        label_indices, prediction_indices = torch.where((iou >= threshold) & class_match)
        if label_indices.numel() == 0:
            continue
        matches = torch.stack(
            (label_indices, prediction_indices, iou[label_indices, prediction_indices]), dim=1
        ).cpu().numpy()
        if matches.shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def binary_accumulator() -> dict[str, float]:
    return {"intersection": 0.0, "union": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0}


def update_binary(accumulator: dict[str, float], prediction: Tensor, target: Tensor, valid: Tensor) -> None:
    prediction = prediction & valid
    target = target & valid
    true_positive = (prediction & target).sum().item()
    accumulator["intersection"] += true_positive
    accumulator["union"] += (prediction | target).sum().item()
    accumulator["tp"] += true_positive
    accumulator["fp"] += (prediction & ~target).sum().item()
    accumulator["fn"] += (~prediction & target & valid).sum().item()


def finish_binary(accumulator: dict[str, float]) -> dict[str, float]:
    epsilon = 1e-12
    tp, fp, fn = accumulator["tp"], accumulator["fp"], accumulator["fn"]
    return {
        "iou": accumulator["intersection"] / max(accumulator["union"], epsilon),
        "f1": 2.0 * tp / max(2.0 * tp + fp + fn, epsilon),
        "precision": tp / max(tp + fp, epsilon),
        "recall": tp / max(tp + fn, epsilon),
    }


def quality_ece(
    prediction: np.ndarray,
    target: np.ndarray,
    bins: int = 15,
) -> tuple[float, list[dict[str, float | int]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(prediction), 1)
    ece = 0.0
    report: list[dict[str, float | int]] = []
    for index in range(bins):
        selected = (prediction >= edges[index]) & (
            prediction <= edges[index + 1] if index == bins - 1 else prediction < edges[index + 1]
        )
        count = int(selected.sum())
        if count == 0:
            continue
        predicted_mean = float(prediction[selected].mean())
        target_mean = float(target[selected].mean())
        gap = abs(predicted_mean - target_mean)
        ece += count / total * gap
        report.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "predicted_quality": predicted_mean,
            "target_quality": target_mean,
            "absolute_gap": gap,
        })
    return ece, report


def detection_report(
    true_positives: list[np.ndarray],
    confidences: list[np.ndarray],
    predicted_classes: list[np.ndarray],
    target_classes: list[np.ndarray],
) -> dict[str, Any]:
    tp = np.concatenate(true_positives, axis=0) if true_positives else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(confidences, axis=0) if confidences else np.zeros(0, dtype=np.float32)
    pred_cls = np.concatenate(predicted_classes, axis=0) if predicted_classes else np.zeros(0, dtype=np.float32)
    target_cls = np.concatenate(target_classes, axis=0) if target_classes else np.zeros(0, dtype=np.float32)
    if target_cls.size == 0:
        raise RuntimeError("No detection labels were found during evaluation")
    if conf.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "predictions": 0,
            "targets": int(target_cls.size),
            "per_class": [],
        }
    _, _, precision, recall, f1, ap, classes, *_ = ap_per_class(
        tp, conf, pred_cls, target_cls, names=CLASS_NAMES
    )
    per_class = []
    for index, class_id in enumerate(classes):
        per_class.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "targets": int((target_cls == class_id).sum()),
            "predictions": int((pred_cls == class_id).sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "ap50": float(ap[index, 0]),
            "ap50_95": float(ap[index].mean()),
        })
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "map50": float(ap[:, 0].mean()),
        "map50_95": float(ap.mean()),
        "predictions": int(conf.size),
        "targets": int(target_cls.size),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolov8n.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--ordinal-pairs", type=int, default=1024)
    args = parser.parse_args()

    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        raise ValueError("imgsz must be positive and divisible by 32")
    if args.batch <= 0:
        raise ValueError("batch must be greater than zero")
    if args.workers < 0:
        raise ValueError("workers must be zero or greater")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be greater than zero")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence-threshold must be in [0, 1]")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("nms-iou must be in [0, 1]")
    if args.ordinal_pairs <= 0:
        raise ValueError("ordinal-pairs must be greater than zero")

    device = torch.device(args.device)
    dataset = BDDMultiTaskDataset(
        args.images, args.labels, args.targets, args.imgsz, augment=False, limit=args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_multitask,
    )
    model = RoadScopeN5Model(args.weights).to(device)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Checkpoint model version {checkpoint.get('model_version')!r} does not match {MODEL_VERSION!r}"
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    detection_tp: list[np.ndarray] = []
    detection_confidence: list[np.ndarray] = []
    detection_prediction_classes: list[np.ndarray] = []
    detection_target_classes: list[np.ndarray] = []
    road_metrics, lane_metrics = binary_accumulator(), binary_accumulator()
    depth_absolute_sum = depth_squared_sum = depth_edge_sum = 0.0
    depth_pixels = depth_edges = ordinal_correct = ordinal_total = 0
    quality_predictions: list[np.ndarray] = []
    quality_targets: list[np.ndarray] = []
    generator = torch.Generator(device=device).manual_seed(391)

    with torch.inference_mode():
        for step, raw_batch in enumerate(loader, 1):
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["img"])
            raw_detection = output["detection"]
            if isinstance(raw_detection, (tuple, list)):
                raw_detection = raw_detection[0]
            predictions = non_max_suppression(
                raw_detection.float(),
                conf_thres=args.confidence_threshold,
                iou_thres=args.nms_iou,
                nc=len(CLASS_NAMES),
                multi_label=True,
                max_det=300,
            )
            image_size = batch["img"].shape[-1]
            for image_index, prediction in enumerate(predictions):
                selected = batch["batch_idx"] == image_index
                target_cls = batch["cls"][selected, 0]
                target_boxes = ops.xywh2xyxy(batch["bboxes"][selected]) * image_size
                detection_tp.append(match_predictions(prediction, target_cls, target_boxes))
                detection_confidence.append(prediction[:, 4].float().cpu().numpy())
                detection_prediction_classes.append(prediction[:, 5].float().cpu().numpy())
                detection_target_classes.append(target_cls.float().cpu().numpy())

            content_valid = batch["content_valid"] > 0.5
            road_prediction = output["road"].sigmoid() > 0.5
            lane_prediction = output["lane"].sigmoid() > 0.5
            update_binary(road_metrics, road_prediction, batch["road"] > 0.5, content_valid)
            update_binary(lane_metrics, lane_prediction, batch["lane"] > 0.5, content_valid)

            depth_prediction = output["depth"].float()
            depth_target = batch["depth"]
            depth_valid = batch["depth_valid"] > 0.5
            difference = depth_prediction - depth_target
            depth_absolute_sum += difference[depth_valid].abs().sum().item()
            depth_squared_sum += difference[depth_valid].square().sum().item()
            depth_pixels += int(depth_valid.sum().item())
            for axis in (-1, -2):
                if axis == -1:
                    pred_edge = depth_prediction[..., 1:] - depth_prediction[..., :-1]
                    target_edge = depth_target[..., 1:] - depth_target[..., :-1]
                    valid_edge = depth_valid[..., 1:] & depth_valid[..., :-1]
                else:
                    pred_edge = depth_prediction[..., 1:, :] - depth_prediction[..., :-1, :]
                    target_edge = depth_target[..., 1:, :] - depth_target[..., :-1, :]
                    valid_edge = depth_valid[..., 1:, :] & depth_valid[..., :-1, :]
                depth_edge_sum += (pred_edge - target_edge)[valid_edge].abs().sum().item()
                depth_edges += int(valid_edge.sum().item())
            for image_index in range(depth_prediction.shape[0]):
                valid_indices = torch.where(depth_valid[image_index].flatten())[0]
                if valid_indices.numel() < 2:
                    continue
                first = valid_indices[
                    torch.randint(
                        valid_indices.numel(),
                        (args.ordinal_pairs,),
                        generator=generator,
                        device=device,
                    )
                ]
                second = valid_indices[
                    torch.randint(
                        valid_indices.numel(),
                        (args.ordinal_pairs,),
                        generator=generator,
                        device=device,
                    )
                ]
                target_delta = depth_target[image_index].flatten()[first] - depth_target[image_index].flatten()[second]
                selected_pairs = target_delta.abs() >= 0.05
                if selected_pairs.any():
                    prediction_flat = depth_prediction[image_index].flatten()
                    prediction_delta = prediction_flat[first] - prediction_flat[second]
                    agreement = (
                        prediction_delta[selected_pairs].sign()
                        == target_delta[selected_pairs].sign()
                    )
                    ordinal_correct += int(agreement.sum().item())
                    ordinal_total += int(selected_pairs.sum().item())

            predicted_quality = output["confidence_logit"].sigmoid().float()
            target_quality = batch["confidence_target"].float()
            quality_predictions.append(predicted_quality.cpu().numpy())
            quality_targets.append(target_quality.cpu().numpy())
            if step % 50 == 0 or step == len(loader):
                print(f"EVALUATED={min(step * args.batch, len(dataset))}/{len(dataset)}", flush=True)

    predicted_quality = np.concatenate(quality_predictions)
    target_quality = np.concatenate(quality_targets)
    ece, reliability_bins = quality_ece(predicted_quality, target_quality)
    report = {
        "scope": {
            "detection": "official BDD100K YOLO labels",
            "dense": "agreement with supplied dense validation targets; not independent ground-truth accuracy",
            "confidence": "agreement with the supplied scene-quality target; not safety probability",
        },
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "images": len(dataset),
        "detection": detection_report(
            detection_tp,
            detection_confidence,
            detection_prediction_classes,
            detection_target_classes,
        ),
        "road_target_agreement": finish_binary(road_metrics),
        "lane_target_agreement": finish_binary(lane_metrics),
        "relative_depth_target_agreement": {
            "mae": depth_absolute_sum / max(depth_pixels, 1),
            "rmse": float(np.sqrt(depth_squared_sum / max(depth_pixels, 1))),
            "edge_mae": depth_edge_sum / max(depth_edges, 1),
            "ordinal_accuracy_margin_0.05": ordinal_correct / max(ordinal_total, 1),
            "valid_pixels": depth_pixels,
            "ordinal_pairs": ordinal_total,
        },
        "quality_target_agreement": {
            "mae": float(np.abs(predicted_quality - target_quality).mean()),
            "brier_mse": float(np.square(predicted_quality - target_quality).mean()),
            "ece_15_bin": ece,
            "reliability_bins": reliability_bins,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
