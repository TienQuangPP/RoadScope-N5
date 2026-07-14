"""BDD100K detection and dense-target dataset for RoadScope-N5 training."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from roadscope_n5.contracts import CLASS_NAMES, TARGET_FORMAT_VERSION


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
cv2.setNumThreads(0)
TARGET_KEYS = {
    "version",
    "road",
    "lane",
    "relative_depth",
    "depth_valid",
    "content_valid",
    "seg_quality",
    "depth_quality",
}
DENSE_MAP_KEYS = ("road", "lane", "relative_depth", "depth_valid", "content_valid")
BINARY_MAP_KEYS = ("road", "lane", "depth_valid", "content_valid")


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    output = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output, scale, left, top


def _read_boxes(path: Path, width: int, height: int, scale: float, left: int, top: int, size: int) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.empty((0, 5), dtype=np.float32)
    labels = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError(f"Expected five YOLO columns in {path}, got shape {labels.shape}")
    if not np.isfinite(labels).all():
        raise ValueError(f"Detection labels contain NaN or Inf: {path}")
    classes, x, y, box_width, box_height = labels.T
    integer_classes = np.isclose(classes, np.rint(classes), atol=1e-6)
    valid_classes = integer_classes & (classes >= 0) & (classes < len(CLASS_NAMES))
    valid_geometry = (
        (x >= 0.0)
        & (x <= 1.0)
        & (y >= 0.0)
        & (y <= 1.0)
        & (box_width > 0.0)
        & (box_width <= 1.0)
        & (box_height > 0.0)
        & (box_height <= 1.0)
    )
    if not (valid_classes & valid_geometry).all():
        invalid_rows = np.flatnonzero(~(valid_classes & valid_geometry)) + 1
        raise ValueError(f"Invalid YOLO rows {invalid_rows.tolist()} in {path}")
    x = (x * width * scale + left) / size
    y = (y * height * scale + top) / size
    box_width = box_width * width * scale / size
    box_height = box_height * height * scale / size
    boxes = np.stack((classes, x, y, box_width, box_height), axis=1)
    boxes[:, 1:] = np.clip(boxes[:, 1:], 0.0, 1.0)
    return boxes.astype(np.float32, copy=False)


def _validate_dense_target(dense: dict[str, np.ndarray], path: Path, target_size: int) -> None:
    expected_shape = (target_size, target_size)
    for key in DENSE_MAP_KEYS:
        value = np.asarray(dense[key])
        if value.shape != expected_shape:
            raise ValueError(f"Invalid {key} shape {value.shape} in {path}; expected {expected_shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"Dense target {key} contains NaN or Inf: {path}")
    for key in BINARY_MAP_KEYS:
        value = np.asarray(dense[key])
        if not np.logical_or(value == 0, value == 1).all():
            raise ValueError(f"Dense target {key} must be binary: {path}")
    relative_depth = np.asarray(dense["relative_depth"])
    if float(relative_depth.min()) < 0.0 or float(relative_depth.max()) > 1.0:
        raise ValueError(f"relative_depth must be in [0, 1]: {path}")
    for key in ("seg_quality", "depth_quality"):
        value = np.asarray(dense[key])
        if value.size != 1:
            raise ValueError(f"Dense target {key} must be scalar: {path}")
        scalar = float(value.reshape(-1)[0])
        if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
            raise ValueError(f"Dense target {key} must be finite and in [0, 1]: {path}")
    content_valid = np.asarray(dense["content_valid"]) > 0
    depth_valid = np.asarray(dense["depth_valid"]) > 0
    if np.any(depth_valid & ~content_valid):
        raise ValueError(f"depth_valid extends outside content_valid: {path}")
    if np.any(np.asarray(dense["road"])[~content_valid]) or np.any(
        np.asarray(dense["lane"])[~content_valid]
    ):
        raise ValueError(f"Road or lane labels extend outside content_valid: {path}")
    if np.any(np.abs(relative_depth[~depth_valid]) > 1e-6):
        raise ValueError(f"relative_depth must be zero outside depth_valid: {path}")


def _photometric(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.75:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-5.0, 5.0)) % 180.0
        hsv[..., 1] *= random.uniform(0.75, 1.25)
        hsv[..., 2] *= random.uniform(0.65, 1.35)
        image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.2, 1.2))
    if random.random() < 0.10:
        noise = np.random.normal(0.0, random.uniform(2.0, 8.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.08:
        fog = random.uniform(0.05, 0.18)
        image = np.clip(image.astype(np.float32) * (1.0 - fog) + 235.0 * fog, 0, 255).astype(np.uint8)
    if random.random() < 0.06:
        overlay = image.copy()
        height, width = image.shape[:2]
        for _ in range(random.randint(20, 45)):
            x = random.randrange(width)
            y = random.randrange(height)
            length = random.randint(max(4, height // 80), max(8, height // 35))
            cv2.line(
                overlay,
                (x, y),
                (min(width - 1, x + length // 3), min(height - 1, y + length)),
                (205, 205, 205),
                1,
            )
        image = cv2.addWeighted(image, 0.82, overlay, 0.18, 0.0)
    return image


def _scene_quality(image: np.ndarray, content_valid: np.ndarray) -> float:
    mask = cv2.resize(content_valid.astype(np.uint8), image.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask]
    if pixels.size == 0:
        return 0.0
    p10, median, p90 = np.percentile(pixels, [10, 50, 90])
    contrast = np.clip((p90 - p10 - 20.0) / 80.0, 0.0, 1.0)
    illumination = np.clip((median - 18.0) / 72.0, 0.0, 1.0)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharp_mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    sharpness_value = float(laplacian[sharp_mask].var()) if sharp_mask.any() else 0.0
    sharpness = np.clip(
        (np.log1p(sharpness_value) - np.log1p(12.0)) / (np.log1p(500.0) - np.log1p(12.0)),
        0.0,
        1.0,
    )
    return float(0.40 * illumination + 0.30 * contrast + 0.30 * sharpness)


def _random_affine(
    image: np.ndarray,
    boxes: np.ndarray,
    dense: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    size = image.shape[0]
    angle = random.uniform(-2.0, 2.0)
    scale = random.uniform(0.90, 1.10)
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, scale)
    matrix[0, 2] += random.uniform(-0.05, 0.05) * size
    matrix[1, 2] += random.uniform(-0.04, 0.04) * size
    image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    if boxes.shape[0]:
        xyxy = np.empty((boxes.shape[0], 4), dtype=np.float32)
        xyxy[:, 0] = (boxes[:, 1] - boxes[:, 3] / 2.0) * size
        xyxy[:, 1] = (boxes[:, 2] - boxes[:, 4] / 2.0) * size
        xyxy[:, 2] = (boxes[:, 1] + boxes[:, 3] / 2.0) * size
        xyxy[:, 3] = (boxes[:, 2] + boxes[:, 4] / 2.0) * size
        corners = np.ones((boxes.shape[0], 4, 3), dtype=np.float32)
        corners[:, :, :2] = xyxy[:, (0, 1, 2, 1, 2, 3, 0, 3)].reshape(-1, 4, 2)
        transformed = corners @ matrix.T
        transformed_xyxy = np.concatenate((transformed.min(1), transformed.max(1)), axis=1)
        transformed_xyxy[:, (0, 2)] = transformed_xyxy[:, (0, 2)].clip(0, size)
        transformed_xyxy[:, (1, 3)] = transformed_xyxy[:, (1, 3)].clip(0, size)
        widths = transformed_xyxy[:, 2] - transformed_xyxy[:, 0]
        heights = transformed_xyxy[:, 3] - transformed_xyxy[:, 1]
        original_area = np.maximum((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]), 1e-6)
        keep = (widths >= 2.0) & (heights >= 2.0) & (widths * heights >= 0.10 * original_area)
        boxes = boxes[keep]
        transformed_xyxy = transformed_xyxy[keep]
        if boxes.shape[0]:
            boxes[:, 1] = (transformed_xyxy[:, 0] + transformed_xyxy[:, 2]) / (2.0 * size)
            boxes[:, 2] = (transformed_xyxy[:, 1] + transformed_xyxy[:, 3]) / (2.0 * size)
            boxes[:, 3] = (transformed_xyxy[:, 2] - transformed_xyxy[:, 0]) / size
            boxes[:, 4] = (transformed_xyxy[:, 3] - transformed_xyxy[:, 1]) / size
    target_size = dense["content_valid"].shape[0]
    dense_matrix = matrix.copy()
    dense_matrix[:, 2] *= target_size / size
    for key in ("road", "lane", "depth_valid", "content_valid"):
        dense[key] = cv2.warpAffine(
            dense[key], dense_matrix, (target_size, target_size), flags=cv2.INTER_NEAREST, borderValue=0
        )
    dense["relative_depth"] = cv2.warpAffine(
        dense["relative_depth"].astype(np.float32),
        dense_matrix,
        (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    dense["road"][dense["content_valid"] == 0] = 0
    dense["lane"][dense["content_valid"] == 0] = 0
    dense["depth_valid"][dense["content_valid"] == 0] = 0
    dense["relative_depth"][dense["depth_valid"] == 0] = 0
    return image, boxes, dense


class BDDMultiTaskDataset(Dataset):
    def __init__(
        self,
        images_root: str | Path,
        labels_root: str | Path,
        targets_root: str | Path,
        imgsz: int = 512,
        augment: bool = False,
        require_targets: bool = True,
        limit: int | None = None,
    ) -> None:
        self.images_root = Path(images_root).resolve()
        self.labels_root = Path(labels_root).resolve()
        self.targets_root = Path(targets_root).resolve()
        if not self.images_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.images_root}")
        if not self.labels_root.is_dir():
            raise FileNotFoundError(f"Label directory not found: {self.labels_root}")
        if require_targets and not self.targets_root.is_dir():
            raise FileNotFoundError(f"Target directory not found: {self.targets_root}")
        self.imgsz = imgsz
        if imgsz <= 0 or imgsz % 32 != 0:
            raise ValueError(f"imgsz must be positive and divisible by 32, got {imgsz}")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")
        self.augment = augment
        self.require_targets = require_targets
        self.images = sorted(path for path in self.images_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if limit is not None:
            self.images = self.images[:limit]
        if not self.images:
            raise FileNotFoundError(f"No images found under {self.images_root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_path = self.images[index]
        relative = image_path.relative_to(self.images_root)
        label_path = (self.labels_root / relative).with_suffix(".txt")
        target_path = (self.targets_root / relative).with_suffix(".npz")
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        original_height, original_width = image.shape[:2]
        image, scale, left, top = _letterbox(image, self.imgsz)
        boxes = _read_boxes(label_path, original_width, original_height, scale, left, top, self.imgsz)
        if not target_path.is_file():
            if self.require_targets:
                raise FileNotFoundError(f"Missing dense target: {target_path}")
            target_size = self.imgsz // 8
            dense = {
                "road": np.zeros((target_size, target_size), dtype=np.uint8),
                "lane": np.zeros((target_size, target_size), dtype=np.uint8),
                "relative_depth": np.zeros((target_size, target_size), dtype=np.float32),
                "depth_valid": np.zeros((target_size, target_size), dtype=np.uint8),
                "content_valid": np.ones((target_size, target_size), dtype=np.uint8),
                "seg_quality": np.float32(0.0),
                "depth_quality": np.float32(0.0),
            }
        else:
            with np.load(target_path, allow_pickle=False) as loaded:
                missing = TARGET_KEYS.difference(loaded.files)
                if missing:
                    raise ValueError(f"Invalid dense target {target_path}: missing {sorted(missing)}")
                version = int(np.asarray(loaded["version"]).item())
                if version != TARGET_FORMAT_VERSION:
                    raise ValueError(
                        f"Invalid target format {version} in {target_path}; "
                        f"expected {TARGET_FORMAT_VERSION}"
                    )
                dense = {key: np.asarray(loaded[key]).copy() for key in TARGET_KEYS}
            target_size = self.imgsz // 8
            _validate_dense_target(dense, target_path, target_size)
        if self.augment:
            image = _photometric(image)
            if random.random() < 0.50:
                image, boxes, dense = _random_affine(image, boxes, dense)
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                boxes[:, 1] = 1.0 - boxes[:, 1]
                for key in DENSE_MAP_KEYS:
                    dense[key] = np.ascontiguousarray(dense[key][:, ::-1])
        confidence_target = min(float(dense["seg_quality"]), float(dense["depth_quality"]))
        if self.augment:
            confidence_target = min(confidence_target, _scene_quality(image, dense["content_valid"]))
        image_tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255.0)
        return {
            "img": image_tensor,
            "cls": torch.from_numpy(boxes[:, :1]),
            "bboxes": torch.from_numpy(boxes[:, 1:]),
            "road": torch.from_numpy(dense["road"].astype(np.float32, copy=False)).unsqueeze(0),
            "lane": torch.from_numpy(dense["lane"].astype(np.float32, copy=False)).unsqueeze(0),
            "depth": torch.from_numpy(dense["relative_depth"].astype(np.float32, copy=False)).unsqueeze(0),
            "depth_valid": torch.from_numpy(dense["depth_valid"].astype(np.float32, copy=False)).unsqueeze(0),
            "content_valid": torch.from_numpy(dense["content_valid"].astype(np.float32, copy=False)).unsqueeze(0),
            "seg_quality": torch.tensor(float(dense["seg_quality"]), dtype=torch.float32),
            "depth_quality": torch.tensor(float(dense["depth_quality"]), dtype=torch.float32),
            "confidence_target": torch.tensor(confidence_target, dtype=torch.float32),
            "path": str(image_path),
        }


def collate_multitask(samples: list[dict[str, Tensor | str]]) -> dict[str, Tensor | list[str]]:
    result: dict[str, Tensor | list[str]] = {}
    stacked_keys = (
        "img",
        "road",
        "lane",
        "depth",
        "depth_valid",
        "content_valid",
        "seg_quality",
        "depth_quality",
        "confidence_target",
    )
    for key in stacked_keys:
        result[key] = torch.stack([sample[key] for sample in samples])  # type: ignore[arg-type]
    classes, boxes, indices = [], [], []
    for index, sample in enumerate(samples):
        sample_classes = sample["cls"]
        sample_boxes = sample["bboxes"]
        classes.append(sample_classes)
        boxes.append(sample_boxes)
        indices.append(torch.full((sample_classes.shape[0],), index, dtype=torch.long))  # type: ignore[union-attr]
    result["cls"] = torch.cat(classes, dim=0)
    result["bboxes"] = torch.cat(boxes, dim=0)
    result["batch_idx"] = torch.cat(indices, dim=0)
    result["paths"] = [str(sample["path"]) for sample in samples]
    return result
