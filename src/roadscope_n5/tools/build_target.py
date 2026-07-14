"""Build one validated RoadScope-N5 dense-target archive from supervised maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from roadscope_n5.contracts import TARGET_FORMAT_VERSION


def _read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise RuntimeError(f"Unreadable target map: {path}")
        if value.ndim == 3:
            value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    value = np.asarray(value).squeeze()
    if value.ndim != 2:
        raise ValueError(f"Expected a two-dimensional map, got {value.shape} from {path}")
    return value


def _binary_map(path: Path, size: int) -> np.ndarray:
    value = _read_array(path)
    value = cv2.resize(value.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    if not np.isfinite(value).all():
        raise ValueError(f"Binary map contains NaN or Inf: {path}")
    threshold = 0.5 if float(value.max()) <= 1.0 else 127.0
    return (value > threshold).astype(np.uint8)


def _depth_map(path: Path, size: int, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    value = _read_array(path).astype(np.float32)
    valid = np.isfinite(value)
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = cv2.resize(value, (size, size), interpolation=cv2.INTER_LINEAR)
    valid = cv2.resize(valid.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    valid_values = value[valid]
    if valid_values.size == 0:
        raise ValueError(f"Depth map has no finite values: {path}")
    if normalize:
        low, high = np.percentile(valid_values, [2.0, 98.0])
        if high <= low:
            raise ValueError(f"Depth map has no usable dynamic range: {path}")
        value = np.clip((value - low) / (high - low), 0.0, 1.0)
    elif float(valid_values.min()) < 0.0 or float(valid_values.max()) > 1.0:
        raise ValueError("Depth values must be in [0, 1]; use --normalize-depth when appropriate")
    value[~valid] = 0.0
    return value.astype(np.float32), valid.astype(np.uint8)


def build_target(
    road_path: Path,
    lane_path: Path,
    depth_path: Path,
    output_path: Path,
    size: int,
    segmentation_quality: float,
    depth_quality: float,
    content_valid_path: Path | None = None,
    depth_valid_path: Path | None = None,
    normalize_depth: bool = False,
) -> Path:
    """Create a compressed target archive after strict shape and range checks."""
    if size <= 0:
        raise ValueError("size must be positive")
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output_path must use the .npz extension")
    for name, value in {
        "segmentation_quality": segmentation_quality,
        "depth_quality": depth_quality,
    }.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    road = _binary_map(road_path, size)
    lane = _binary_map(lane_path, size)
    relative_depth, finite_depth = _depth_map(depth_path, size, normalize_depth)
    content_valid = (
        _binary_map(content_valid_path, size)
        if content_valid_path is not None
        else np.ones((size, size), dtype=np.uint8)
    )
    supplied_depth_valid = (
        _binary_map(depth_valid_path, size)
        if depth_valid_path is not None
        else np.ones((size, size), dtype=np.uint8)
    )
    depth_valid = (finite_depth & supplied_depth_valid & content_valid).astype(np.uint8)
    road[content_valid == 0] = 0
    lane[content_valid == 0] = 0
    relative_depth[depth_valid == 0] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        version=np.int16(TARGET_FORMAT_VERSION),
        road=road,
        lane=lane,
        relative_depth=relative_depth,
        depth_valid=depth_valid,
        content_valid=content_valid,
        seg_quality=np.float32(segmentation_quality),
        depth_quality=np.float32(depth_quality),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road", type=Path, required=True)
    parser.add_argument("--lane", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--segmentation-quality", type=float, default=1.0)
    parser.add_argument("--depth-quality", type=float, default=1.0)
    parser.add_argument("--content-valid", type=Path)
    parser.add_argument("--depth-valid", type=Path)
    parser.add_argument("--normalize-depth", action="store_true")
    args = parser.parse_args()
    output = build_target(
        road_path=args.road,
        lane_path=args.lane,
        depth_path=args.depth,
        output_path=args.output,
        size=args.size,
        segmentation_quality=args.segmentation_quality,
        depth_quality=args.depth_quality,
        content_valid_path=args.content_valid,
        depth_valid_path=args.depth_valid,
        normalize_depth=args.normalize_depth,
    )
    print(f"WROTE={output.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
