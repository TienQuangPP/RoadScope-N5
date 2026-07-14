# Dense Target Format

## Purpose

RoadScope-N5 training combines object-detection labels with aligned dense maps.
Each image has one compressed NumPy archive with the same relative path and
base filename.

Example:

```text
images/train/city/frame-001.jpg
labels/train/city/frame-001.txt
targets/train/city/frame-001.npz
```

The detection label uses normalized YOLO text format:

```text
class_id center_x center_y width height
```

All coordinates are in `[0, 1]` relative to the original image.

## Archive Contract

The dense archive format version is `4`.

| Key | Recommended dtype | Shape at input 512 | Meaning |
|---|---|---|---|
| `version` | `int16` scalar | scalar | target format version, value `4` |
| `road` | `uint8` | `64 x 64` | binary drivable-road mask |
| `lane` | `uint8` | `64 x 64` | binary lane-marking mask |
| `relative_depth` | `float32` | `64 x 64` | normalized relative-depth target |
| `depth_valid` | `uint8` | `64 x 64` | pixels allowed in depth loss |
| `content_valid` | `uint8` | `64 x 64` | pixels belonging to image content |
| `seg_quality` | `float32` scalar | scalar | segmentation-target quality in `[0, 1]` |
| `depth_quality` | `float32` scalar | scalar | depth-target quality in `[0, 1]` |

The dense size must equal `imgsz / 8`. For a different training input size,
create targets at the corresponding stride-8 resolution.

## Validity Rules

- Every required key must exist.
- Every dense map must have the exact expected shape.
- Every dense map must contain only finite values.
- Road, lane, depth-valid, and content-valid maps must be binary.
- Relative-depth values must be in `[0, 1]`.
- Invalid depth pixels must be set to zero.
- `depth_valid` must not extend outside `content_valid`.
- Road and lane pixels outside `content_valid` must be zero.
- Quality values must be finite and in `[0, 1]`.
- Image, label, and dense-target paths must remain aligned.

## Build One Archive

Install the project, then run:

```bash
roadscope-n5-build-target \
  --road maps/road/frame-001.png \
  --lane maps/lane/frame-001.png \
  --depth maps/depth/frame-001.npy \
  --output targets/train/frame-001.npz \
  --size 64 \
  --segmentation-quality 1.0 \
  --depth-quality 1.0
```

Optional validity maps can be supplied with `--content-valid` and
`--depth-valid`.

If a relative-depth source is finite but not already in `[0, 1]`, use
`--normalize-depth`. The builder then applies robust per-image percentile
normalization. This changes scale and should be documented in any experiment.

## Programmatic Creation

```python
from pathlib import Path

from roadscope_n5.tools.build_target import build_target

build_target(
    road_path=Path("road.png"),
    lane_path=Path("lane.png"),
    depth_path=Path("depth.npy"),
    output_path=Path("target.npz"),
    size=64,
    segmentation_quality=1.0,
    depth_quality=1.0,
)
```

## Quality Values

Quality values weight dense losses. They are not model confidence, safety
probability, or ground-truth certainty. If every dense label is manually
verified under the same protocol, `1.0` is a reasonable default. If quality is
estimated, publish the exact method and perform an ablation against equal
weights.
