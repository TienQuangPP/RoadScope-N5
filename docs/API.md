# Python and CLI API

## Python Runtime

The primary integration class is `roadscope_n5.inference.RoadScopeRuntime`.

```python
import cv2

from roadscope_n5.inference import RoadScopeRuntime

runtime = RoadScopeRuntime(
    model="roadscope_n5.onnx",
    provider="auto",
    confidence=0.25,
    iou=0.50,
    max_detections=300,
)

image = cv2.imread("input/example.jpg")
prediction = runtime.predict(image)
```

The input must be a non-empty `uint8` BGR NumPy array with shape
`H x W x 3`, as returned by `cv2.imread`.

## Runtime Constructor

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `model` | path-like | released model | ONNX file |
| `provider` | string | `auto` | `auto`, `cuda`, or `cpu` |
| `confidence` | float | `0.25` | minimum detection confidence |
| `iou` | float | `0.50` | NMS IoU threshold |
| `max_detections` | integer | `300` | maximum boxes per image |

Thresholds must be in `[0, 1]`. A requested CUDA provider fails explicitly if
CUDAExecutionProvider is unavailable; it does not silently pretend to use the
GPU.

## Prediction Object

`predict` returns an immutable `Prediction` dataclass.

| Field | Type | Shape or meaning |
|---|---|---|
| `boxes` | `np.ndarray` | `K x 4`, source-resolution `xyxy` boxes |
| `scores` | `np.ndarray` | `K` confidence values |
| `classes` | `np.ndarray` | `K` integer class IDs |
| `road` | `np.ndarray` | `H x W` road probability |
| `lane` | `np.ndarray` | `H x W` lane probability |
| `relative_depth` | `np.ndarray` | `H x W` normalized relative depth |
| `quality` | float | bounded scene-quality signal |
| `object_depths` | `np.ndarray` | median relative value per box |
| `proximity_ranks` | `np.ndarray` | rank 1 is nearest under the model convention |
| `inference_ms` | float | network execution time only |

Preprocessing, post-processing, visualization, disk I/O, and JSON encoding are
not included in `inference_ms`.

## Class Contract

Class IDs are stable and defined in `roadscope_n5.contracts.CLASS_NAMES`:

```text
0 pedestrian
1 rider
2 car
3 truck
4 bus
5 train
6 motorcycle
7 bicycle
8 traffic light
9 traffic sign
```

## CLI

After `pip install -e .`:

```bash
roadscope-n5 --source input --output output
```

Without installing the project:

```bash
python run.py --source input --output output
```

| Option | Default | Meaning |
|---|---|---|
| `--source` | required | image file or directory |
| `--model` | `roadscope_n5.onnx` | ONNX model path |
| `--output` | `output/` | result directory |
| `--provider` | `auto` | `auto`, `cuda`, or `cpu` |
| `--confidence` | `0.25` | detection threshold |
| `--iou` | `0.50` | class-aware NMS threshold |
| `--road-threshold` | `0.50` | road overlay threshold |
| `--lane-threshold` | `0.50` | lane overlay threshold |
| `--max-det` | `300` | maximum boxes per image |
| `--limit` | unlimited | process only the first N images |

Directories are scanned recursively. Duplicate filename stems are rejected
because they would overwrite outputs.

## Errors and Privacy

The runtime fails on a missing model, wrong ONNX contract, invalid threshold,
unreadable input set, duplicate stems, non-finite model output, or failed image
write. JSON reports use paths relative to the repository when possible and
otherwise store only the source filename, avoiding accidental disclosure of a
user's parent directories.
