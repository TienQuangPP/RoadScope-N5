"""Stable class, output, and dense-target contracts shared across RoadScope-N5."""

CLASS_NAMES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

MODEL_OUTPUT_NAMES = (
    "detections",
    "road",
    "lane",
    "relative_depth",
    "confidence",
)

TARGET_FORMAT_VERSION = 4
