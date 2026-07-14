from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from roadscope_n5.inference import (
    decode_detections,
    letterbox,
    nms,
    reject_duplicate_stems,
    RoadScopeRuntime,
    restore_boxes,
    restore_dense,
    safe_source_label,
    write_image,
)


MODEL = ROOT / "roadscope_n5.onnx"


class GeometryTests(unittest.TestCase):
    def test_letterbox_and_restore_boxes(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        _, scale, left, top, _, _ = letterbox(image, 512)
        source_box = np.array([[100.0, 50.0, 900.0, 600.0]], dtype=np.float32)
        model_box = source_box.copy()
        model_box[:, (0, 2)] = model_box[:, (0, 2)] * scale + left
        model_box[:, (1, 3)] = model_box[:, (1, 3)] * scale + top

        restored = restore_boxes(model_box, scale, left, top, 1280, 720)

        np.testing.assert_allclose(restored, source_box, atol=1e-4)

    def test_restore_dense_returns_source_geometry(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        _, _, left, top, resized_width, resized_height = letterbox(image, 512)
        dense = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(1, 1, 64, 64)

        restored = restore_dense(
            dense,
            512,
            left,
            top,
            resized_width,
            resized_height,
            1280,
            720,
            cv2.INTER_LINEAR,
        )

        self.assertEqual(restored.shape, (720, 1280))
        self.assertTrue(np.isfinite(restored).all())


class DetectionTests(unittest.TestCase):
    def test_nms_is_class_aware(self) -> None:
        boxes = np.array(
            [[0.0, 0.0, 100.0, 100.0], [2.0, 2.0, 98.0, 98.0], [0.0, 0.0, 100.0, 100.0]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 1], dtype=np.int64)

        keep = nms(boxes, scores, classes, threshold=0.5, max_det=10)

        np.testing.assert_array_equal(keep, np.array([0, 2], dtype=np.int64))

    def test_decode_detections_accepts_export_layout(self) -> None:
        raw = np.zeros((1, 14, 1), dtype=np.float32)
        raw[0, :4, 0] = [100.0, 80.0, 40.0, 20.0]
        raw[0, 6, 0] = 0.9

        boxes, scores, classes = decode_detections(raw, 0.25, 0.5, 10)

        np.testing.assert_allclose(boxes, [[80.0, 70.0, 120.0, 90.0]])
        np.testing.assert_allclose(scores, [0.9])
        np.testing.assert_array_equal(classes, [2])


class SafetyAndIoTests(unittest.TestCase):
    def test_invalid_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "provider"):
            RoadScopeRuntime(MODEL, provider="invalid")

    def test_non_uint8_image_is_rejected_before_inference(self) -> None:
        runtime = object.__new__(RoadScopeRuntime)
        image = np.zeros((8, 8, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "uint8"):
            runtime.predict(image)

    def test_duplicate_stems_are_rejected(self) -> None:
        images = [Path("camera-a/frame.jpg"), Path("camera-b/frame.png")]
        with self.assertRaisesRegex(RuntimeError, "Duplicate input stems"):
            reject_duplicate_stems(images)

    def test_external_source_label_hides_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "private-directory" / "frame.jpg"
            self.assertEqual(safe_source_label(external), "frame.jpg")

    def test_write_image_reports_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame.png"
            write_image(output, np.zeros((8, 8, 3), dtype=np.uint8))
            self.assertTrue(output.is_file())
            self.assertIsNotNone(cv2.imread(str(output)))


if __name__ == "__main__":
    unittest.main()
