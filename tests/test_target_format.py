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

from roadscope_n5.contracts import TARGET_FORMAT_VERSION
from roadscope_n5.tools.build_target import build_target


class TargetFormatTests(unittest.TestCase):
    def test_build_target_writes_valid_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            road = np.zeros((16, 16), dtype=np.uint8)
            road[8:] = 255
            lane = np.zeros((16, 16), dtype=np.uint8)
            lane[:, 7:9] = 255
            depth = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
            road_path = root / "road.png"
            lane_path = root / "lane.png"
            depth_path = root / "depth.npy"
            output_path = root / "target.npz"
            self.assertTrue(cv2.imwrite(str(road_path), road))
            self.assertTrue(cv2.imwrite(str(lane_path), lane))
            np.save(depth_path, depth)

            build_target(road_path, lane_path, depth_path, output_path, 8, 0.9, 0.8)

            with np.load(output_path, allow_pickle=False) as target:
                self.assertEqual(int(target["version"]), TARGET_FORMAT_VERSION)
                self.assertEqual(target["road"].shape, (8, 8))
                self.assertEqual(target["lane"].shape, (8, 8))
                self.assertEqual(target["relative_depth"].shape, (8, 8))
                self.assertTrue(np.isfinite(target["relative_depth"]).all())

    def test_build_target_rejects_invalid_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((4, 4), dtype=np.uint8)
            map_path = root / "map.png"
            depth_path = root / "depth.npy"
            self.assertTrue(cv2.imwrite(str(map_path), image))
            np.save(depth_path, image.astype(np.float32))

            with self.assertRaisesRegex(ValueError, "segmentation_quality"):
                build_target(map_path, map_path, depth_path, root / "target.npz", 4, 1.1, 1.0)

    def test_build_target_rejects_nonfinite_binary_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = np.zeros((4, 4), dtype=np.float32)
            binary[0, 0] = np.nan
            depth = np.zeros((4, 4), dtype=np.float32)
            binary_path = root / "binary.npy"
            depth_path = root / "depth.npy"
            np.save(binary_path, binary)
            np.save(depth_path, depth)

            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                build_target(binary_path, binary_path, depth_path, root / "target.npz", 4, 1.0, 1.0)

    def test_build_target_requires_npz_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((4, 4), dtype=np.uint8)
            map_path = root / "map.png"
            depth_path = root / "depth.npy"
            self.assertTrue(cv2.imwrite(str(map_path), image))
            np.save(depth_path, image.astype(np.float32))

            with self.assertRaisesRegex(ValueError, "npz"):
                build_target(map_path, map_path, depth_path, root / "target.bin", 4, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
