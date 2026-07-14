from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from roadscope_n5.contracts import MODEL_OUTPUT_NAMES


MODEL = ROOT / "roadscope_n5.onnx"


class ModelContractTests(unittest.TestCase):
    def test_released_model_contract(self) -> None:
        session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        self.assertEqual(model_input.name, "images")
        self.assertEqual(model_input.shape[1:], [3, 512, 512])
        self.assertEqual(tuple(output.name for output in session.get_outputs()), MODEL_OUTPUT_NAMES)

    def test_released_model_outputs_are_finite(self) -> None:
        session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        outputs = session.run(None, {model_input.name: np.zeros((1, 3, 512, 512), dtype=np.float32)})
        self.assertEqual(len(outputs), len(MODEL_OUTPUT_NAMES))
        self.assertTrue(all(np.isfinite(output).all() for output in outputs))


if __name__ == "__main__":
    unittest.main()
