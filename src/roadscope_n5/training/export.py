"""Export a trained RoadScope-N5 checkpoint to ONNX for edge runtimes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .checkpoint_io import load_checkpoint
from .model import MODEL_VERSION, RoadScopeN5Model


class ExportWrapper(nn.Module):
    def __init__(self, model: RoadScopeN5Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        output = self.model(images)
        detection: Any = output["detection"]
        if isinstance(detection, (tuple, list)):
            detection = detection[0]
        if isinstance(detection, dict):
            detection = torch.cat((detection["boxes"], detection["scores"]), dim=1)
        return (
            detection,
            output["road"].sigmoid(),
            output["lane"].sigmoid(),
            output["depth"],
            output["confidence_logit"].sigmoid(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        raise ValueError("imgsz must be positive and divisible by 32")
    if args.output.suffix.lower() != ".onnx":
        raise ValueError("output must use the .onnx extension")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Checkpoint model version {checkpoint.get('model_version')!r} does not match {MODEL_VERSION!r}"
        )
    model = RoadScopeN5Model(args.weights)
    model.eval()
    model(torch.zeros(1, 3, args.imgsz, args.imgsz))
    model.load_state_dict(checkpoint["model"])
    wrapper = ExportWrapper(model.eval())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, args.imgsz, args.imgsz),
        args.output,
        input_names=["images"],
        output_names=["detections", "road", "lane", "relative_depth", "confidence"],
        dynamic_axes={
            "images": {0: "batch"},
            "detections": {0: "batch"},
            "road": {0: "batch"},
            "lane": {0: "batch"},
            "relative_depth": {0: "batch"},
            "confidence": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
