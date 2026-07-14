"""RoadScope-N5 multi-task road-scene perception network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG


MODEL_VERSION = "roadscope-n5"


class RoadScopeN5Model(nn.Module):
    def __init__(self, weights: str | Path = "yolov8n.pt", classes: int = 10, width: int = 64) -> None:
        super().__init__()
        self.detector = DetectionModel("yolov8n.yaml", ch=3, nc=classes, verbose=False)
        self.detector.args = DEFAULT_CFG
        source = YOLO(str(weights)).model
        self.detector.load(source, verbose=False)
        self.features: tuple[Tensor, Tensor, Tensor] | None = None
        self.detector.model[-1].register_forward_pre_hook(self._capture)
        self.project = nn.ModuleList([nn.LazyConv2d(width, 1) for _ in range(3)])
        self.feature_logits = nn.Parameter(torch.zeros(3))
        self.fuse = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.road = nn.Conv2d(width, 1, 1)
        self.lane = nn.Conv2d(width, 1, 1)
        self.depth = nn.Conv2d(width, 1, 1)
        self.confidence = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 1))
        # Learned log variances prevent one auxiliary task from overwhelming
        # detector learning when dense-target quality differs by scene.
        self.task_log_vars = nn.Parameter(torch.zeros(4))

    def _capture(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        value = inputs[0]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise RuntimeError("Expected three multi-scale backbone features")
        self.features = tuple(value)

    def forward(self, images: Tensor) -> dict[str, Any]:
        self.features = None
        detection = self.detector(images)
        if self.features is None:
            raise RuntimeError("Backbone features unavailable")
        size = self.features[0].shape[-2:]
        fused = None
        weights = self.feature_logits.softmax(dim=0)
        for index, (projection, feature) in enumerate(zip(self.project, self.features)):
            current = projection(feature)
            if current.shape[-2:] != size:
                current = F.interpolate(
                    current,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            current = current * weights[index]
            fused = current if fused is None else fused + current
        fused = self.fuse(fused)
        return {
            "detection": detection,
            "road": self.road(fused),
            "lane": self.lane(fused),
            "depth": torch.sigmoid(self.depth(fused)),
            "confidence_logit": self.confidence(fused).squeeze(1),
        }

    def detection_loss(self, batch: dict[str, Tensor], prediction: Any) -> Tensor:
        components, _ = self.detector.loss(batch, prediction)
        return components.sum() / max(int(batch["img"].shape[0]), 1)

    def balance_losses(self, detection: Tensor, segmentation: Tensor, depth: Tensor, confidence: Tensor) -> Tensor:
        losses = torch.stack((detection, segmentation, depth, confidence))
        log_vars = self.task_log_vars.clamp(-5.0, 5.0)
        precision = torch.exp(-log_vars)
        return (precision * losses + log_vars).sum()
