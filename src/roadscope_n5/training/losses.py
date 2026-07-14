"""Quality-aware auxiliary losses for dense supervision targets."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def _quality_mean(values: Tensor, quality: Tensor) -> Tensor:
    quality = quality.clamp(0.0, 1.0)
    return (values * quality).sum() / quality.sum().clamp_min(1e-6)


def _dice(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    probability = (logits.sigmoid() * valid).flatten(1)
    target = (target * valid).flatten(1)
    overlap = 2.0 * (probability * target).sum(1) + 1.0
    total = probability.sum(1) + target.sum(1) + 1.0
    return 1.0 - overlap / total


def segmentation_loss(
    road_logits: Tensor,
    lane_logits: Tensor,
    road: Tensor,
    lane: Tensor,
    valid: Tensor,
    quality: Tensor,
) -> Tensor:
    valid = (valid > 0.5).to(road_logits.dtype)
    pixel_count = valid.flatten(1).sum(1).clamp_min(1.0)
    road_bce = (
        (F.binary_cross_entropy_with_logits(road_logits, road, reduction="none") * valid)
        .flatten(1)
        .sum(1)
        / pixel_count
    )
    lane_bce = (
        (F.binary_cross_entropy_with_logits(lane_logits, lane, reduction="none") * valid)
        .flatten(1)
        .sum(1)
        / pixel_count
    )
    per_image = road_bce + _dice(road_logits, road, valid) + 1.5 * lane_bce + 1.5 * _dice(lane_logits, lane, valid)
    return _quality_mean(per_image, quality)


def relative_depth_loss(prediction: Tensor, target: Tensor, valid: Tensor, quality: Tensor) -> Tensor:
    valid = valid > 0.5
    pixel_count = valid.flatten(1).sum(1).clamp_min(1)
    difference = F.smooth_l1_loss(prediction, target, reduction="none")
    value_loss = (difference * valid).flatten(1).sum(1) / pixel_count
    pred_dx, target_dx = prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1]
    pred_dy, target_dy = prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    valid_dx, valid_dy = valid[..., 1:] & valid[..., :-1], valid[..., 1:, :] & valid[..., :-1, :]
    grad_x = ((pred_dx - target_dx).abs() * valid_dx).flatten(1).sum(1) / valid_dx.flatten(1).sum(1).clamp_min(1)
    grad_y = ((pred_dy - target_dy).abs() * valid_dy).flatten(1).sum(1) / valid_dy.flatten(1).sum(1).clamp_min(1)
    flat_prediction = prediction.flatten(1)
    flat_target = target.flatten(1)
    flat_valid = valid.flatten(1)
    shift = max(flat_target.shape[1] // 2 - 1, 1)
    paired_prediction = flat_prediction.roll(shifts=shift, dims=1)
    paired_target = flat_target.roll(shifts=shift, dims=1)
    paired_valid = flat_valid & flat_valid.roll(shifts=shift, dims=1)
    target_delta = flat_target - paired_target
    ordinal_valid = paired_valid & (target_delta.abs() >= 0.05)
    ordinal = F.softplus(-10.0 * target_delta.sign() * (flat_prediction - paired_prediction))
    ordinal_loss = (ordinal * ordinal_valid).sum(1) / ordinal_valid.sum(1).clamp_min(1)
    return _quality_mean(value_loss + 0.25 * (grad_x + grad_y) + 0.10 * ordinal_loss, quality)


def confidence_loss(logit: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logit, target.clamp(0.0, 1.0))
