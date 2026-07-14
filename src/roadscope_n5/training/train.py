"""Train the RoadScope-N5 multi-task model with resumable checkpoints."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint_io import load_checkpoint
from .dataset import BDDMultiTaskDataset, collate_multitask
from .losses import confidence_loss, relative_depth_loss, segmentation_loss
from .model import MODEL_VERSION, RoadScopeN5Model


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def save_checkpoint(
    path: Path,
    model: RoadScopeN5Model,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    scaler: GradScaler,
    epoch: int,
    args: argparse.Namespace,
    training_state: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "epoch": epoch,
        "model_version": MODEL_VERSION,
        "model": model.state_dict(),
        "args": {
            key: value.as_posix() if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if training_state:
        checkpoint.update({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        })
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def optimizer_groups(model: RoadScopeN5Model, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolov8n.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="Stop cleanly after this 1-based epoch while preserving the full scheduler",
    )
    parser.add_argument("--distributed", action="store_true", help="Run with torchrun DistributedDataParallel")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if args.batch <= 0:
        raise ValueError("batch must be greater than zero")
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        raise ValueError("imgsz must be positive and divisible by 32")
    if args.workers < 0:
        raise ValueError("workers must be zero or greater")
    if args.lr <= 0.0:
        raise ValueError("lr must be greater than zero")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be zero or greater")
    if args.accumulate <= 0:
        raise ValueError("accumulate must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be greater than zero")
    if args.save_period <= 0:
        raise ValueError("save-period must be greater than zero")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        raise ValueError("stop-after-epoch must be between 1 and epochs")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    rank, local_rank, world_size = 0, 0, 1
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    args.world_size = world_size
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dataset = BDDMultiTaskDataset(args.images, args.labels, args.targets, args.imgsz, augment=True, limit=args.limit)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if args.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_multitask,
        drop_last=False,
    )
    model = RoadScopeN5Model(args.weights).to(device)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))
    model.train()
    runner: torch.nn.Module = model
    if args.distributed:
        runner = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        if rank == 0:
            print(f"DISTRIBUTED_WORLD_SIZE={world_size}", flush=True)
    optimizer = AdamW(optimizer_groups(model, args.weight_decay), lr=args.lr)
    updates_per_epoch = max((len(loader) + args.accumulate - 1) // args.accumulate, 1)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=updates_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        if checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError(
                f"Checkpoint model version {checkpoint.get('model_version')!r} does not match {MODEL_VERSION!r}"
            )
        previous_args = checkpoint.get("args", {})
        for key in ("epochs", "batch", "imgsz", "accumulate", "seed", "lr", "weight_decay", "world_size"):
            if key in previous_args and previous_args[key] != getattr(args, key):
                raise ValueError(
                    f"Resume mismatch for {key}: checkpoint={previous_args[key]} current={getattr(args, key)}"
                )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.csv"
    if start_epoch > 0 and not metrics_path.is_file():
        raise FileNotFoundError(f"Resume requires the existing metrics file: {metrics_path}")
    if start_epoch == 0 and rank == 0:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow((
                "epoch", "total", "detection", "segmentation", "depth", "confidence", "lr",
                "logvar_detection", "logvar_segmentation", "logvar_depth", "logvar_confidence",
                "fusion_p3", "fusion_p4", "fusion_p5", "grad_norm", "images_per_second", "epoch_seconds",
            ))
    end_epoch = args.epochs if args.stop_after_epoch is None else min(args.epochs, args.stop_after_epoch)
    if end_epoch <= start_epoch:
        raise ValueError(f"stop-after-epoch={end_epoch} must be greater than resumed start epoch={start_epoch}")
    for epoch in range(start_epoch, end_epoch):
        epoch_started = time.perf_counter()
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(5, dtype=np.float64)
        gradient_norm_total = 0.0
        optimizer_updates = 0
        for step, raw_batch in enumerate(loader):
            batch = move_batch(raw_batch, device)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = runner(batch["img"])
                det = model.detection_loss(batch, prediction["detection"])
                seg = segmentation_loss(
                    prediction["road"],
                    prediction["lane"],
                    batch["road"],
                    batch["lane"],
                    batch["content_valid"],
                    batch["seg_quality"],
                )
                depth = relative_depth_loss(
                    prediction["depth"],
                    batch["depth"],
                    batch["depth_valid"],
                    batch["depth_quality"],
                )
                confidence = confidence_loss(prediction["confidence_logit"], batch["confidence_target"])
                total = model.balance_losses(det, seg, depth, confidence) / args.accumulate
            scaler.scale(total).backward()
            update = (step + 1) % args.accumulate == 0 or step + 1 == len(loader)
            if update:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                gradient_norm_total += float(gradient_norm.item())
                optimizer_updates += 1
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
            values = np.array((total.item() * args.accumulate, det.item(), seg.item(), depth.item(), confidence.item()))
            totals += values
            if rank == 0 and ((step + 1) % 50 == 0 or step + 1 == len(loader)):
                mean = totals / (step + 1)
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step + 1}/{len(loader)} "
                    f"total={mean[0]:.4f} det={mean[1]:.4f} "
                    f"seg={mean[2]:.4f} depth={mean[3]:.4f} "
                    f"conf={mean[4]:.4f}",
                    flush=True,
                )
        totals_tensor = torch.tensor(totals, dtype=torch.float64, device=device)
        if args.distributed:
            dist.all_reduce(totals_tensor, op=dist.ReduceOp.SUM)
        mean = totals_tensor.cpu().numpy() / max(len(loader) * world_size, 1)
        if rank == 0:
            epoch_seconds = time.perf_counter() - epoch_started
            images_per_second = len(dataset) / max(epoch_seconds, 1e-9)
            log_vars = model.task_log_vars.detach().cpu().tolist()
            fusion = model.feature_logits.detach().softmax(dim=0).cpu().tolist()
            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow((
                    epoch + 1, *mean.tolist(), optimizer.param_groups[0]["lr"],
                    *log_vars, *fusion,
                    gradient_norm_total / max(optimizer_updates, 1),
                    images_per_second, epoch_seconds,
                ))
            save_checkpoint(args.output / "last.pt", model, optimizer, scheduler, scaler, epoch, args)
            if (epoch + 1) % args.save_period == 0:
                save_checkpoint(
                    args.output / f"epoch{epoch + 1}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    args,
                    training_state=False,
                )
            print(
                f"EPOCH_COMPLETE={epoch + 1} seconds={epoch_seconds:.1f} images_per_second={images_per_second:.2f} "
                f"fusion={[round(value, 4) for value in fusion]} log_vars={[round(value, 4) for value in log_vars]}",
                flush=True,
            )
        if args.distributed:
            dist.barrier()
        if device.type == "cuda":
            peak = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.float64, device=device)
            if args.distributed:
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"CUDA_MAX_MEMORY_GB={peak.item() / 1e9:.3f}", flush=True)
            torch.cuda.reset_peak_memory_stats(device)
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
