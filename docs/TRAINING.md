# Training Guide

## Scope

The public pipeline trains one RoadScope-N5 network with detection, road, lane,
relative-depth, and scene-quality outputs. Training requires legal access to
images, detection labels, and aligned dense targets.

## Environment

Create an isolated environment and install the training extras:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

Install a PyTorch build compatible with the local CUDA driver when GPU
training is required.

## Data Layout

```text
dataset/
|-- images/
|   |-- train/
|   `-- val/
|-- labels/
|   |-- train/
|   `-- val/
`-- targets/
    |-- train/
    `-- val/
```

Subdirectories and base filenames must match across all three roots. See
[Dense Target Format](TARGET_FORMAT.md).

BDD100K detection uses these ten class IDs:

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

## Single-GPU Training

```bash
roadscope-n5-train \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --targets dataset/targets/train \
  --weights yolov8n.pt \
  --output runs/train \
  --epochs 100 \
  --batch 16 \
  --imgsz 512 \
  --workers 4
```

Default optimization settings:

| Setting | Default |
|---|---|
| Optimizer | AdamW |
| Learning rate | `2e-3` |
| Weight decay | `5e-4` |
| Scheduler | OneCycleLR |
| Warmup fraction | `0.10` |
| Gradient clipping | norm `10.0` |
| Mixed precision | enabled on CUDA |
| Seed | `42` |
| Input size | `512` |
| Epochs | `100` |

Use `--accumulate N` when memory limits require gradient accumulation. Use
`--limit N` only for debugging; results from a limited run are not a full
benchmark.

## Multi-GPU Training

Launch one process per GPU:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m roadscope_n5.training.train \
  --distributed \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --targets dataset/targets/train \
  --weights yolov8n.pt \
  --output runs/train \
  --epochs 100 \
  --batch 16
```

`--batch` is the per-process batch size. The effective global batch is:

```text
batch_per_gpu x number_of_gpus x gradient_accumulation
```

## Checkpoints and Resume

The output directory contains:

- `last.pt`, including model, optimizer, scheduler, scaler, completed epoch,
  and normalized run arguments;
- `epochN.pt`, model-only periodic checkpoints;
- `metrics.csv`, one row per completed epoch.

Resume with the same training configuration:

```bash
roadscope-n5-train \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --targets dataset/targets/train \
  --weights yolov8n.pt \
  --output runs/train \
  --epochs 100 \
  --batch 16 \
  --resume runs/train/last.pt
```

The runner rejects resume changes to epoch count, batch, input size,
accumulation, seed, learning rate, weight decay, or world size. This protects
the OneCycle schedule and optimizer state from silent corruption.

`--stop-after-epoch N` performs a clean stop after epoch `N` while preserving
the original full schedule for a later resume.

## Evaluation

```bash
roadscope-n5-evaluate \
  --images dataset/images/val \
  --labels dataset/labels/val \
  --targets dataset/targets/val \
  --weights yolov8n.pt \
  --checkpoint runs/train/last.pt \
  --output runs/evaluation.json \
  --imgsz 512 \
  --batch 32
```

Detection metrics use object labels. Dense metrics report agreement with the
supplied validation targets. They must not be presented as independent
ground-truth accuracy unless those targets are genuine, independently audited
ground truth.

## Export

```bash
roadscope-n5-export \
  --checkpoint runs/train/last.pt \
  --weights yolov8n.pt \
  --output runs/roadscope_n5.onnx \
  --imgsz 512
```

Validate numerical agreement before deployment:

```bash
python -m roadscope_n5.tools.validate_onnx \
  --checkpoint runs/train/last.pt \
  --weights yolov8n.pt \
  --onnx runs/roadscope_n5.onnx \
  --output runs/onnx_validation.json
```

## Reproducibility Checklist

Report at least:

- dataset version and exact train/validation split;
- image, label, and dense-target counts;
- target provenance and quality protocol;
- random seed;
- input size, batch size, GPU count, and accumulation;
- package versions and hardware;
- epoch count and selected checkpoint;
- confidence and NMS thresholds;
- full metric definitions;
- median and p95 latency with warmup and precision stated.
