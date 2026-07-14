# RoadScope-N5 Model Card

## Model Summary

RoadScope-N5 is a compact multi-task road-scene perception model. It shares one
feature extractor across object detection, road segmentation, lane
segmentation, relative-depth estimation, and scene-quality estimation.

The public deployment artifact is one standalone ONNX graph. It does not call
or load any additional perception model during inference.

| Property | Value |
|---|---|
| Artifact | `roadscope_n5.onnx` |
| File size | 12,638,390 bytes |
| SHA-256 | `f534d7b0fe27b7bfc2ea5d239661fc25b681c896e50949c6c9f389e22b2b8fcf` |
| PyTorch training parameters | 3,115,785 |
| Stored ONNX parameters | 3,110,581 |
| ONNX opset | 17 |
| Input | RGB float32 tensor, `N x 3 x 512 x 512`, range `[0, 1]` |
| Dense output stride | 8 |
| Dense native resolution | `64 x 64` for a `512 x 512` input |
| Detection classes | 10 |
| Runtime | ONNX Runtime |
| License | GNU AGPL-3.0 |

The ONNX count excludes training-only state that is not required by the
inference graph, so it is slightly smaller than the PyTorch training count.

## Architecture

The model uses a compact YOLOv8n detection backbone. Three multi-scale feature
maps are projected to 64 channels, resized to the highest dense resolution,
weighted by learned softmax coefficients, and summed. Two convolutional fusion
blocks feed road, lane, relative-depth, and scene-quality heads. The original
detection head remains active.

This design makes RoadScope-N5 a project-specific multi-task network while
preserving a small deployment footprint. See
[Architecture](docs/ARCHITECTURE.md) for the source-level description.

## Input Processing

Source images are decoded as BGR images, resized with preserved aspect ratio,
and padded to `512 x 512`. The image is converted to RGB, transposed to NCHW,
converted to float32, and divided by 255.

Bounding boxes and dense outputs are mapped back to the original source
resolution. Consequently, callers may provide images with different sizes and
aspect ratios.

## Output Semantics

### Detection

The detection tensor contains center-format boxes and class scores. Public
post-processing applies confidence filtering, class-aware non-maximum
suppression, source-resolution restoration, and clipping.

### Road and Lane

Road and lane outputs are probability maps. The demo applies a default
threshold of `0.50`; callers may select a different operating threshold.

### Relative Depth

The depth head predicts normalized relative structure. Larger displayed values
represent nearer structure under the released convention. Values are not
metres and are not directly comparable across unrelated images.

### Scene Quality

The quality head emits a bounded image-level signal. It is intended to expose
degraded visual conditions. It is not calibrated as a probability of safety,
accuracy, collision, or successful vehicle control.

## Training Interface

The public training code accepts official object-detection labels and aligned
dense supervision maps. Dense targets include road, lane, relative depth,
validity masks, and bounded quality values. Users must document the provenance
and legal status of any target set they provide.

Training uses quality-weighted segmentation and depth losses, learned
multi-task loss balancing, photometric augmentation, small affine transforms,
mixed precision, AdamW, and OneCycleLR.

## Evaluation

The repository includes evaluators for:

- detection precision and recall;
- AP50 and AP50:95;
- road and lane IoU/pixel agreement;
- relative-depth MAE, RMSE, edge consistency, and ordinal agreement;
- quality MAE and expected calibration error;
- model size and runtime latency.

This model card intentionally does not claim a universal accuracy or latency
number. Results depend on the evaluation labels, data split, thresholds,
runtime provider, precision, hardware, and thermal state. Use the supplied
tools and publish the exact evaluation protocol with any reported number.

## Intended Uses

- static-image perception research;
- educational multi-task-learning experiments;
- qualitative road-scene visualization;
- deployment and optimization experiments on non-safety-critical hardware;
- reproducible comparisons using documented datasets and metrics.

## Out-of-Scope Uses

- direct steering, braking, or acceleration control;
- collision avoidance without independent safety systems;
- metric distance estimation;
- claims of autonomous-driving certification;
- surveillance, identity recognition, or individual profiling;
- use where an error can directly cause injury or material damage.

## Known Limitations

- Small, distant, heavily occluded, or low-contrast objects may be missed.
- Strong glare, rain, fog, snow, blur, and darkness can degrade every output.
- Lane markings may be absent or confused with road edges and reflections.
- Road segmentation does not encode obstacle occupancy.
- Monocular relative depth has no absolute physical scale.
- Static-image inference has no temporal context or motion estimate.
- Quality estimation can itself be wrong or poorly calibrated out of domain.

## License and Attribution

RoadScope-N5 is released under GNU AGPL-3.0. Review
[Third-Party Notices](THIRD_PARTY_NOTICES.md) before redistribution,
commercialization, or embedded deployment.
