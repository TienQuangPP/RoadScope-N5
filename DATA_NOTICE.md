# Data and Demo Image Notice

## Repository Scope

This repository contains the RoadScope-N5 source code, released ONNX model, and a
small qualitative demo gallery. It does not redistribute the training or
validation datasets.

## Demo Scenes

The five road-scene images under `input/` originate from BDD100K. The rendered
files under `output/` are derived visualizations produced from those images.
They are included only to demonstrate the public inference interface and are
not ground-truth annotations, benchmark evidence, or a safety claim.

The repository's AGPL-3.0 license applies to RoadScope-N5 code and the released
model artifact. It does not relicense third-party image content. Before
copying, redistributing, or using the demo scenes outside this repository,
review the current BDD100K terms and obtain any permission your use requires.

Official project: <https://github.com/bdd100k/bdd100k>

## Dataset Citation

When using BDD100K in research, cite:

F. Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask
Learning," in *Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition*, 2020.

## User-Supplied Data

RoadScope-N5 does not upload images or labels. Users are responsible for the
lawful collection, storage, processing, and redistribution of any data they
provide to the runtime or training pipeline.
