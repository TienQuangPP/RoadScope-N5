# Deployment Guide

## Released Artifact

`roadscope_n5.onnx` is a self-contained opset-17 graph with a dynamic batch and a
fixed `512 x 512` spatial input. It produces five named outputs:

```text
detections
road
lane
relative_depth
confidence
```

The public runtime validates this contract before processing an image.

## CPU Inference

Install the default requirements and run:

```bash
python run.py --source input --output output --provider cpu
```

The default requirements install the CPU ONNX Runtime package.

## CUDA Inference

Install an ONNX Runtime GPU build compatible with the machine's CUDA and cuDNN
environment. Do not keep incompatible CPU and GPU runtime packages in the same
environment.

Then run:

```bash
python run.py --source input --output output --provider cuda
```

The command fails if CUDAExecutionProvider is unavailable. Confirm the provider
recorded in each JSON result instead of assuming that GPU execution occurred.

## ONNX Benchmark

After an editable install:

```bash
roadscope-n5-benchmark-onnx \
  --model roadscope_n5.onnx \
  --provider cpu \
  --warmup 20 \
  --iterations 200 \
  --output runs/onnx_benchmark.json
```

The report includes mean, median, p95, throughput, batch, provider, model size,
and input shape. Network execution time excludes image decoding, resize,
post-processing, rendering, and disk I/O.

## FP16 Conversion

Install training extras, then convert eligible internal tensors while keeping
float32 input/output types:

```bash
python -m roadscope_n5.tools.convert_fp16 \
  --model roadscope_n5.onnx \
  --output runs/roadscope_n5-fp16.onnx \
  --report runs/fp16-report.json
```

FP16 is not automatically faster on every CPU, GPU, or accelerator. Validate
the converted graph and benchmark it on the intended runtime.

## Static INT8 Quantization

Use a representative calibration set that covers day, night, weather,
occlusion, and traffic density:

```bash
python -m roadscope_n5.tools.quantize_onnx \
  --model roadscope_n5.onnx \
  --images calibration/images \
  --output runs/roadscope_n5-int8.onnx \
  --report runs/int8-quantization.json \
  --samples 512 \
  --imgsz 512
```

The tool performs static QDQ quantization. Calibration samples are selected at
evenly spaced positions after deterministic filename sorting.

Quantized support varies by execution provider. A smaller file does not prove
lower latency, and lower latency does not prove acceptable output agreement.

## Reduced-Precision Validation

Compare a candidate model against FP32 on a separate image set:

```bash
python -m roadscope_n5.tools.validate_int8 \
  --fp32 roadscope_n5.onnx \
  --candidate runs/roadscope_n5-int8.onnx \
  --images validation/images \
  --output runs/int8-validation.json \
  --samples 200 \
  --confidence 0.25 \
  --iou 0.50
```

The report compares detection count, matched IoU, score difference, road/lane
agreement, depth difference, quality difference, and runtime. Review each task;
one aggregate number can hide a serious regression.

## PyTorch Runtime Benchmark

For a training checkpoint:

```bash
python -m roadscope_n5.tools.benchmark_runtime \
  --weights yolov8n.pt \
  --checkpoint runs/train/last.pt \
  --output runs/pytorch_benchmark.json \
  --imgsz 512
```

This measures the PyTorch network and is not interchangeable with ONNX,
TensorRT, OpenVINO, NPU, or end-to-end application latency.

## Embedded Integration Checklist

1. Confirm supported operators and opset on the target runtime.
2. Preserve the exact RGB, NCHW, `[0, 1]`, letterbox preprocessing contract.
3. Preserve output names and class order.
4. Remove padding before resizing dense maps to the source image.
5. Validate every optimized model against FP32 on held-out data.
6. Report median and p95 end-to-end latency after warmup.
7. Record peak memory, model storage, power mode, temperature, and precision.
8. Test degraded visual conditions and empty scenes.
9. Add independent safety monitoring before any physical-system experiment.

## Benchmark Reporting

A reproducible latency claim should state:

- exact processor or accelerator;
- runtime and version;
- precision and quantization method;
- input resolution and batch;
- warmup and measured iteration counts;
- network-only or end-to-end timing;
- median, p95, and throughput;
- power/thermal mode;
- whether post-processing runs on CPU or accelerator.
