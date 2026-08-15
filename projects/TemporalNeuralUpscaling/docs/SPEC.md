# Specification

## Purpose and Boundaries

Build a public, reproducible Android neural-upscaling project demonstrating a real on-device inference pipeline. The focus is deployment, heterogeneous execution, layout and transfer, synchronization, profiling, memory, sustained performance, thermal behavior, and power where measurable.

The model is a vehicle for systems learning. Complex training, SOTA super-resolution research, Unreal Engine, and Unity are out of scope.

## Required Phases

1. **Spatial Neural SR:** float32 NCHW RGB input, lightweight PyTorch network, exactly 2x RGB output, PC inference, saved output in a later task, tests, and baseline benchmark.
2. **ExecuTorch:** export, unsupported-operator/partition record, and numerical parity with eager PyTorch.
3. **Android CPU:** minimal app with model load, tensor conversion, inference, display, and timing.
4. **Qualcomm QNN/NPU:** verified delegation on supported Snapdragon hardware, fallback detection, initialization/load/inference metrics.
5. **Quantization:** compare applicable FP32, FP16, INT8 size, latency, quality/numerics, memory, and compatibility.
6. **GPU Pipeline:** Vulkan texture input, preprocess, conversion, composite/display; measure readback/upload, copies, transfer, and synchronization.
7. **Temporal Neural Upscaling:** only after spatial stability; investigate history, motion vectors, depth, reprojection, ghosting, disocclusion, motion, thin geometry, and detail.
8. **Final Demo/Benchmark:** compare input, bilinear, and neural output (plus temporal if complete) with inference and end-to-end CPU/GPU/NPU/transfer/sync/memory results; include thermal, power, sustained results where practical.

## Benchmark and Release Requirements

Each result identifies hardware, backend/software, precision, shape, warmup, sample count, statistic, and inference-only/end-to-end scope. End-to-end reports separate preprocess, inference, transfer, synchronization, composite, and total, and explain bottlenecks, bounds, fallback, copies, and quantization benefit.

Document prerequisites, commands, versions, devices, limitations, third-party sources, and licenses. Never commit credentials, private/restricted code, proprietary SDK binaries, or large generated models.

## Completed Task Acceptance Criteria

### Phase 1 Task 1

- Minimal 2x PyTorch module runs on installed CPU PyTorch.
- `[N, 3, H, W]` produces `[N, 3, 2H, 2W]`.
- Invalid rank/channel count gives clear errors.
- Standard-library automated tests pass.
- CLI benchmark warms up and reports mean/median/min latency and throughput.
- No image I/O, training, Android, QNN, Vulkan, quantization, or temporal implementation.

### Phase 1 Task 2

- Convert any Pillow input to RGB and then to normalized float32 `[1, 3, H, W]`.
- Run the existing 2x spatial model with deterministic untrained weights.
- Convert the output tensor to an RGB image.
- Save the original RGB input, bilinear 2x baseline, and neural 2x output as PNG.
- Provide a deterministic, project-generated synthetic source with no external license dependency.
- Test RGB conversion, shapes, deterministic output, and lossless PNG save/load.
- Provide a CLI for synthetic or user-supplied input.
- Make no claim about the image quality of the untrained model.

### Phase 1 Task 3

- Produce a reproducible local inference graph and ATen operator inventory without starting ExecuTorch integration.
- Compare traced and eager output across representative batch and spatial shapes.
- Record dtype/layout and static/dynamic shape assumptions.
- Identify PixelShuffle, convolution lowering, and host-side validation as deployment checks.
- Clearly distinguish a TorchScript preflight from a supported `torch.export`/ExecuTorch artifact.
