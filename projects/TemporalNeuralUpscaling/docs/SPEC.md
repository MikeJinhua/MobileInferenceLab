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

### Phase 1 Task 4

- Benchmark deterministic 64x64, 320x180, and 960x540 RGB inputs on the local PC CPU.
- Fix and report PyTorch thread count, warmup, iterations, environment, model size, and tensor contract.
- Separately measure preprocessing, bilinear 2x, inference, output conversion, and directly measured neural end-to-end latency.
- Report mean, median, P90, P95, minimum, and maximum in machine-readable JSON and Markdown.
- Exclude and disclose PNG I/O, mobile/device transfer, display, thermal, and power costs.
- Treat results as an engineering baseline for an untrained model, never as an SR quality claim.

### Phase 2 Task 1 — Toolchain Selection

- Select and date a mutually compatible Python, PyTorch, and ExecuTorch release set using official sources.
- Keep Phase 1 and ExecuTorch dependencies in separate virtual environments.
- Define portable, XNNPACK, dynamic-shape, operator-partition, and numerical-parity validation gates.
- Record native Windows prerequisites and the separate verified WSL/Linux path needed for future QNN work.
- Do not install ExecuTorch, build Android artifacts, or begin QNN integration in this planning task.

### Phase 2 Task 2 — Environment and Smoke Export

- Create an ignored `.venv-executorch` independent from Phase 1.
- Install the selected pinned versions and pass `pip check`.
- Export a minimal Add model through `torch.export`, XNNPACK lowering, and `.pte` serialization.
- Load the `.pte` with the packaged Python runtime and verify exact eager/export/runtime parity.
- Record artifact size and environment versions, but do not commit generated `.pte` files.
- Do not export the project SR model or start Android/QNN in this task.

### Phase 2 Task 3 — Static Portable SpatialSR2x Export

- Export the tensor-only `SpatialSR2x.network` with fixed float32 NCHW input `[1,3,64,64]`.
- Lower without a backend partitioner to validate ExecuTorch portable kernels.
- Serialize an ignored `.pte`, load it with the Python runtime, and execute `forward`.
- Require output `[1,3,128,128]`, float32 dtype, and eager/export/runtime numerical parity within documented tolerances.
- Record the `torch.export` operator inventory and artifact size.
- Do not evaluate XNNPACK delegation/fallback or dynamic shapes in this task.

### Phase 2 Task 4 — Static XNNPACK Delegation Analysis

- Lower fixed float32 `[1,3,64,64]` `SpatialSR2x.network` with `XnnpackPartitioner`.
- Inspect the post-partition graph and record delegate count, backend identity, and residual portable operators.
- Require one XNNPACK delegate and no portable operator fallback for the observed artifact.
- Serialize and execute the ignored `.pte`; require float32 `[1,3,128,128]` output and eager/runtime parity within `1e-5`.
- Scope the delegation claim to the exact static artifact and toolchain; do not infer QNN or dynamic-shape support.

### Phase 2 Task 5 — Bounded Dynamic Shapes

- Fix batch to 1, channels to 3, dtype to float32, and bound height/width independently to `16..128`.
- Export portable and XNNPACK `.pte` files from one `[1,3,64,64]` example.
- Execute lower-bound, upper-bound, odd, square, and non-square shapes and require exact 2x output shape.
- Require eager/runtime numerical parity within `1e-5` on both backends.
- Record XNNPACK delegate/fallback topology and representative out-of-range rejection.
- Treat mobile performance and the fixed-versus-dynamic deployment choice as Phase 3 measurements.

### Phase 3 Task 1 — Android Toolchain Readiness

- Select a versioned Android/Gradle/JDK/NDK contract compatible with ExecuTorch 1.3.1.
- Select the official Maven Central ExecuTorch AAR and XNNPACK CPU path for the first app.
- Add a non-mutating environment checker covering JDK, SDK, SDK Manager, platform tools, NDK, CMake, Ninja, and connected-device state.
- Record the actual host result and installation instructions without silently installing system-level tools.
- Do not scaffold or claim a buildable Android app until the strict readiness gate passes.

### Phase 3 Task 2 — Minimal Android App Build and Launch

- Separate the prebuilt-AAR build gate from the later native C++ toolchain gate.
- Create a minimal version-pinned Java Android application with the ExecuTorch 1.3.1 Maven dependency.
- Use only a launcher Activity and status UI; do not implement model inference yet.
- Build a debug APK, install it on the connected arm64 device, and verify launch/process state without a crash.
- Keep `local.properties`, Gradle caches, build output, APKs, and device identifiers out of Git.

### Phase 3 Task 3 — Static XNNPACK Device Inference

- Reproducibly generate the static XNNPACK `.pte` into an ignored Android asset path.
- Fail the Android build clearly when the generated model asset is absent.
- Copy the packaged asset to app-private storage, load it through ExecuTorch 1.3.1, and create a deterministic float32 `[1,3,64,64]` tensor.
- Execute the model on a connected arm64 phone and require `[1,3,128,128]` finite output.
- Require repeated-device inference determinism and checksum agreement with a PC eager reference within documented tolerances.
- Do not claim image quality or a performance benchmark; image conversion/display and separated timing remain P3.4.

### Phase 3 Task 4 — Android RGB Pipeline and CPU Timing

- Generate a deterministic, publicly reproducible RGB source without downloading media.
- Convert Bitmap RGB to normalized float32 NCHW, run the static XNNPACK model, and convert its 2x output back to RGB.
- Display the original 64x64 image, bilinear 128x128 baseline, and neural 128x128 output on the phone.
- Separately report model copy/load and preprocessing, inference, postprocessing, directly measured neural end-to-end, and bilinear timing.
- Use warmup and repeated samples and report mean, median, P95, minimum, and maximum.
- Treat random-weight output as pipeline validation only; do not claim SR image quality.

### Phase 4 Task 1 — QNN Toolchain and Device Readiness

- Select and document a version-matched ExecuTorch, QAIRT/QNN, Linux/WSL, Android NDK, device SoC, HTP architecture, and initial precision route.
- Add a non-mutating checker for WSL, NDK, QNN SDK layout, required Python support, connected arm64 device, and supported Qualcomm SoC.
- Inspect the installed QNN backend for current-model operator evidence while treating real partitioning as the compatibility gate.
- Keep proprietary SDK contents, generated models, runtime libraries, device serials, and local reports out of Git.
- Do not export or claim QNN/NPU execution until the strict readiness gate passes.

### Phase 4 Task 2 — Licensed QNN Build Environment

- Use WSL 2 with Ubuntu 22.04 x86_64 and keep all SDKs and virtual environments outside the repository.
- Install and verify Android NDK r26c, Python 3.10, PyTorch 2.12 CPU, ExecuTorch 1.3.1, and the required Qualcomm backend Python dependencies.
- Install the user-approved Qualcomm QNN Community SDK `2.37.0.250724` from the official source and require its public layout/version markers.
- Configure reproducible `QNN_SDK_ROOT`, `ANDROID_NDK_ROOT`, host-library paths, and compiler/tool discovery without committing local absolute paths.
- Pass dependency, backend-import, SDK-layout, device-SoC, and strict readiness gates before starting model lowering.
- Do not create a QNN model, Android runtime, or NPU performance claim in this environment-only task.

### Phase 4 Task 3 — Static HTP FP16 Lowering

- Export only the deterministic static float32 NCHW `[1,3,64,64]` tensor core for SM8550 / HTP v73 FP16.
- Require a QNN delegate and enumerate every portable fallback in the lowered edge graph.
- Serialize the generated `.pte` and machine-readable report only below ignored `results/`.
- Verify eager/export shape, dtype, finiteness, determinism, and numerical parity on the host.
- Do not claim Android runtime compatibility, HTP execution, NPU latency, or NPU numerical parity until P4.4 device evidence exists.
