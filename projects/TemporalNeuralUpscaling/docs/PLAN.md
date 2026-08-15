# Implementation Plan

Advance one verified task at a time. Prefer a small operator set and explicit tensors. Backend compatibility is measured during deployment, never inferred from PyTorch success.

## Phase 1 — Spatial SR

1. Minimal inference-only 2x tensor model, tests, and microbenchmark.
2. Deterministic image input/output and bilinear comparison with a documented lightweight dependency.
3. Export preparation and graph/operator inventory.
4. Repeatable PC quality/latency baselines on small and 540p-class inputs.

Initial model: `Conv2d -> ReLU -> Conv2d -> PixelShuffle(2)`, using float32 NCHW RGB. This is compact and avoids exotic operators. PixelShuffle avoids learned transposed convolution, but ExecuTorch/QNN support and performance must be verified in Phases 2/4. If it is unsupported or poorly delegated, document and compare a resize-plus-convolution alternative before changing the model.

### P1.3 export-readiness method

The detected PyTorch 2.0.0 environment predates the public `torch.export` API expected by current ExecuTorch workflows. P1.3 therefore uses a frozen TorchScript trace only as a local ATen operator inventory and shape/parity preflight; it does not create or claim an ExecuTorch-compatible artifact. The trace targets `SpatialSR2x.network`, the tensor-only inference core. Python input validation remains a host-side contract because trace cannot preserve its data-dependent Python guards. Phase 2 must select compatible PyTorch/ExecuTorch versions, use the supported export API, and revalidate operators, dynamic shapes, and delegation.

### P1.4 PC baseline method

Measure a deterministic synthetic RGB source at 64x64, 320x180, and 960x540 on CPU with one PyTorch thread. Report warmup, sample count, mean, median, P90, P95, minimum, and maximum for RGB-to-tensor preprocessing, Pillow bilinear 2x, model inference, tensor-to-RGB conversion, and an independently measured neural end-to-end path. Exclude PNG disk I/O and label results as a local PyTorch CPU baseline, not a mobile or sustained-performance result. Independent stage distributions are diagnostic and are not summed to replace the directly measured end-to-end distribution.

## Later Phases

- **Phase 2 (complete):** the documented `ExecuTorch 1.3.1 + PyTorch 2.12.0 + Python 3.10` environment passes minimal, static portable, static XNNPACK, and bounded-dynamic portable/XNNPACK export and runtime parity. The dynamic contract fixes batch/RGB/float32 and bounds height/width to `16..128`; its XNNPACK graph has three delegates and two portable reshape fallbacks. See the Phase 2 reports for exact evidence.
- **Phase 3:** minimal Android UI plus native C++ model/tensor path and CPU metrics.
- **Phase 4:** document licensed QAIRT/QNN setup, verify delegation/fallback, measure initialization/load/inference/memory.
- **Phase 5:** comparable FP32/FP16/INT8 artifacts and results.
- **Phase 6:** Vulkan image/frame textures, preprocess/composite, instrumentation before copy reduction.
- **Phase 7:** create a separate temporal spec; add history, motion vectors, and depth incrementally.
- **Phase 8:** publish comparison UI, architecture, deployment instructions, profiling, optimization history, and end-to-end results.

Planned Android native layout: `app/` for UI/lifecycle; `native/renderer/`, `inference/`, `shaders/`, `bridge/`, and `common/` for focused C++ modules.

## Environment Plan

- Now: Git, Python 3.10+, PyTorch 2.0+, and NumPy as a normal PyTorch dependency; standard-library `unittest` avoids adding pytest.
- Phase 1 Task 2: optionally Pillow in a project virtual environment for image I/O.
- Phase 2 portable/XNNPACK: separate `.venv-executorch` with Python 3.10, PyTorch 2.12.0, and ExecuTorch 1.3.1. Use native Windows for the initial pip/export path.
- Phase 4 QNN: WSL Ubuntu 22.04, compatible compiler/NDK, supported Snapdragon device, and licensed QAIRT/QNN SDK. Do not assume the native Windows export environment is the QNN build environment.
- Later only when needed: Android Studio, supported JDK/SDK/NDK, compatible CMake/Ninja/platform tools, QAIRT/QNN, and Vulkan tooling.

Do not silently install system-level or large dependencies. Select exact versions after choosing the target device and consulting compatibility requirements.
