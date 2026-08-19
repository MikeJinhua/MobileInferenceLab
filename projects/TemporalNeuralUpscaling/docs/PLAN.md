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
- **Phase 3 (complete):** the official Maven Central `executorch-android:1.3.1` AAR loads the reproducibly generated static XNNPACK model on an arm64 phone. The app implements deterministic RGB Bitmap-to-NCHW conversion, 2x inference, RGB output/display beside bilinear, and separated load/preprocess/inference/postprocess/direct-end-to-end timing. Direct C++ integration remains deferred until lower-level instrumentation or Vulkan interop requires it.
- **Phase 4:** target SM8550/HTP v73 through WSL Ubuntu 22.04, Android NDK 26c, ExecuTorch 1.3.1, and matching QAIRT/QNN `2.37.0.250724`. Start with the static 64x64 graph and HTP FP16 so QNN integration is separated from Phase 5 quantization. Build/use a QNN-capable runtime matching the AOT SDK, inspect delegation and every fallback, verify device parity, then measure initialization/load/inference/memory. Keep the working XNNPACK environment and fallback baseline intact.

### Phase 4 execution gates

1. **P4.2 environment:** verify WSL/Ubuntu, NDK, host tools, isolated Python dependencies, licensed QNN SDK layout/version, environment variables, connected SM8550, and a strict readiness report.
2. **P4.3 lowering (complete):** the static 64x64 HTP FP16 artifact lowers to one `QnnBackend` delegate with no portable fallback; host eager/export contract and determinism pass. The target-specific offline context is not executable on the x86 host.
3. **P4.4 device (complete):** the version-matched official Android QNN runner restores and executes the offline HTP context on SM8550. Determinism/parity, process-to-method-ready timing, a 20-sample warm inference distribution, and peak runner RSS are recorded against the qualified retained XNNPACK baseline.

### P4.4 runtime route

Use the official ExecuTorch `v1.3.1` source and Qualcomm `qnn_executor_runner` first, built externally in WSL with NDK r26c and QNN 2.37.0.250724. Deploy the runner, `libqnn_executorch_backend.so`, the minimal matching QNN HTP/v73 runtime-library set, and the ignored P4.3 `.pte` to an app-private-independent test directory under `/data/local/tmp`. This runner gate gives direct backend logs and isolates HTP/runtime correctness from Java/AAR packaging. Do not replace or disturb the working Maven/XNNPACK Android app. After device execution/parity and measurements pass, integrate QNN into the UI only as a later explicit task if it adds learning value.

The source checkout, build tree, Qualcomm libraries, runner binary, deployed files, outputs, and device identifiers stay outside Git. Repository scripts may describe and automate the build/deploy commands but must resolve external roots through environment variables or `$HOME` defaults.

Phase 4 outcome: the static random-weight model is fully delegated at AOT and physically executes through QNN HTP v73 on Snapdragon 8 Gen 2. The native tensor runner is the reproducible backend gate; QNN UI/image-pipeline integration is not required to prove HTP execution and may be considered separately after Phase 5 precision comparisons.
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
