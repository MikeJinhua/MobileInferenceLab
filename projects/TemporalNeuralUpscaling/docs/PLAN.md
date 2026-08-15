# Implementation Plan

Advance one verified task at a time. Prefer a small operator set and explicit tensors. Backend compatibility is measured during deployment, never inferred from PyTorch success.

## Phase 1 — Spatial SR

1. Minimal inference-only 2x tensor model, tests, and microbenchmark.
2. Deterministic image input/output and bilinear comparison with a documented lightweight dependency.
3. Export preparation and graph/operator inventory.
4. Repeatable PC quality/latency baselines on small and 540p-class inputs.

Initial model: `Conv2d -> ReLU -> Conv2d -> PixelShuffle(2)`, using float32 NCHW RGB. This is compact and avoids exotic operators. PixelShuffle avoids learned transposed convolution, but ExecuTorch/QNN support and performance must be verified in Phases 2/4. If it is unsupported or poorly delegated, document and compare a resize-plus-convolution alternative before changing the model.

## Later Phases

- **Phase 2:** pin compatible PyTorch/ExecuTorch versions, export, inspect graph/unsupported operators, compare outputs.
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
- Later only when needed: matched ExecuTorch/PyTorch; Android Studio, supported JDK/SDK/NDK, CMake, Ninja/platform tools; licensed QAIRT/QNN for the target; Vulkan tooling.

Do not silently install system-level or large dependencies. Select exact versions after choosing the target device and consulting compatibility requirements.
