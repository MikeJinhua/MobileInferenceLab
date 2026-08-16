# Task Ledger

Updated: 2026-08-16

Active phase: **Phase 3 — Android CPU**

## P1.1 — Minimal 2x tensor model and microbenchmark

Status: **complete (2026-08-15)**

- [x] Minimal PyTorch 2x RGB model.
- [x] Output-shape and invalid-input tests.
- [x] Inference-only CLI benchmark.
- [x] No later-phase scope.

Changed files: `model/__init__.py`, `model/spatial_sr.py`, `tests/test_spatial_sr.py`, `benchmark/benchmark_model.py`, plus repository/project bootstrap documentation.

Verification:

- `python -m unittest discover -s tests -v` — PASS, 5/5 tests.
- `python -m benchmark.benchmark_model --height 64 --width 64 --warmup 5 --iterations 20 --threads 1` — PASS.
- CPU/PyTorch 2.0.0+cpu result: output `(1, 3, 128, 128)` from `(1, 3, 64, 64)`; mean 0.646 ms, median 0.634 ms, minimum 0.549 ms, 1548.31 inferences/s.
- Result is a short inference-only smoke benchmark, not a stable performance claim.

Limitations: random weights do not establish visual quality; PC CPU timing excludes end-to-end costs; PixelShuffle backend support remains unverified. The original missing-NumPy warning was resolved in the project `.venv` during P1.2.

## P1.2 — Deterministic image inference and saved comparison

Status: **complete (2026-08-15)**

- [x] Define a project-generated sample policy and NumPy/Pillow dependencies.
- [x] Convert Pillow RGB image to normalized float32 NCHW tensor and back.
- [x] Run deterministic untrained-model inference and bilinear baseline.
- [x] Save RGB input, bilinear 2x, and neural 2x PNG files.
- [x] Test channels, shapes, determinism, and image save/load.
- [x] Add a CLI for synthetic and user-supplied images.

Changed files: `requirements-phase1.txt`, `assets/README.md`, `pipeline/__init__.py`, `pipeline/image_pipeline.py`, `tools/__init__.py`, `tools/run_sr_image.py`, `tests/test_image_pipeline.py`, `README.md`, and SDD/environment documents.

Verification:

- `.venv\Scripts\python.exe -m unittest discover -s tests -v` — PASS, 9/9 tests (P1.1 and P1.2).
- `.venv\Scripts\python.exe -m tools.run_sr_image --synthetic-size 96x64 --output-dir results\p1_2` — PASS.
- `.venv\Scripts\python.exe -m tools.run_sr_image --input results\p1_2\input.png --output-dir results\p1_2_from_input` — PASS.
- Generated RGB PNGs: `input.png` at 96x64, `bilinear_2x.png` at 192x128, and `neural_2x.png` at 192x128.

Limitations: neural output uses deterministic random weights solely to validate the end-to-end image path; it is not an SR quality result. Generated `results/` files remain local and ignored by Git.

## P1.3 — Export readiness and operator inventory

Status: **complete (2026-08-15)**

- [x] Detect available PyTorch export/graph-capture capabilities.
- [x] Trace the tensor-only inference core for a local ATen inventory.
- [x] Verify eager/trace shape and numerical parity across representative shapes.
- [x] Record layout, dtype, dynamic-shape limits, operator risks, and the Phase 2 gate.
- [x] Avoid producing or claiming an ExecuTorch/mobile artifact.

Changed files: `analysis/__init__.py`, `analysis/export_readiness.py`, `tools/inspect_export_readiness.py`, `tests/test_export_readiness.py`, `docs/EXPORT_READINESS.md`, `README.md`, `docs/SPEC.md`, `docs/PLAN.md`, `docs/ENVIRONMENT.md`, and `docs/TASKS.md`.

Verification:

- `.venv\Scripts\python.exe -m tools.inspect_export_readiness --output docs\EXPORT_READINESS.md` — PASS.
- `.venv\Scripts\python.exe -m unittest discover -s tests -v` — PASS, 11/11 tests.
- Inventory: `aten::_convolution` x2, `aten::relu` x1, `aten::pixel_shuffle` x1.
- Eager/trace maximum absolute difference: 0 for `(1,3,16,24)`, `(1,3,17,21)`, and `(2,3,32,40)` inputs.

Limitations: installed PyTorch 2.0.0 has no public `torch.export`; TorchScript is only a preflight. Dynamic shapes are observed, not formally declared. ExecuTorch support and QNN delegation—especially PixelShuffle—remain unverified.

## P1.4 — PC baseline report

Status: **complete (2026-08-15)**

- [x] Benchmark deterministic 64x64, 320x180, and 960x540 inputs.
- [x] Fix CPU execution to one PyTorch thread and record environment/methodology.
- [x] Measure preprocessing, bilinear, inference, output conversion, and direct neural end-to-end paths.
- [x] Report mean, median, P90, P95, minimum, and maximum.
- [x] Generate committed Markdown and JSON reports and document exclusions.

Changed files: `benchmark/pipeline_benchmark.py`, `tools/run_pc_baseline.py`, `tests/test_pipeline_benchmark.py`, `docs/PC_BASELINE.md`, `docs/PC_BASELINE.json`, `README.md`, `docs/SPEC.md`, `docs/PLAN.md`, and `docs/TASKS.md`.

Verification:

- `.venv\Scripts\python.exe -m unittest discover -s tests -v` — PASS, 13/13 tests.
- `.venv\Scripts\python.exe -m tools.run_pc_baseline --sizes 64x64 320x180 960x540 --warmup 5 --iterations 20 --threads 1` — PASS.
- `python -m json.tool docs\PC_BASELINE.json` — PASS.
- Median inference / direct neural end-to-end: 0.481 / 0.783 ms at 64x64; 12.175 / 17.939 ms at 320x180; 102.309 / 160.689 ms at 960x540.

Limitations: local eager PyTorch CPU result on an untrained model; PNG I/O, display, transfer, synchronization, sustained load, thermal, power, and mobile execution are excluded. Twenty samples per stage form a development baseline, not a publication-grade performance study.

## P2.1 — Select the ExecuTorch toolchain

Status: **complete (2026-08-15)**

- [x] Review official installation, export, Windows, Android, and Qualcomm guidance.
- [x] Select Python 3.10 + PyTorch 2.12.0 + ExecuTorch 1.3.1.
- [x] Define separate `.venv-executorch` and preserve the Phase 1 environment.
- [x] Define portable, XNNPACK, parity, partition/fallback, and static-before-dynamic gates.
- [x] Record native Windows readiness and defer WSL/QNN setup.
- [x] Install no packages or SDKs during this planning task.

Changed files: `.gitignore`, `requirements-executorch.txt`, `docs/EXECUTORCH_TOOLCHAIN.md`, `README.md`, `docs/SPEC.md`, `docs/PLAN.md`, `docs/ENVIRONMENT.md`, and `docs/TASKS.md`.

Verification:

- Official stable documentation line checked: ExecuTorch 1.3.
- Release metadata checked: release/1.3 requires PyTorch 2.12-era tooling; selected stable PyTorch 2.12.0 and ExecuTorch patch 1.3.1.
- Host inspection: Visual Studio 2022 Enterprise/MSVC detected; Developer PowerShell available; Clang component not detected; WSL Ubuntu 22.04 not available.
- Existing Phase 1 test suite remains the regression gate; this task changes no runtime code.

Limitations: the toolchain is selected but not installed or smoke-tested. ExecuTorch 1.4.1 was released one day before this decision but is not yet the official documented stable line; revisit only through a deliberate upgrade task.

## P2.2 — Install toolchain and run minimal export

Status: **complete (2026-08-15)**

- [x] Create ignored `.venv-executorch` without reusing Phase 1 packages.
- [x] Install PyTorch 2.12.0+cpu and ExecuTorch 1.3.1; pass `pip check`.
- [x] Export the minimal Add model with `torch.export` and XNNPACK.
- [x] Serialize a 1,584-byte `.pte`, load it in the packaged runtime, and execute `forward`.
- [x] Verify eager/export/runtime result parity with maximum absolute difference 0.
- [x] Add a conditional automated smoke test and document the Windows bundled-`flatc.exe` workaround.

Changed files: `tools/smoke_executorch.py`, `tests/test_executorch_smoke.py`, `docs/EXECUTORCH_SMOKE.md`, `README.md`, `docs/SPEC.md`, `docs/ENVIRONMENT.md`, `docs/EXECUTORCH_TOOLCHAIN.md`, and `docs/TASKS.md`.

Verification:

- `.venv-executorch\Scripts\python.exe -m pip check` — PASS.
- `.venv-executorch\Scripts\python.exe -m tools.smoke_executorch` — PASS; XNNPACK `.pte` size 1,584 bytes; output 4.0; parity differences 0.
- `.venv-executorch\Scripts\python.exe -m unittest discover -s tests -v` — PASS, 14/14 tests.
- `.venv\Scripts\python.exe -m unittest discover -s tests -v` — PASS, 13 tests plus the expected ExecuTorch-only skip.

Limitations: this validates only a minimal Add graph. The Windows wheel needs `FLATC_EXECUTABLE` pointed at its bundled `flatc.exe`; upstream deprecation/CPU-probe warnings remain visible. No SR model, Android, QNN, Vulkan, or dynamic-shape work was performed.

## P2.3 — Static portable SpatialSR2x export

Status: **complete (2026-08-15)**

- [x] Export `SpatialSR2x.network` with static float32 `[1,3,64,64]` input.
- [x] Lower without a delegate and serialize a portable `.pte`.
- [x] Load and execute `forward` with the packaged Python runtime.
- [x] Verify float32 `[1,3,128,128]` output and numerical parity.
- [x] Record the `torch.export` operator inventory and artifact size.
- [x] Add a conditional automated test; perform no XNNPACK model analysis or dynamic export.

Changed files: `tools/executorch_utils.py`, `tools/smoke_executorch.py`, `tools/export_spatial_sr_portable.py`, `tests/test_spatial_sr_executorch.py`, `docs/SPATIAL_SR_PORTABLE_EXPORT.md`, `docs/EXPORT_READINESS.md`, `docs/EXECUTORCH_TOOLCHAIN.md`, `README.md`, `docs/SPEC.md`, and `docs/TASKS.md`.

Verification:

- `.venv-executorch\Scripts\python.exe -m tools.export_spatial_sr_portable` — PASS.
- Portable `.pte`: 12,720 bytes; input `[1,3,64,64]`; output `[1,3,128,128]`; float32.
- Operators: `aten.conv2d.default` x2, `aten.relu.default` x1, `aten.pixel_shuffle.default` x1.
- Eager/export max difference 0; eager/runtime max difference `2.086162567138672e-7`; mean difference `2.5059028629925706e-8`.
- ExecuTorch environment tests: PASS, 15/15.
- Phase 1 environment tests: PASS, 13 tests plus 2 expected ExecuTorch-only skips.

Limitations: the artifact is static-shape and portable-only. XNNPACK/QNN delegation, fallback, dynamic shapes, Android execution, image quality, and mobile performance remain unverified.

## P2.4 — Static XNNPACK delegation and fallback analysis

Status: **complete (2026-08-15)**

- [x] Lower static `SpatialSR2x.network` with `XnnpackPartitioner`.
- [x] Inspect the post-partition graph and backend identity.
- [x] Verify one `XnnpackBackend` delegate and zero portable fallback operators.
- [x] Serialize and execute the XNNPACK `.pte` with eager/runtime parity.
- [x] Add a conditional automated delegation/runtime test.

Changed files: `tools/export_spatial_sr_xnnpack.py`, `tests/test_spatial_sr_xnnpack.py`, `docs/SPATIAL_SR_XNNPACK_EXPORT.md`, `docs/EXPORT_READINESS.md`, `README.md`, `docs/SPEC.md`, and `docs/TASKS.md`.

Verification:

- `.venv-executorch\Scripts\python.exe -m tools.export_spatial_sr_xnnpack` — PASS.
- XNNPACK `.pte`: 12,592 bytes; one `XnnpackBackend` delegate; no portable fallback operators.
- Input `[1,3,64,64]`; output `[1,3,128,128]`; float32.
- Eager/export max difference 0; eager/runtime max difference `2.980232238769531e-7`; mean difference `2.542216748224746e-8`.
- ExecuTorch environment tests: PASS, 16/16.
- Phase 1 environment tests: PASS, 13 tests plus 3 expected ExecuTorch-only skips.

Limitations: the claim applies only to the static PC artifact and ExecuTorch 1.3.1. Dynamic shapes, Android execution/performance, QNN delegation, quantization, and image quality remain unverified.

## P2.5 — Bounded dynamic shapes

Status: **complete (2026-08-15)**

- [x] Declare batch-1 RGB float32 with independent height/width bounds `16..128`.
- [x] Export portable and XNNPACK bounded-dynamic `.pte` files.
- [x] Execute boundary, odd, square, and non-square inputs with correct 2x outputs.
- [x] Verify eager/runtime parity and representative out-of-range export rejection.
- [x] Record dynamic XNNPACK delegate and portable fallback topology.

Changed files: `tools/export_spatial_sr_dynamic.py`, `tests/test_spatial_sr_dynamic.py`, `docs/SPATIAL_SR_DYNAMIC_EXPORT.md`, `docs/EXPORT_READINESS.md`, `README.md`, `docs/SPEC.md`, `docs/PLAN.md`, and `docs/TASKS.md`.

Verification:

- `.venv-executorch\Scripts\python.exe -m tools.export_spatial_sr_dynamic` — PASS.
- Five shapes from `[1,3,16,16]` through `[1,3,128,128]` pass portable and XNNPACK runtime parity.
- Portable `.pte`: 13,616 bytes; maximum eager/runtime difference `2.682209014892578e-7`.
- XNNPACK `.pte`: 14,640 bytes; maximum eager/runtime difference `2.8312206268310547e-7`.
- Dynamic XNNPACK graph: three delegates, two portable `view_copy` fallbacks, and symbolic shape arithmetic.
- Export contract rejects representative below-minimum and above-maximum shapes.
- ExecuTorch environment tests: PASS, 17/17.
- Phase 1 environment tests: PASS, 13 tests plus 4 expected ExecuTorch-only skips.

Limitations: the contract supports only batch 1, RGB float32, and dimensions `16..128`. Tests run on the packaged PC runtime; Android performance, QNN, quantization, image quality, and arbitrary shapes remain unverified.

Phase 2 outcome: **complete**. Static export offers full single-partition XNNPACK delegation; bounded dynamic export offers verified size flexibility with reshape fallback. Phase 3 must measure both on Android before choosing the app artifact.

## P3.1 — Android toolchain decision and readiness audit

Status: **complete (2026-08-16)**

- [x] Select Maven Central `executorch-android:1.3.1` with XNNPACK for the first CPU app.
- [x] Pin JDK 17, AGP 8.9.x, Gradle 8.11.1, API/Build Tools 35, and NDK r28c.
- [x] Add a non-mutating Android prerequisite checker and unit tests.
- [x] Record the actual host tools and connected-device state.
- [x] Install no system-level packages and create no unverified Android scaffold.

Changed files: `tools/check_android_environment.py`, `tests/test_android_environment.py`, `docs/ANDROID_TOOLCHAIN.md`, `README.md`, `docs/SPEC.md`, `docs/PLAN.md`, and `docs/TASKS.md`.

Verification:

- `.venv\Scripts\python.exe -m tools.check_android_environment --output results\p3_1\android_environment.json` — PASS (audit completed).
- Found: ADB 36.0.0 and CMake; no connected Android device.
- Missing: JDK, Android SDK/SDK Manager, Android NDK, and Ninja.
- `ready_for_android_build: false`; this is the expected evidence from the current host, not a successful build claim.
- Strict readiness gate returns the expected exit code 1 while prerequisites are missing.
- Phase 1 environment tests: PASS, 16 passed plus 4 expected ExecuTorch-only skips.
- ExecuTorch environment tests: PASS, 20/20.

Limitations: P3.1 establishes the integration route and readiness gate only. No Android project, APK, device inference, or timing result exists yet.

## Remaining Work

- [ ] P3.2: install/verify the pinned Android prerequisites, then scaffold and build the minimal app (**next; environment gate currently not ready**).
- [ ] P3.3: package the reproducibly generated static XNNPACK `.pte` and implement device inference.
- [ ] P3.4: add RGB conversion/display and separated Android CPU timing.
- [ ] Phase 4 QNN/NPU.
- [ ] Phase 5 quantization.
- [ ] Phase 6 Vulkan pipeline.
- [ ] Phase 7 temporal reconstruction.
- [ ] Phase 8 final demo/end-to-end benchmark.
