# Task Ledger

Updated: 2026-08-20

Active phase: **Phase 4 — Qualcomm QNN/NPU**

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

## P3.2 — Minimal Android app build and device launch

Status: **complete (2026-08-17)**

- [x] Discover and adopt the installed Quail 3/JBR 25/API 37 toolchain.
- [x] Separate the prebuilt-AAR readiness gate from future native C++ prerequisites.
- [x] Create a minimal Java launcher app using Maven `executorch-android:1.3.1`.
- [x] Build the debug APK with AGP 9.3.0 and Gradle 9.5.0.
- [x] Install and cold-launch the app on the connected arm64 Samsung phone.
- [x] Add static Android scaffold regression tests.

Changed files: `android/settings.gradle.kts`, `android/build.gradle.kts`, `android/gradle.properties`, `android/gradlew`, `android/gradlew.bat`, `android/gradle/wrapper/*`, `android/app/build.gradle.kts`, `android/app/src/main/AndroidManifest.xml`, `android/app/src/main/java/com/mike/mobileinferencelab/temporalsr/MainActivity.java`, `android/app/src/main/res/values/*`, `tools/check_android_environment.py`, `tests/test_android_scaffold.py`, `docs/ANDROID_TOOLCHAIN.md`, `README.md`, `docs/SPEC.md`, and `docs/TASKS.md`.

Verification:

- Android readiness: AAR build and connected-device gates PASS; native C++ gate remains false because NDK/Ninja are absent.
- `android\gradlew.bat -p android :app:assembleDebug` — PASS.
- Debug APK: 27,928,417 bytes; generated under ignored `android/app/build/`.
- `adb install -r ...\app-debug.apk` — Success.
- `adb shell am start -W .../.MainActivity` — status OK, cold launch 412 ms.
- Device process and focused Activity confirmed; no `AndroidRuntime` fatal exception observed.
- Phase 1 environment tests: PASS, 18 passed plus 4 expected ExecuTorch-only skips.
- ExecuTorch environment tests: PASS, 22/22.
- Incremental Android rebuild: PASS in 1 second, 35 tasks up-to-date.

Limitations: the app currently displays only a Phase 3 status message. It resolves/packages the ExecuTorch AAR but does not load a `.pte`, create tensors, run inference, display SR output, or report inference timing. The launch time is not an ML benchmark.

## P3.3 — Static XNNPACK device inference

Status: **complete (2026-08-17)**

- [x] Add a reproducible Android model-asset preparation command.
- [x] Add a Gradle gate that rejects a missing generated `.pte`.
- [x] Copy the asset to app-private storage and load it with the ExecuTorch AAR.
- [x] Run deterministic `[1,3,64,64]` input twice on the connected arm64 phone.
- [x] Validate output shape, finiteness, repeat determinism, and PC eager checksum agreement.
- [x] Keep the generated model, APK, report, and device identifiers out of Git.

Changed files: `tools/prepare_android_model.py`, `android/app/build.gradle.kts`, `android/app/src/main/java/com/mike/mobileinferencelab/temporalsr/MainActivity.java`, `android/app/src/main/java/com/mike/mobileinferencelab/temporalsr/SpatialSrRunner.java`, `android/app/src/main/res/values/strings.xml`, `tests/test_android_scaffold.py`, `docs/ANDROID_DEVICE_INFERENCE.md`, `README.md`, `docs/SPEC.md`, and `docs/TASKS.md`.

Verification:

- `.venv-executorch\Scripts\python.exe -m tools.prepare_android_model` — PASS.
- Static XNNPACK asset: 12,592 bytes; one delegate; zero fallback; SHA-256 `d2edf8943f59b463fae7fcdc1eee147e604d45a2cc8e42dfe9f6c2dcd0cc4b13`.
- `android\gradlew.bat -p android :app:assembleDebug` — PASS; model-asset gate executed.
- APK reinstall and Activity launch — PASS.
- Device output `[1,3,128,128]`; finite min/max `-0.429966062` / `0.347079456`.
- Repeated-device maximum difference: 0.
- Device checksum `501.158083683`; PC eager reference `501.158030012808`; difference `5.36697771e-05` (tolerance `1e-3`).
- Phase 1 environment tests: PASS, 19 passed plus 4 expected ExecuTorch-only skips.
- ExecuTorch environment tests: PASS, 23/23.
- Final APK build, reinstall, cold launch, and `status=PASS` logcat gate — PASS.

Limitations: deterministic random weights validate only the deployment pipeline. This is not an image-quality result or performance benchmark; Bitmap conversion, output display, separated latency statistics, sustained load, thermal, power, QNN, and Vulkan remain unverified.

## P3.4 — Android RGB image pipeline and CPU timing

Status: **complete (2026-08-17)**

- [x] Generate a deterministic, project-owned 64x64 RGB image on device.
- [x] Convert Bitmap RGB to float32 NCHW and neural output back to clamped RGB.
- [x] Display original, Android bilinear 2x, and ExecuTorch/XNNPACK neural 2x images.
- [x] Warm up and separately measure model copy/load, preprocess, inference, postprocess, direct neural E2E, and bilinear.
- [x] Report mean, median, P95, minimum, and maximum over 20 samples.
- [x] Build, reinstall, cold-launch, inspect the UI, and require a device `status=PASS` log.

Changed files: `android/app/src/main/java/com/mike/mobileinferencelab/temporalsr/ImageSrPipeline.java`, `android/app/src/main/java/com/mike/mobileinferencelab/temporalsr/MainActivity.java`, `android/app/src/main/res/values/strings.xml`, `tests/test_android_scaffold.py`, `docs/ANDROID_IMAGE_PIPELINE.md`, `README.md`, `docs/SPEC.md`, `docs/PLAN.md`, and `docs/TASKS.md`.

Verification:

- `android\gradlew.bat -p android :app:assembleDebug` — PASS, 36 tasks.
- Phase 1 environment tests — PASS, 20 passed plus 4 expected ExecuTorch-only skips.
- ExecuTorch environment tests — PASS, 24/24.
- APK reinstall and Activity cold launch — PASS; launch `TotalTime` 198 ms is not an ML metric.
- Device UI — PASS; all three labeled RGB images were visually checked at 64x64 input and 128x128 outputs.
- Device XNNPACK inference median/P95: 0.372/1.044 ms.
- Direct neural E2E median/P95: 1.018/8.002 ms; bilinear median/P95: 0.124/0.142 ms.
- One-time model copy/load: 2.396/9.558 ms. Full distributions are in `docs/ANDROID_IMAGE_PIPELINE.md`.

Limitations: deterministic random weights validate the connected image pipeline only and do not establish SR quality. This short run excludes UI rendering, sustained load, thermal, power, memory, QNN, Vulkan, and PNG I/O. Generated model/APK/screenshots/logs and device identifiers remain ignored.

Phase 3 outcome: **complete**. A reproducible static XNNPACK model now executes from RGB input to visible 2x RGB output on a physical Android phone with separated CPU timing.

## P4.1 — QNN toolchain and device readiness

Status: **complete (2026-08-17)**

- [x] Select the SM8550 / HTP v73 target and a version-matched QNN deployment route.
- [x] Pin WSL Ubuntu 22.04, Android NDK 26c, ExecuTorch 1.3.1, and QNN `2.37.0.250724` for the first gate.
- [x] Choose static 64x64 HTP FP16 before Phase 5 quantization.
- [x] Add a non-mutating host/device readiness checker and unit tests.
- [x] Confirm the installed backend contains QNN PixelShuffle/DepthToSpace handling, without claiming delegation.
- [x] Record missing prerequisites and public-repository safety boundaries.

Changed files: `tools/check_qnn_environment.py`, `tests/test_qnn_environment.py`, `docs/QNN_READINESS.md`, `README.md`, `docs/SPEC.md`, `docs/PLAN.md`, and `docs/TASKS.md`.

Verification:

- `.venv-executorch\Scripts\python.exe -m tools.check_qnn_environment --output results\p4_1\qnn_environment.json` — audit PASS; overall readiness false.
- Device checks — PASS: arm64-v8a, Qualcomm `SM8550`, supported schema target HTP v73.
- ExecuTorch Qualcomm backend — present; current-model PixelShuffle has explicit backend handling in installed source.
- Missing gates — WSL Ubuntu 22.04, Android NDK, `QNN_SDK_ROOT`/valid SDK layout, and `py-cpuinfo`.
- `--strict` — expected exit code 1 while prerequisites are absent.
- Phase 1 environment tests — PASS, 24 passed plus 4 expected ExecuTorch-only skips.
- ExecuTorch environment tests — PASS, 28/28.

Limitations: no QNN SDK was installed, no model was lowered, no QNN-capable Android runtime was built, and no NPU execution/delegation/performance claim exists. Installed-source operator handlers are only preflight evidence.

## P4.2 — Licensed QNN build environment

Status: **complete (2026-08-20)**

- [x] Install and initialize WSL 2 with Ubuntu 22.04 x86_64.
- [x] Install Ubuntu build essentials, Python 3.10, CMake, Ninja, Git, curl, unzip, and zip.
- [x] Install and SHA-1 verify Android NDK r26c (`26.2.11394342`) outside the repository.
- [x] Create isolated `$HOME/.venvs/tnu-qnn` and install PyTorch 2.12.0 CPU plus ExecuTorch 1.3.1 core.
- [x] Install NumPy, Pillow, `py-cpuinfo`, and requests; verify the Qualcomm backend imports without auto-install side effects.
- [x] Obtain explicit user approval for Qualcomm QNN Community SDK `2.37.0.250724`.
- [x] Finish official QNN host prerequisite and SDK download/extraction.
- [x] Verify `QNN_README.txt`, `sdk.yaml`, SDK version, host libraries, and Android target libraries.
- [x] Install all declared ExecuTorch Python dependencies and pass `pip check`.
- [x] Configure reproducible WSL `QNN_SDK_ROOT`, `ANDROID_NDK_ROOT`, and host-library environment.
- [x] Extend the readiness checker to evaluate the real WSL toolchain/SDK rather than Windows-only SDK paths.
- [x] Pass the strict QNN readiness gate with the connected SM8550 device.
- [x] Run both project test environments and update the readiness report with exact results.
- [x] Confirm that no SDK binary, generated model, cache, local path, or device serial is tracked by Git.

Current evidence:

- WSL reports Ubuntu 22.04.1 LTS, x86_64, and WSL version 2.
- NDK r26c archive SHA-1 matched Android's published `7faebe2ebd3590518f326c82992603170f07c96e` and reports revision `26.2.11394342`.
- Linux environment imports ExecuTorch 1.3.1 Qualcomm backend with PyTorch `2.12.0+cpu`.
- Official QNN bootstrap completed and reported QNN SDK `2.37.0.250724` ready under the ignored per-user cache.
- `sdk.yaml` reports QAIRT `2.37.0`, build ID `250724175447_124859`, Ubuntu 22.04, and Android NDK r26c.
- SDK inspection found `QNN_README.txt`, `sdk.yaml`, x86_64 host `libQnnHtp.so`/`libQnnSystem.so`, aarch64 Android target libraries, and the HTP v73 skeleton library required by SM8550.
- The complete ExecuTorch 1.3.1 dependency set is installed in the isolated WSL virtual environment; `python -m pip check` reports no broken requirements.
- `tools/qnn_env.sh` derives SDK/NDK paths from `$HOME`, validates public markers, and adds only the QNN x86_64 host-library directory to `LD_LIBRARY_PATH`.
- The extended strict checker validates the actual WSL host tools, NDK, QNN version/libraries, virtual environment, `pip check`, Qualcomm backend import, and connected SM8550; all required checks pass.

Changed files: `tools/qnn_env.sh`, `tools/check_qnn_environment.py`, `tests/test_qnn_environment.py`, `docs/QNN_READINESS.md`, `README.md`, and `docs/TASKS.md`.

Verification:

- WSL `python -m pip check` — PASS; no broken requirements.
- `.venv-executorch\Scripts\python.exe -m tools.check_qnn_environment --strict --output results\p4_2\qnn_environment.json` — PASS; every required check true, connected target SM8550 / HTP v73.
- `.venv\Scripts\python.exe -m unittest discover -s tests -v` — PASS, 26 passed plus 4 expected ExecuTorch-only skips.
- `.venv-executorch\Scripts\python.exe -m unittest discover -s tests -v` — PASS, 30/30.
- `git diff --check`, ignored-report check, and tracked-file scan — PASS; no SDK/runtime library, generated model/report, local properties, or device serial is tracked.

Limitations: the environment is ready, but no QNN model has been lowered, no Android QNN runtime exists, and no HTP execution or NPU performance result is claimed.

## P4.3 — Static HTP FP16 lowering and delegation analysis

Status: **pending**

- [ ] Add a reproducible static `[1,3,64,64]` QNN HTP FP16 export command using the pinned WSL environment.
- [ ] Lower the current `SpatialSR2x.network` and serialize only an ignored generated artifact.
- [ ] Inspect and record QNN delegate count, backend identity, and every portable fallback operator.
- [ ] Verify output dtype/shape, finiteness, determinism, and eager/export parity where the available host runtime permits.
- [ ] Add automated tests for the export contract and delegation report.
- [ ] Update documentation and confirm proprietary/generated artifacts remain untracked.

## Remaining Work

- [x] P3.4: add RGB conversion/display and separated Android CPU timing.
- [x] P4.2: install/verify the licensed QNN build environment and pass the strict gate.
- [ ] P4.3: lower the static model and verify QNN delegation/fallback (**next**).
- [ ] P4.4: run on-device HTP inference and measure initialization/load/inference/memory.
- [ ] Phase 5 quantization.
- [ ] Phase 6 Vulkan pipeline.
- [ ] Phase 7 temporal reconstruction.
- [ ] Phase 8 final demo/end-to-end benchmark.
