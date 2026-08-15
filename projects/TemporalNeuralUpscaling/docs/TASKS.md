# Task Ledger

Updated: 2026-08-15

Active phase: **Phase 1 — Spatial Neural SR complete; Phase 2 not started**

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

## Remaining Work

- [ ] Phase 2.1: select and document a compatible PyTorch/ExecuTorch toolchain (**next**).
- [ ] Phase 3 Android CPU.
- [ ] Phase 4 QNN/NPU.
- [ ] Phase 5 quantization.
- [ ] Phase 6 Vulkan pipeline.
- [ ] Phase 7 temporal reconstruction.
- [ ] Phase 8 final demo/end-to-end benchmark.
