# Task Ledger

Updated: 2026-08-15

Active phase: **Phase 1 — Spatial Neural SR**

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

Limitations: random weights do not establish visual quality; PC CPU timing excludes end-to-end costs; PixelShuffle backend support remains unverified; missing NumPy produces a non-fatal PyTorch warning.

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

## Remaining Work

- [ ] P1.3 export readiness/operator inventory (**next**).
- [ ] P1.4 PC baseline report.
- [ ] Phase 2 ExecuTorch.
- [ ] Phase 3 Android CPU.
- [ ] Phase 4 QNN/NPU.
- [ ] Phase 5 quantization.
- [ ] Phase 6 Vulkan pipeline.
- [ ] Phase 7 temporal reconstruction.
- [ ] Phase 8 final demo/end-to-end benchmark.
