# Android RGB Image Pipeline and CPU Timing

P3.4 completes the first visible on-device super-resolution pipeline. The Android app generates a deterministic, project-owned 64x64 RGB test image in memory, converts it to float32 NCHW, runs the static 2x XNNPACK model through ExecuTorch, converts the output back to RGB, and displays the input, bilinear 2x baseline, and neural 2x output.

The model has deterministic random weights. The neural image proves that the pipeline is connected; it is not evidence of useful SR quality.

## Device and method

- Device: Samsung SM-S916U, Android 13, arm64-v8a.
- Runtime/backend: ExecuTorch Android 1.3.1, XNNPACK CPU, float32.
- Shapes: `[1,3,64,64]` input and `[1,3,128,128]` output.
- Warmup: 5 model executions; measured samples: 20.
- Clock: Android monotonic elapsed realtime in nanoseconds.
- Neural E2E is measured directly around Bitmap-to-tensor, inference, and tensor-to-Bitmap. It excludes model copy/load and UI rendering.
- Bilinear is measured separately with Android `Bitmap.createScaledBitmap`.

## Observed development result

All values are milliseconds from one cold app launch on 2026-08-17.

| Stage | Mean | Median | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|
| RGB preprocess | 0.305 | 0.299 | 0.513 | 0.080 | 1.010 |
| XNNPACK inference | 0.524 | 0.372 | 1.044 | 0.309 | 2.262 |
| RGB postprocess | 1.204 | 0.352 | 5.396 | 0.321 | 10.482 |
| Direct neural E2E | 2.037 | 1.018 | 8.002 | 0.724 | 12.050 |
| Bilinear 2x | 0.129 | 0.124 | 0.142 | 0.111 | 0.246 |

One-time model asset copy was 2.396 ms and model load was 9.558 ms. The complete development run, including setup, warmup, all neural samples, and bilinear samples, was 61.188 ms; it is not a per-frame metric.

## Validation and limitations

The app was rebuilt, reinstalled, cold-launched, and emitted `status=PASS`. The UI was visually checked on the connected phone and displayed all three labeled RGB images at the expected sizes. Generated APKs, model assets, screenshots, logs, and device identifiers remain ignored and are not repository artifacts.

This is a short CPU development benchmark. It does not measure sustained load, thermal behavior, energy, memory, display composition, PNG I/O, GPU/NPU execution, or image quality. Outliers in Bitmap allocation/conversion are visible in the P95 and maximum values.
