# PC Baseline Report

This is a local Phase 1 CPU baseline for pipeline engineering. `SpatialSR2x` has deterministic random weights, so these measurements do not represent SR image quality or a trained final model.

## Environment

- OS: `Windows 11 10.0.22621`
- CPU: `12th Gen Intel(R) Core(TM) i5-12490F`
- Logical CPUs: 12
- Python: `3.10.6`
- PyTorch: `2.0.0+cpu`
- Device / dtype / layout: `cpu` / `float32` / `NCHW RGB`
- PyTorch threads: 1
- Model parameters: 2,188 (8,752 FP32 bytes)
- Warmup / measured iterations per stage: 5 / 20

## Results

| Input | Stage | Mean ms | Median ms | P90 ms | P95 ms | Min ms | Max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x64 | `rgb_to_tensor` | 0.028 | 0.027 | 0.032 | 0.033 | 0.026 | 0.033 |
| 64x64 | `bilinear_2x` | 0.089 | 0.087 | 0.087 | 0.091 | 0.086 | 0.128 |
| 64x64 | `model_inference` | 0.490 | 0.481 | 0.512 | 0.516 | 0.472 | 0.561 |
| 64x64 | `tensor_to_rgb` | 0.238 | 0.217 | 0.296 | 0.339 | 0.211 | 0.341 |
| 64x64 | `neural_end_to_end` | 0.786 | 0.783 | 0.808 | 0.810 | 0.760 | 0.818 |
| 320x180 | `rgb_to_tensor` | 0.233 | 0.210 | 0.284 | 0.311 | 0.180 | 0.398 |
| 320x180 | `bilinear_2x` | 1.217 | 1.133 | 1.431 | 1.608 | 1.074 | 1.966 |
| 320x180 | `model_inference` | 12.187 | 12.175 | 13.002 | 13.402 | 11.151 | 13.537 |
| 320x180 | `tensor_to_rgb` | 5.501 | 5.339 | 6.071 | 6.457 | 5.156 | 6.973 |
| 320x180 | `neural_end_to_end` | 17.964 | 17.939 | 19.186 | 19.402 | 16.299 | 19.406 |
| 960x540 | `rgb_to_tensor` | 8.522 | 7.833 | 9.397 | 13.866 | 6.936 | 14.770 |
| 960x540 | `bilinear_2x` | 13.037 | 12.683 | 14.431 | 14.962 | 12.103 | 15.681 |
| 960x540 | `model_inference` | 103.577 | 102.309 | 108.855 | 110.399 | 97.737 | 118.262 |
| 960x540 | `tensor_to_rgb` | 52.636 | 52.220 | 54.445 | 54.949 | 49.440 | 59.377 |
| 960x540 | `neural_end_to_end` | 161.998 | 160.689 | 167.821 | 170.129 | 156.464 | 171.175 |

Stage definitions:

- `rgb_to_tensor`: Pillow RGB to normalized contiguous float32 NCHW tensor.
- `bilinear_2x`: Pillow bilinear baseline only.
- `model_inference`: eager PyTorch CPU model only, with a prepared tensor.
- `tensor_to_rgb`: prepared output tensor to Pillow RGB.
- `neural_end_to_end`: RGB-to-tensor + inference + tensor-to-RGB, measured directly.

PNG load/save, application UI, memory transfer to another device, synchronization, display, and sustained thermal/power behavior are excluded. Stages were measured independently; their distribution statistics must not be summed as a substitute for `neural_end_to_end`.

## Interpretation

This baseline is useful for regression checks and later platform comparisons. It is not a PC-versus-mobile performance claim: Android CPU, ExecuTorch, QNN/NPU, Vulkan transfer, precision changes, and trained-model effects remain unmeasured.

- 64x64: `model_inference` is the largest independently measured neural stage (0.481 ms, 61.4% of the directly measured end-to-end median).
- 320x180: `model_inference` is the largest independently measured neural stage (12.175 ms, 67.9% of the directly measured end-to-end median).
- 960x540: `model_inference` is the largest independently measured neural stage (102.309 ms, 63.7% of the directly measured end-to-end median).

The percentages compare independently measured stage medians with the direct end-to-end median, so they are diagnostic approximations rather than an additive latency breakdown.

Machine-readable data: [PC_BASELINE.json](PC_BASELINE.json).
