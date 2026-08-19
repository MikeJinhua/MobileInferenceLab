# SM8550 QNN HTP Device Inference

P4.4 proves that the P4.3 static FP16 artifact executes through the QNN HTP runtime on the connected Snapdragon 8 Gen 2 (`SM8550`, HTP v73) phone. This is a tensor-runner validation of the deterministic random-weight model, not an image-quality result.

## Runtime and deployment

- ExecuTorch source/runtime: v1.3.1, pinned commit `e2f18eb23c45bd22ca332b0b8b49a81de304b472`.
- Build host: WSL 2, Ubuntu 22.04, CMake 3.31.6, Android NDK r26c, arm64-v8a, Android API 30.
- Qualcomm runtime: QNN 2.37.0.250724, HTP v73 FP16.
- Model: ignored 62,976-byte static `[1,3,64,64] -> [1,3,128,128]` `.pte`, one `QnnBackend` delegate and no portable fallback.
- Device route: official `qnn_executor_runner`, `libqnn_executorch_backend.so`, and the matching minimal HTP/v73 runtime set under `/data/local/tmp/tnu_qnn`.

The runner log shows QNN backend type 2 (the HTP backend selected in the compile spec), restores the offline QNN context, loads the matching v73 stub/skeleton, executes successfully, and returns the expected tensor. The deployment command is:

```bash
bash tools/run_qnn_on_device.sh
```

`tools/build_qnn_runner.sh` reproducibly builds the external v1.3.1 checkout. SDK files, runner binaries, models, raw tensors, ETDump, logs, and device identifiers remain ignored/untracked.

## Numerical validation

The input is deterministic float32 little-endian NCHW data generated with seed `20260820`. The device output is compared element-by-element with the PC eager result.

| Check | Result |
|---|---:|
| Output shape/dtype | `[1,3,128,128]`, float32 |
| Finite output | PASS |
| Repeated-device maximum difference | 0 |
| Eager/device maximum absolute difference | 0.000331074 |
| Eager/device mean absolute difference | 0.000057550 |
| Device output sum | 399.258198 |
| Eager output sum | 399.231130 |

The small difference is consistent with an FP16 HTP graph returning float32 output. The acceptance tolerance is `1e-2`; this run is comfortably below it.

## Timing and memory

Each of 20 samples performs five warmups followed by a block of 20 timed executions. The table reports per-execution latency derived from each block. Timing is inside the native runner around `method->execute()` and excludes model/backend load, file input, output writing, ADB, UI, and image conversion.

| Metric | Result |
|---|---:|
| Mean | 0.121130 ms |
| Median | 0.105225 ms |
| P95 | 0.172500 ms |
| Minimum | 0.099400 ms |
| Maximum | 0.208600 ms |
| Model file open from process start | 1.576 ms |
| Backend + method load after file open | 206.989 ms |
| Method ready from process start | 208.565 ms |
| Peak runner RSS | 11,496 KiB |

Peak RSS is sampled from `/proc/<pid>/status` during a separate 5,000-execution run. It is process resident memory, not isolated HTP/DSP memory or total system power cost.

The retained Phase 3 XNNPACK app measured a 0.372 ms median at the same tensor shape. The QNN runner median is lower, but this is not a strict backend speedup ratio: the runners, instrumentation layers, and app/JNI overhead differ. A direct image-pipeline comparison would require QNN integration into the Android app and remains a possible follow-up, not part of this native HTP execution gate.

## Limitations

- Random weights validate deployment only and say nothing about SR visual quality.
- This short run does not measure sustained thermal behavior, power, DSP-only memory, Android UI/JNI overhead, or RGB preprocessing/postprocessing.
- The QNN runner is deployed to a temporary test directory rather than packaged in the public APK because Qualcomm runtime libraries cannot be committed.
- Results apply only to this SM8550, static 64x64 input, FP16 graph, and pinned software stack.
