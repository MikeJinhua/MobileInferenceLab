# Qualcomm QNN / HTP Readiness

P4.1 selected the deployment route, and P4.2 established the licensed build environment. No QNN `.pte` has been exported and no NPU execution is claimed yet.

## Selected route

- Target device SoC: `SM8550` (Snapdragon 8 Gen 2), mapped by ExecuTorch 1.3.1 to HTP v73 with 8 MiB VTCM.
- Host: WSL Ubuntu 22.04, matching the ExecuTorch Qualcomm backend's verified host route.
- Android native toolchain: NDK 26c for the first QNN build, as verified by the upstream Qualcomm backend guide.
- QAIRT/QNN: `2.37.0.250724`, the version selected by the installed ExecuTorch 1.3.1 package. The AOT SDK and Android runtime libraries must use the same QNN version.
- First artifact: static batch-1 RGB `64x64 -> 128x128`, QNN HTP FP16. Quantized HTP is deferred to Phase 5 so the first NPU gate does not mix backend integration with quantization.
- Acceptance: inspect the lowered graph, require the intended QNN delegate, enumerate every portable fallback, then verify device output shape/finiteness/parity before measuring initialization/load/inference/memory.

The current graph contains two convolutions, ReLU, and PixelShuffle. The installed QNN backend contains convolution/ReLU builders and explicitly preserves `aten.pixel_shuffle.default`; its HTP quantization rules map PixelShuffle to QNN DepthToSpace. This is encouraging source-level evidence, not proof that the complete FP16 graph will delegate on SM8550. P4.2 must run the real partitioner with the installed SDK.

## Verified host and device state

Command:

```powershell
.venv-executorch\Scripts\python.exe -m tools.check_qnn_environment --strict --output results\p4_2\qnn_environment.json
```

| Check | Result |
|---|---|
| ExecuTorch Qualcomm backend files | PASS |
| Connected arm64 device | PASS |
| Device SoC present in ExecuTorch QNN schema | PASS: SM8550 / HTP v73 |
| WSL Ubuntu 22.04 and host tools | PASS |
| Android NDK | PASS: r26c / 26.2.11394342 |
| QNN SDK markers, version, host/Android/v73 libraries | PASS: 2.37.0 / build 250724 |
| Isolated ExecuTorch dependencies and `pip check` | PASS |
| WSL Qualcomm backend import | PASS |
| Ready for QNN export/device validation | **YES** |

The strict command returns zero only when every required gate passes. Reports are generated below ignored `results/`; no device serial is collected.

## Reproducing the local environment

The licensed SDK, NDK, Python environment, downloaded archives, and caches remain outside Git. From the project directory inside WSL, run `source tools/qnn_env.sh`; the script derives default paths from `$HOME`, validates SDK/NDK markers, and configures the QNN x86_64 host-library path. Override `QNN_SDK_ROOT` or `ANDROID_NDK_ROOT` before sourcing when using a different external installation location.

P4.3 may now lower only the static `64x64` HTP FP16 model, inspect the delegate/fallback graph, and validate the output contract. Building the Android QNN runtime and proving NPU execution remain P4.4.

## Version and licensing rules

The QNN compile SDK and Android runtime must match. A context produced by a newer SDK can fail against older runtime libraries. Qualcomm SDK headers, shared libraries, tools, license files, generated context binaries, QNN `.pte` files, local paths, and device identifiers must stay untracked. Only project code, commands, metadata, and measured results may be committed.

Official references:

- [ExecuTorch Qualcomm AI Engine backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
- [ExecuTorch Android backends](https://docs.pytorch.org/executorch/stable/android-backends.html)
- [ExecuTorch release/1.3 SM8550 schema](https://github.com/pytorch/executorch/blob/release/1.3/backends/qualcomm/serialization/qc_schema.py)
- [Qualcomm QAIRT documentation](https://docs.qualcomm.com/)
