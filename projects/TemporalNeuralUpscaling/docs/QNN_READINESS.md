# Qualcomm QNN / HTP Readiness

P4.1 is a read-only readiness and deployment decision. It does not download the licensed QAIRT/QNN SDK, build a QNN runtime, export a QNN `.pte`, or claim NPU execution.

## Selected route

- Target device SoC: `SM8550` (Snapdragon 8 Gen 2), mapped by ExecuTorch 1.3.1 to HTP v73 with 8 MiB VTCM.
- Host: WSL Ubuntu 22.04, matching the ExecuTorch Qualcomm backend's verified host route.
- Android native toolchain: NDK 26c for the first QNN build, as verified by the upstream Qualcomm backend guide.
- QAIRT/QNN: `2.37.0.250724`, the version selected by the installed ExecuTorch 1.3.1 package. The AOT SDK and Android runtime libraries must use the same QNN version.
- First artifact: static batch-1 RGB `64x64 -> 128x128`, QNN HTP FP16. Quantized HTP is deferred to Phase 5 so the first NPU gate does not mix backend integration with quantization.
- Acceptance: inspect the lowered graph, require the intended QNN delegate, enumerate every portable fallback, then verify device output shape/finiteness/parity before measuring initialization/load/inference/memory.

The current graph contains two convolutions, ReLU, and PixelShuffle. The installed QNN backend contains convolution/ReLU builders and explicitly preserves `aten.pixel_shuffle.default`; its HTP quantization rules map PixelShuffle to QNN DepthToSpace. This is encouraging source-level evidence, not proof that the complete FP16 graph will delegate on SM8550. P4.2 must run the real partitioner with the installed SDK.

## Observed host and device state

Command:

```powershell
.venv-executorch\Scripts\python.exe -m tools.check_qnn_environment --output results\p4_1\qnn_environment.json
```

| Check | Result |
|---|---|
| ExecuTorch Qualcomm backend files | PASS |
| Connected arm64 device | PASS |
| Device SoC present in ExecuTorch QNN schema | PASS: SM8550 / HTP v73 |
| WSL Ubuntu 22.04 | MISSING |
| Android NDK | MISSING |
| `QNN_SDK_ROOT` and SDK layout markers | MISSING |
| QNN Python dependency `py-cpuinfo` | MISSING |
| Ready for QNN export/device validation | **NO** |

The strict command intentionally returns exit code 1 until all required checks pass. Reports are generated below ignored `results/`; no device serial is collected.

## Required next setup

1. Enable/install WSL 2 with Ubuntu 22.04. This is a system-level change and is not performed by the audit.
2. Install Android NDK 26c and expose `ANDROID_NDK_ROOT` inside WSL.
3. Obtain QAIRT/QNN `2.37.0.250724` through Qualcomm's official software channel, accept its license, keep it outside this repository, and set `QNN_SDK_ROOT` to the directory containing `QNN_README.txt` and `sdk.yaml`.
4. Create a QNN-specific ExecuTorch 1.3.1 environment/source checkout and install its declared preparation dependencies. Do not alter the working XNNPACK environment.
5. Re-run this audit in WSL before attempting export.

## Version and licensing rules

The QNN compile SDK and Android runtime must match. A context produced by a newer SDK can fail against older runtime libraries. Qualcomm SDK headers, shared libraries, tools, license files, generated context binaries, QNN `.pte` files, local paths, and device identifiers must stay untracked. Only project code, commands, metadata, and measured results may be committed.

Official references:

- [ExecuTorch Qualcomm AI Engine backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
- [ExecuTorch Android backends](https://docs.pytorch.org/executorch/stable/android-backends.html)
- [ExecuTorch release/1.3 SM8550 schema](https://github.com/pytorch/executorch/blob/release/1.3/backends/qualcomm/serialization/qc_schema.py)
- [Qualcomm QAIRT documentation](https://docs.qualcomm.com/)
