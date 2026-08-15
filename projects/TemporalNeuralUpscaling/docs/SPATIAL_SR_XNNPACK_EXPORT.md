# SpatialSR2x XNNPACK Export Report

Date: 2026-08-15

## Scope

Phase 2.4 lowers the same static float32 `SpatialSR2x.network` used in P2.3 to the ExecuTorch XNNPACK backend. It records the post-partition graph, checks portable fallback, executes the generated `.pte`, and compares the result with eager PyTorch. Dynamic shapes, Android, QNN, Vulkan, quantization, training, and image-quality evaluation are outside this task.

## Reproduction

```powershell
.venv-executorch\Scripts\python.exe -m tools.export_spatial_sr_xnnpack
```

Generated files are local and ignored by Git: `results/p2_4/spatial_sr_xnnpack.pte` and `results/p2_4/xnnpack_report.json`.

## Result

| Item | Observed value |
| --- | --- |
| PyTorch / ExecuTorch | `2.12.0+cpu` / `1.3.1` |
| Input / output | float32 `[1,3,64,64]` / `[1,3,128,128]` |
| Export operators | `conv2d` x2, `relu` x1, `pixel_shuffle` x1 |
| Delegate nodes / backend | 1 / `XnnpackBackend` |
| Portable fallback operators | none |
| `.pte` size | 12,592 bytes |
| Eager/export max difference | 0 |
| Eager/runtime max / mean difference | `2.980232238769531e-7` / `2.542216748224746e-8` |

The lowered top-level graph contains one `executorch_call_delegate` and no residual portable operator. Therefore, for this exact static artifact and toolchain, the complete graph—including the PixelShuffle path—is delegated to XNNPACK. This does not imply QNN support or delegation, or establish behavior for other shapes or versions.

## Limitations

The packaged Python runtime verifies functional execution on the development PC, not Android latency or sustained performance. The model still uses deterministic random weights; the result validates deployment mechanics only and makes no super-resolution quality claim. Upstream ExecuTorch deprecation and Windows CPU-probe warnings remain visible but do not change the passing result.
