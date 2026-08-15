# SpatialSR2x Portable ExecuTorch Export

Run date: 2026-08-15

## Result

The project model's tensor-only inference core successfully completed:

```text
SpatialSR2x.network
  -> torch.export.export (static [1,3,64,64])
  -> ExecuTorch edge lowering without a backend partitioner
  -> portable .pte
  -> ExecuTorch Python runtime
  -> float32 [1,3,128,128]
```

| Item | Result |
| --- | --- |
| PyTorch | 2.12.0+cpu |
| ExecuTorch | 1.3.1 |
| Backend | portable kernels; no delegate |
| Input | float32 NCHW `[1,3,64,64]` |
| Output | float32 NCHW `[1,3,128,128]` |
| Model seed | 20260815 |
| Eager vs `torch.export` max absolute difference | 0 |
| Eager vs portable runtime max absolute difference | 2.086162567138672e-7 |
| Eager vs portable runtime mean absolute difference | 2.5059028629925706e-8 |
| `.pte` size | 12,720 bytes |

The runtime difference is below the task tolerance of `1e-5` and is consistent with normal floating-point implementation differences.

## Exported Operator Inventory

| Operator | Count |
| --- | ---: |
| `aten.conv2d.default` | 2 |
| `aten.relu.default` | 1 |
| `aten.pixel_shuffle.default` | 1 |

This proves that the portable runtime in the selected package can execute the model's PixelShuffle path. It does not prove that XNNPACK or QNN will delegate PixelShuffle.

## Reproduction

```powershell
.venv-executorch\Scripts\python.exe -m tools.export_spatial_sr_portable `
  --output results\p2_3\spatial_sr_portable.pte `
  --report results\p2_3\portable_report.json
```

Generated `.pte` and JSON files remain local under `results/p2_3/` and are ignored by Git.

## Boundaries

- Input batch, channel count, height, and width are static in this artifact.
- Python validation in `SpatialSR2x.forward` is outside the exported `network` core and remains a host contract.
- No XNNPACK/QNN partitioner was used, so this report contains no delegation or fallback claim.
- The model still has deterministic random weights and provides no SR-quality evidence.
- Android execution, dynamic shapes, quantization, and performance are unmeasured.
