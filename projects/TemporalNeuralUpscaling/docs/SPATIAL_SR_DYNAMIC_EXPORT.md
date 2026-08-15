# SpatialSR2x Bounded Dynamic Export Report

Date: 2026-08-15

## Contract

P2.5 fixes batch to 1, channels to RGB, and dtype to float32 while declaring independent height and width bounds of `16..128`. The export example is `[1,3,64,64]`. This bounded contract avoids claiming arbitrary image sizes and gives the ExecuTorch memory planner a finite maximum.

```powershell
.venv-executorch\Scripts\python.exe -m tools.export_spatial_sr_dynamic
```

The command creates ignored portable and XNNPACK `.pte` files plus `results/p2_5/dynamic_report.json`.

## Runtime Results

Both artifacts produced the correct 2x output for all tested inputs:

| Input | Output |
| --- | --- |
| `[1,3,16,16]` | `[1,3,32,32]` |
| `[1,3,17,31]` | `[1,3,34,62]` |
| `[1,3,64,64]` | `[1,3,128,128]` |
| `[1,3,96,48]` | `[1,3,192,96]` |
| `[1,3,128,128]` | `[1,3,256,256]` |

| Backend | `.pte` size | Maximum eager/runtime difference |
| --- | ---: | ---: |
| portable | 13,616 bytes | `2.682209014892578e-7` |
| XNNPACK | 14,640 bytes | `2.8312206268310547e-7` |

The exported PyTorch contract rejects representative out-of-range inputs `[1,3,15,64]` and `[1,3,64,129]`. Applications must still validate the same contract before invoking the runtime.

## Dynamic XNNPACK Partition

Unlike the static graph's single complete delegate, the dynamic graph contains three `XnnpackBackend` delegates and two portable `aten.view_copy` operators. Two symbolic-size reads and two shape multiplications remain at the top level. PixelShuffle is decomposed around dynamic reshape operations, so P2.5 records a partial-fallback topology even though all tested outputs pass.

This is functional compatibility evidence, not a performance result. Whether a fixed resolution or bounded-dynamic artifact is preferable must be decided using Android measurements in Phase 3.

## Limitations

Only batch 1, RGB float32, and height/width `16..128` are supported. The tests use the packaged PC runtime, not Android. They establish neither QNN compatibility nor image quality; the network still has deterministic random weights.
