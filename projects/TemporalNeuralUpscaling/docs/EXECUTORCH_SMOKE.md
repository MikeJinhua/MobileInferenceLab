# ExecuTorch Environment Smoke Test

Run date: 2026-08-15

## Result

The isolated Windows toolchain successfully completed:

```text
PyTorch Add model
  -> torch.export.export
  -> ExecuTorch edge lowering
  -> XNNPACK partitioning
  -> .pte serialization
  -> ExecuTorch Python runtime load
  -> runtime execution
```

| Item | Result |
| --- | --- |
| Python | 3.10.6 x64 |
| PyTorch | 2.12.0+cpu |
| ExecuTorch | 1.3.1 |
| Backend | XNNPACK |
| Inputs | 1.25 and 2.75 float32 tensors |
| Eager / runtime output | 4.0 / 4.0 |
| Eager vs `torch.export` max absolute difference | 0 |
| Eager vs ExecuTorch runtime max absolute difference | 0 |
| `.pte` size | 1,584 bytes |
| `pip check` | PASS |

Command:

```powershell
.venv-executorch\Scripts\python.exe -m tools.smoke_executorch `
  --output results\p2_2\add_xnnpack.pte `
  --report results\p2_2\smoke_report.json
```

The `.pte` and JSON output are local generated artifacts under `results/p2_2/` and are ignored by Git.

## Windows Wheel Finding

The ExecuTorch 1.3.1 Windows wheel contains `executorch/data/bin/flatc.exe`, but the serializer did not automatically resolve that location and fell back to searching PATH. The smoke tool resolves the bundled executable and sets `FLATC_EXECUTABLE` before serialization. No external FlatBuffers compiler was downloaded or committed.

## Observed Upstream Warnings

- PyTorch reports deprecated constant registration for an ExecuTorch enum.
- ExecuTorch/PyTree emits `LeafSpec` future warnings during lowering.
- The Python runtime API is marked experimental.
- The Windows XNNPACK runtime logs unsuccessful Linux-style CPU information probes before continuing.

These warnings did not affect serialization, runtime loading, execution, or numerical parity. Keep them visible and re-evaluate them when upgrading the toolchain.

## Scope Boundary

This test proves that the host export and packaged runtime path works for a minimal Add graph. It does not prove that `SpatialSR2x`, PixelShuffle, dynamic shapes, Android, or QNN are supported. Those remain separate evidence-gated tasks.
