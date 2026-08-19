# Static SpatialSR2x QNN HTP FP16 Export

P4.3 lowers the deterministic, untrained `SpatialSR2x.network` for Snapdragon 8 Gen 2 (`SM8550`, HTP v73) using ExecuTorch 1.3.1 and QNN 2.37.0.250724. It validates model conversion and partitioning only; P4.4 must prove execution on the phone.

## Reproduction

Run from Windows with WSL:

```powershell
wsl.exe -d Ubuntu -- bash /mnt/f/mobile_ai/projects/TemporalNeuralUpscaling/tools/run_qnn_export.sh
```

Or, from the project directory inside WSL:

```bash
bash tools/run_qnn_export.sh
```

The wrapper validates the external SDK/NDK environment, configures the official bootstrap's host `libc++`, selects the ExecuTorch wheel's bundled `flatc`, and invokes `tools.export_spatial_sr_qnn`. Generated `.pte` and JSON files are written under ignored `results/p4_3/`.

## Verified result

| Property | Result |
|---|---|
| Input/output | float32 NCHW `[1,3,64,64]` → `[1,3,128,128]` |
| QNN target | SM8550 / HTP v73 / FP16 |
| Exported operators | convolution ×2, ReLU ×1, PixelShuffle ×1 |
| QNN delegates | 1, backend `QnnBackend` |
| Portable fallback | none |
| Generated `.pte` size | 62,976 bytes |
| Eager/export maximum difference | 0 |
| Repeated eager maximum difference | 0 |
| Output finite | yes |

The QNN partitioner explicitly reported support for both convolutions, ReLU, and PixelShuffle, and produced one fully delegated subgraph. The offline artifact contains an SM8550-targeted HTP context, so it is not executed by the x86 host runtime. Device output shape, numerical comparison, backend loading, and proof of HTP execution are P4.4 gates.

The TorchAO wheel emits warnings about optional CUTLASS/MXFP8 extensions on this CPU-only host. They did not prevent QNN lowering and are not used by this FP16 export path.

## Safety and limitations

The Qualcomm SDK, SDK libraries, bootstrap `libc++`, generated `.pte`, JSON report, local paths, and device identifiers remain untracked. Only the public export integration, commands, tests, and observed metadata are committed. Random weights make this a deployment-pipeline artifact, not an SR quality result.
