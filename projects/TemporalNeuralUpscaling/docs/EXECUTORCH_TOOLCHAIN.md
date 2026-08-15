# ExecuTorch Toolchain Decision

Decision date: 2026-08-15

## Selected Baseline

| Component | Selected version/environment | Reason |
| --- | --- | --- |
| Python | 3.10 x64 | Already available and within ExecuTorch's documented Python 3.10–3.13 range for release 1.3 |
| PyTorch | 2.12.0 CPU | ExecuTorch release/1.3 declares `torch>=2.12.0a0`; use the stable 2.12 release rather than a nightly |
| ExecuTorch | 1.3.1 | Latest patch in the documentation's current stable 1.3 line |
| Initial backend | portable, then XNNPACK | Establish `.pte` correctness first, then mobile-CPU delegation before QNN |
| Phase 2 host | Native Windows 11 x64 | Supported for ExecuTorch host tools; keeps initial export close to the current VS Code workflow |
| Future QNN host | WSL Ubuntu 22.04 x64 | Qualcomm backend documentation lists Ubuntu 22.04/WSL as a verified host path |

The version set is recorded in `requirements-executorch.txt`. It intentionally does not reuse the Phase 1 `.venv`, which contains PyTorch 2.0 and NumPy 1.x.

## Why Not ExecuTorch 1.4.1 Yet?

ExecuTorch 1.4.1 appeared on PyPI on 2026-08-14, while the official documentation version selector still marks 1.3 as the stable release. The release/1.4 package declares `torch>=2.13.0a0`, which would move this project onto a newer PyTorch line immediately. For a learning project whose next goal is a small reproducible export, the documented 1.3.1/2.12 baseline has lower churn. Upgrade only through a dedicated task after 1.4 documentation and QNN guidance are stable, or if 1.3 blocks this model.

This is a time-sensitive decision and should be rechecked before a fresh environment is created later.

## Official Workflow to Follow

The documented export flow is:

```text
eval PyTorch model + example inputs
  -> torch.export.export
  -> ExecuTorch edge lowering
  -> optional backend partitioner (XNNPACK first)
  -> to_executorch
  -> serialized .pte
```

The first executable Phase 2 task will use a fixed `[1, 3, 64, 64]` float32 NCHW input. Dynamic height/width will be a later explicit task because the official exporter requires varying dimensions and bounds to be declared.

## Environment Separation

Planned Windows setup, to be executed only in Phase 2.2:

```powershell
cd projects/TemporalNeuralUpscaling
python -m venv .venv-executorch
.venv-executorch\Scripts\python.exe -m pip install --upgrade pip
.venv-executorch\Scripts\python.exe -m pip install -r requirements-executorch.txt
```

Run native build-dependent commands from Visual Studio Developer PowerShell. Do not install these packages into the system Python or the Phase 1 `.venv`.

## Detected Host Readiness

- Visual Studio 2022 Enterprise with MSVC x64 tools: detected.
- Visual Studio Developer PowerShell launcher: detected.
- `cl.exe` / `clang-cl.exe`: not visible in the current ordinary PowerShell session.
- Visual Studio Clang component: not detected; install **C++ Clang tools for Windows** only if the wheel/export path requires native compilation or source build.
- CMake 4.3.0: installed, but ExecuTorch release/1.3 source build declares CMake `<4.0`; use a compatible environment-local CMake if source building becomes necessary.
- WSL Ubuntu 22.04: not currently available; defer installation until QNN preparation.
- Android NDK and Qualcomm QNN SDK: not installed; not required for portable/XNNPACK export.

## Validation Matrix

Phase 2 must pass these gates in order:

1. **Environment smoke test:** import `torch` and `executorch`; record exact versions; export the official minimal Add example.
2. **PyTorch export:** export the tensor-only `SpatialSR2x.network` with a fixed input and compare eager/exported outputs.
3. **Portable `.pte`:** lower and serialize; run through the Python ExecuTorch runtime if included by the selected Windows wheel; compare outputs.
4. **XNNPACK `.pte`:** lower with `XnnpackPartitioner`, inspect delegated versus fallback partitions, and compare outputs.
5. **Shape policy:** keep static 64x64 first; add bounded dynamic H/W only after static correctness.
6. **Operator decision:** retain PixelShuffle unless actual export or partition evidence shows failure. Do not redesign based on speculation.

Generated `.pte` files remain ignored by Git. Commit export scripts, operator/partition reports, commands, package lock evidence, and numerical results instead.

## Sources

- [ExecuTorch 1.3 Getting Started](https://docs.pytorch.org/executorch/1.3/getting-started.html)
- [ExecuTorch 1.3 model export and lowering](https://docs.pytorch.org/executorch/1.3/using-executorch-export.html)
- [ExecuTorch stable-version selector](https://docs.pytorch.org/executorch/versions.html)
- [ExecuTorch release/1.3 package metadata](https://github.com/pytorch/executorch/blob/release/1.3/pyproject.toml)
- [ExecuTorch Qualcomm backend prerequisites](https://docs.pytorch.org/executorch/1.3/backends-qualcomm.html)
- [ExecuTorch Android integration](https://docs.pytorch.org/executorch/1.3/using-executorch-android.html)
- [ExecuTorch 1.3.1 on PyPI](https://pypi.org/project/executorch/1.3.1/)
