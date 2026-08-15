# Development Environment

Detected on 2026-08-15 from Windows/VS Code workspace `F:\mobile_ai`. No packages or SDKs were installed by this task.

## Available

| Component | Detected value | Phase 1 status |
| --- | --- | --- |
| Git | 2.46.2.windows.1 | Ready |
| Python (`python`) | 3.10.6 at `C:\Python310\python.exe` | Ready |
| Python launcher (`py`) | 3.11.1 | Available, but project commands currently use `python` 3.10.6 |
| PyTorch | 2.0.0+cpu | Ready for Task P1.1; CPU only |
| CMake | 4.3.0 | Available for later native work |
| ADB | 36.0.0-13206524 | Available at `E:\DeveloperTools\platform-tools\adb.exe` |

## Project-local Dependencies and Missing Components

| Component | Observation | Action |
| --- | --- | --- |
| NumPy | 1.26.4 installed in project `.venv` | Ready for Phase 1 image/tensor interop |
| Pillow | 11.3.0 installed in project `.venv` | Ready for Phase 1 image I/O |
| pytest | Not installed | Not required; Phase 1 tests use `unittest` |
| Java/JDK | `java` not on PATH | Install/configure with Android Studio before Phase 3 |
| Android SDK | `ANDROID_HOME` not set | Select/install and configure before Phase 3 |
| Android NDK | `ANDROID_NDK_HOME` not set | Install a compatible NDK before native Android work |
| Ninja | Not on PATH | Install with Android/native toolchain when required |
| Vulkan SDK | `VULKAN_SDK` not set | Defer until Phase 6; Android device Vulkan support also needs target validation |
| ExecuTorch | Not checked/installed | Pin with a compatible PyTorch version in Phase 2 |
| Qualcomm QAIRT/QNN | Not checked/installed | Obtain licensed SDK after selecting supported Snapdragon hardware |

## Recommended Phase 1 Isolation

The repository-local `.venv` was created with `--system-site-packages` to reuse the existing CPU PyTorch, then NumPy and Pillow were installed inside it. Recreate dependencies from `requirements-phase1.txt`; choose exact PyTorch/ExecuTorch pins before export work because the current PyTorch 2.0.0 is old and compatibility is version-sensitive.

Large SDKs (Android Studio/SDK/NDK, ExecuTorch dependencies, QAIRT/QNN, Vulkan SDK) remain manual, phase-gated installations.

## Export Readiness Note

PyTorch 2.0.0 does not expose the public `torch.export` API used by current ExecuTorch workflows. P1.3 uses TorchScript tracing only for a local operator/parity preflight. Phase 2 requires selecting and installing a mutually compatible PyTorch/ExecuTorch version pair in an isolated environment before producing any ExecuTorch artifact.

## Phase 2.1 Decision

The selected initial export baseline is Python 3.10 + PyTorch 2.12.0 + ExecuTorch 1.3.1 in `.venv-executorch`. No packages were installed by Phase 2.1. Visual Studio 2022 Enterprise/MSVC is present, but the current shell is not a Developer PowerShell and the Clang component was not detected. WSL Ubuntu 22.04 is not available and remains a manual prerequisite for the later QNN phase. See `EXECUTORCH_TOOLCHAIN.md`.
