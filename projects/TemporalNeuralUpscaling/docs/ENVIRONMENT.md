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

## Missing or Not Configured

| Component | Observation | Action |
| --- | --- | --- |
| NumPy | Not installed for Python 3.10; PyTorch emits a warning | Install in a project virtual environment before image/tensor interop work |
| pytest | Not installed | Not required; Phase 1 tests use `unittest` |
| Java/JDK | `java` not on PATH | Install/configure with Android Studio before Phase 3 |
| Android SDK | `ANDROID_HOME` not set | Select/install and configure before Phase 3 |
| Android NDK | `ANDROID_NDK_HOME` not set | Install a compatible NDK before native Android work |
| Ninja | Not on PATH | Install with Android/native toolchain when required |
| Vulkan SDK | `VULKAN_SDK` not set | Defer until Phase 6; Android device Vulkan support also needs target validation |
| ExecuTorch | Not checked/installed | Pin with a compatible PyTorch version in Phase 2 |
| Qualcomm QAIRT/QNN | Not checked/installed | Obtain licensed SDK after selecting supported Snapdragon hardware |

## Recommended Phase 1 Isolation

Do not modify the system Python implicitly. Before Task P1.2, create a repository-local virtual environment and install a compatible, pinned PyTorch/NumPy set. Exact pins should be chosen before ExecuTorch work because the current PyTorch 2.0.0 is old and ExecuTorch compatibility is version-sensitive.

Large SDKs (Android Studio/SDK/NDK, ExecuTorch dependencies, QAIRT/QNN, Vulkan SDK) remain manual, phase-gated installations.
