# Android CPU Toolchain Decision and Readiness

Decision date: 2026-08-16

## Phase 3 Integration Route

The first Android CPU path will consume `org.pytorch:executorch-android:1.3.1` from Maven Central and use the Java/Kotlin `Module`, `Tensor`, and `EValue` APIs. The official AAR already contains the Java API, JNI/native ExecuTorch runtime, portable kernels, and XNNPACK for `arm64-v8a` and `x86_64`. This keeps the first device milestone focused on model loading, tensor conversion, inference, display, and measurement.

Phase 3 will not build ExecuTorch from source. A direct C++ integration remains appropriate later when Vulkan interop or lower-level instrumentation requires it, but it is not necessary to prove the first Android CPU inference path.

Official references:

- [Using ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)
- [ExecuTorch XNNPACK backend](https://docs.pytorch.org/executorch/stable/android-xnnpack.html)
- [Install Android Studio](https://developer.android.com/studio/install)
- [AGP 8.9 compatibility](https://developer.android.com/build/releases/agp-8-9-0-release-notes)
- [Install Android NDK and CMake](https://developer.android.com/studio/projects/install-ndk)

## Pinned Build Contract

- ExecuTorch Android AAR: `1.3.1`, matching the exporter/runtime used in Phase 2.
- JDK: 17.
- Android Gradle Plugin: 8.9.x.
- Gradle: 8.11.1.
- Compile SDK / Build Tools: API 35 / 35.0.0.
- NDK: r28c, matching the ExecuTorch 1.3 Android CI guidance.
- First device ABI: `arm64-v8a`; `x86_64` may be used for an emulator smoke test.
- CPU backend: XNNPACK. No QNN, Vulkan, or quantization work belongs to Phase 3.

The NDK is included in readiness even though the prebuilt AAR hides most native build work: this repository's later Android phases require native C++/Vulkan, so establishing one consistent side-by-side NDK now avoids a second incompatible setup.

## Current Host Audit

Run from the project directory:

```powershell
.venv\Scripts\python.exe -m tools.check_android_environment
.venv\Scripts\python.exe -m tools.check_android_environment --strict
```

The first command reports state without changing the machine. `--strict` exits nonzero until build prerequisites exist.

Observed on 2026-08-16:

| Component | Result |
| --- | --- |
| JDK / `JAVA_HOME` | missing |
| Android Studio / SDK / `sdkmanager` | missing |
| Android NDK | missing |
| Ninja | missing |
| CMake | found at `C:\Program Files\CMake\bin\cmake.exe` |
| ADB | 36.0.0 at `E:\DeveloperTools\platform-tools\adb.exe` |
| Connected Android device | none |

The host is not ready to build or validate an Android app. P3.2 begins only after the strict checker passes. Installing Android Studio through its Setup Wizard is the recommended route; add SDK Platform 35, Build Tools 35.0.0, NDK r28c, CMake, and Ninja, then configure the SDK/JDK paths outside version control.

Do not commit `local.properties`, SDK/NDK contents, downloaded AARs, APKs, signing keys, or device identifiers.
