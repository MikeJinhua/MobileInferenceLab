# Android CPU Toolchain Decision and Readiness

Decision date: 2026-08-16

## Phase 3 Integration Route

The first Android CPU path will consume `org.pytorch:executorch-android:1.3.1` from Maven Central and use the Java/Kotlin `Module`, `Tensor`, and `EValue` APIs. The official AAR already contains the Java API, JNI/native ExecuTorch runtime, portable kernels, and XNNPACK for `arm64-v8a` and `x86_64`. This keeps the first device milestone focused on model loading, tensor conversion, inference, display, and measurement.

Phase 3 will not build ExecuTorch from source. A direct C++ integration remains appropriate later when Vulkan interop or lower-level instrumentation requires it, but it is not necessary to prove the first Android CPU inference path.

Official references:

- [Using ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)
- [ExecuTorch XNNPACK backend](https://docs.pytorch.org/executorch/stable/android-xnnpack.html)
- [Install Android Studio](https://developer.android.com/studio/install)
- [AGP 9.3 compatibility](https://developer.android.com/build/releases/agp-9-3-0-release-notes)
- [Install Android NDK and CMake](https://developer.android.com/studio/projects/install-ndk)

## Actual Build Contract

- ExecuTorch Android AAR: `1.3.1`, matching the exporter/runtime used in Phase 2.
- Android Studio: Quail 3 / 2026.1.3.
- JDK: Studio bundled JBR 25.0.2.
- Android Gradle Plugin: 9.3.0.
- Gradle: 9.5.0.
- Compile SDK / Build Tools: API 37 / 36.0.0.
- Future native NDK: r28c, matching the ExecuTorch 1.3 Android CI guidance; not installed for P3.2.
- First device ABI: `arm64-v8a`; `x86_64` may be used for an emulator smoke test.
- CPU backend: XNNPACK. No QNN, Vulkan, or quantization work belongs to Phase 3.

The installed versions differ from the original conservative proposal but form a current officially supported combination: Quail 3 supports AGP through 9.3, and AGP 9.3 supports API 37 with Gradle 9.5 and Build Tools 36.0.0. AGP requires at least JDK 17; the actual Gradle 9.5 build passes with Studio's JBR 25.

The prebuilt AAR path does not require an NDK, CMake, or Ninja for P3.2. Those are a separate native-build gate for later direct C++/Vulkan work.

## Current Host Audit

Run from the project directory:

```powershell
.venv\Scripts\python.exe -m tools.check_android_environment
.venv\Scripts\python.exe -m tools.check_android_environment --strict
```

The first command reports state without changing the machine. `--strict` exits nonzero until build prerequisites exist.

Latest observation on 2026-08-17:

| Component | Result |
| --- | --- |
| Android Studio / JDK | Quail 3 with JBR 25.0.2 at `E:\Program Files\Android\Android Studio` |
| Android SDK | API 37 and Build Tools 36.0.0 installed |
| Android NDK / Ninja | missing; not required for the prebuilt AAR app |
| CMake | found at `C:\Program Files\CMake\bin\cmake.exe` |
| ADB | 36.0.0 at `E:\DeveloperTools\platform-tools\adb.exe` |
| Connected Android device | Samsung SM-S916U, Android 13/API 33, arm64-v8a |

The prebuilt-AAR build and device gates now pass. The native-build gate remains false until NDK and Ninja are installed.

P3.2 built a 27,928,417-byte debug APK, installed it on the connected SM-S916U, and cold-started `MainActivity` in 412 ms as reported by `am start -W`. This is application launch time, not model inference timing.

The committed `gradle-wrapper.jar` comes from the official Gradle 9.5.0 release tag and is covered by Gradle's Apache License 2.0 distribution. Its SHA-256 is `497C8C2A7E5031F6AA847F88104AA80A93532EC32EE17BDB8D1D2F67A194A9C7`. The Gradle distribution itself is downloaded into the user's ignored Gradle cache.

Do not commit `local.properties`, SDK/NDK contents, downloaded AARs, APKs, signing keys, or device identifiers.
