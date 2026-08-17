# Android Static XNNPACK Device Inference

Date: 2026-08-17

## Scope

P3.3 proves that the Phase 2 static `SpatialSR2x` XNNPACK artifact can be packaged in the Android app, loaded by the ExecuTorch 1.3.1 AAR, and executed on a real arm64 phone. It validates a tensor path only. Image decoding, RGB conversion from a Bitmap, output display, and structured timing belong to P3.4.

The model still has deterministic random weights. This result proves deployment correctness and makes no super-resolution image-quality claim.

## Reproduction

From the project directory:

```powershell
.venv-executorch\Scripts\python.exe -m tools.prepare_android_model
$env:JAVA_HOME='E:\Program Files\Android\Android Studio\jbr'
.\android\gradlew.bat -p android :app:assembleDebug
adb install -r android\app\build\outputs\apk\debug\app-debug.apk
adb shell am start -W -n com.mike.mobileinferencelab.temporalsr/.MainActivity
adb logcat -d -s TNUInference:I *:S
```

The first command regenerates `android/app/src/main/assets/spatial_sr_xnnpack.pte`. The `.pte`, APK, reports, and build outputs are ignored by Git. Gradle intentionally fails at `verifySpatialSrModel` when the generated asset is missing, instead of silently building an app that cannot infer.

## Artifact Evidence

| Item | Result |
| --- | --- |
| Export stack | PyTorch 2.12.0+cpu / ExecuTorch 1.3.1 |
| Model seed | 20260815 |
| Input / output | float32 `[1,3,64,64]` / `[1,3,128,128]` |
| Backend topology | one `XnnpackBackend` delegate, zero portable fallback operators |
| Artifact size | 12,592 bytes |
| SHA-256 | `d2edf8943f59b463fae7fcdc1eee147e604d45a2cc8e42dfe9f6c2dcd0cc4b13` |

## Device Evidence

Device: Samsung SM-S916U, Android 13/API 33, `arm64-v8a`.

The app creates a deterministic float32 input, performs inference twice, validates the exact output shape and finite values, requires repeat maximum difference at most `1e-6`, and compares the float64-accumulated output checksum with the PC eager reference using a `1e-3` tolerance.

```text
status=PASS output_shape=[1, 3, 128, 128]
min=-0.429966062 max=0.347079456
checksum=501.158083683
repeat_max_diff=0.00000000
checksum_ref_diff=5.36697771e-05
```

The successful execution of the delegated `.pte` through the Android AAR also confirms that the packaged runtime can resolve the XNNPACK backend on this device.

## Limitations

This is one functional smoke run, not a latency benchmark. It does not separate model load, input preparation, inference, output conversion, or UI time. It does not test sustained load, thermals, power, memory, dynamic shapes, QNN, Vulkan, or quantization. P3.4 will add the image path and properly separated measurements.
