"""Static regression tests for the minimal Android CPU scaffold."""

from pathlib import Path
import unittest


ANDROID_ROOT = Path(__file__).resolve().parents[1] / "android"


class AndroidScaffoldTest(unittest.TestCase):
    def test_versioned_build_contract(self) -> None:
        root_build = (ANDROID_ROOT / "build.gradle.kts").read_text(encoding="utf-8")
        app_build = (ANDROID_ROOT / "app" / "build.gradle.kts").read_text(encoding="utf-8")
        wrapper = (
            ANDROID_ROOT / "gradle" / "wrapper" / "gradle-wrapper.properties"
        ).read_text(encoding="utf-8")

        self.assertIn('version "9.3.0"', root_build)
        self.assertIn("gradle-9.5.0-bin.zip", wrapper)
        self.assertIn("compileSdk = 37", app_build)
        self.assertIn('executorch-android:1.3.1', app_build)
        self.assertIn("verifySpatialSrModel", app_build)

    def test_launcher_activity_is_declared(self) -> None:
        manifest = (
            ANDROID_ROOT / "app" / "src" / "main" / "AndroidManifest.xml"
        ).read_text(encoding="utf-8")
        activity = (
            ANDROID_ROOT
            / "app"
            / "src"
            / "main"
            / "java"
            / "com"
            / "mike"
            / "mobileinferencelab"
            / "temporalsr"
            / "MainActivity.java"
        )
        self.assertIn("android.intent.action.MAIN", manifest)
        self.assertIn("android.intent.category.LAUNCHER", manifest)
        self.assertTrue(activity.is_file())

    def test_device_runner_uses_static_tensor_contract(self) -> None:
        runner = (
            ANDROID_ROOT
            / "app"
            / "src"
            / "main"
            / "java"
            / "com"
            / "mike"
            / "mobileinferencelab"
            / "temporalsr"
            / "SpatialSrRunner.java"
        ).read_text(encoding="utf-8")
        self.assertIn("Module.load", runner)
        self.assertIn("Tensor.fromBlob", runner)
        self.assertIn("module.forward", runner)
        self.assertIn("{1, 3, 128, 128}", runner)

    def test_android_image_pipeline_has_rgb_display_and_stage_timing(self) -> None:
        pipeline = (
            ANDROID_ROOT / "app" / "src" / "main" / "java" / "com" / "mike"
            / "mobileinferencelab" / "temporalsr" / "ImageSrPipeline.java"
        ).read_text(encoding="utf-8")
        self.assertIn("bitmapToNchw", pipeline)
        self.assertIn("nchwToBitmap", pipeline)
        self.assertIn("Bitmap.createScaledBitmap", pipeline)
        self.assertIn("WARMUP = 5", pipeline)
        self.assertIn("ITERATIONS = 20", pipeline)
        for stage in ("preprocess", "inference", "postprocess", "neuralEndToEnd"):
            self.assertIn(stage, pipeline)


if __name__ == "__main__":
    unittest.main()
