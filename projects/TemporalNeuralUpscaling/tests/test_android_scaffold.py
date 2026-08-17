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


if __name__ == "__main__":
    unittest.main()
