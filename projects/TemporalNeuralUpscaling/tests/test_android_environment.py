"""Tests for the non-mutating Android environment inspector."""

from pathlib import Path
import tempfile
import unittest

from tools.check_android_environment import find_sdk_root, first_existing_path


class AndroidEnvironmentTest(unittest.TestCase):
    def test_first_existing_path_skips_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            found = first_existing_path([None, "Z:/does-not-exist", directory])
        self.assertEqual(found, Path(directory).resolve())

    def test_android_home_has_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            found = find_sdk_root(
                {"ANDROID_HOME": first, "ANDROID_SDK_ROOT": second}, candidates=()
            )
        self.assertEqual(found, Path(first).resolve())

    def test_candidate_is_used_without_environment_variable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            found = find_sdk_root({}, candidates=(Path(directory),))
        self.assertEqual(found, Path(directory).resolve())


if __name__ == "__main__":
    unittest.main()
