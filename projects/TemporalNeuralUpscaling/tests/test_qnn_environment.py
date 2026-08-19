"""Tests for the non-mutating QNN readiness inspector."""

from pathlib import Path
import tempfile
import unittest

from tools.check_qnn_environment import (
    decode_process_output,
    find_ndk_versions,
    parse_device_properties,
    parse_os_release,
    parse_key_values,
    valid_qnn_sdk,
)


class QnnEnvironmentTest(unittest.TestCase):
    def test_qnn_sdk_requires_public_layout_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(valid_qnn_sdk(root))
            (root / "QNN_README.txt").touch()
            (root / "sdk.yaml").touch()
            self.assertTrue(valid_qnn_sdk(root))

    def test_android_sdk_ndk_versions_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory)
            (sdk / "ndk" / "26.3.11579264").mkdir(parents=True)
            self.assertEqual(find_ndk_versions(sdk, {}), ("26.3.11579264",))

    def test_device_properties_do_not_include_serial(self) -> None:
        values = parse_device_properties(
            "abi=arm64-v8a\nsoc_model=SM8550\nboard=kalama\nhardware=qcom\n"
        )
        self.assertEqual(values["soc_model"], "SM8550")
        self.assertEqual(set(values), {"abi", "soc_model", "board", "hardware"})

    def test_utf16_wsl_output_is_decoded(self) -> None:
        raw = "Ubuntu-22.04\r\n".encode("utf-16le")
        self.assertEqual(decode_process_output(raw), "Ubuntu-22.04")

    def test_os_release_detects_ubuntu_version_independent_of_distro_name(self) -> None:
        values = parse_os_release('ID=ubuntu\nVERSION_ID="22.04"\n')
        self.assertEqual(values, {"ID": "ubuntu", "VERSION_ID": "22.04"})

    def test_wsl_probe_key_values_are_parsed(self) -> None:
        values = parse_key_values("qnn_version=2.37.0\nqnn_markers=true\n")
        self.assertEqual(values["qnn_version"], "2.37.0")
        self.assertEqual(values["qnn_markers"], "true")


if __name__ == "__main__":
    unittest.main()
