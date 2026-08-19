"""Tests for deterministic QNN runner input/reference generation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.prepare_qnn_device_inputs import prepare


class QnnDeviceInputsTest(unittest.TestCase):
    def test_raw_files_have_static_tensor_sizes_and_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_report = prepare(Path(first))
            second_report = prepare(Path(second))
            self.assertEqual(first_report, second_report)
            self.assertEqual(first_report["input_bytes"], 1 * 3 * 64 * 64 * 4)
            self.assertEqual(first_report["reference_bytes"], 1 * 3 * 128 * 128 * 4)
            first_input = np.fromfile(Path(first) / "input.raw", dtype="<f4")
            second_input = np.fromfile(Path(second) / "input.raw", dtype="<f4")
            np.testing.assert_array_equal(first_input, second_input)


if __name__ == "__main__":
    unittest.main()
