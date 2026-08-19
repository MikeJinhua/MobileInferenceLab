"""Tests for QNN device-output analysis."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analyze_qnn_device_output import analyze
from tools.export_spatial_sr_qnn import OUTPUT_SHAPE


class QnnDeviceOutputTest(unittest.TestCase):
    def test_fp16_scale_difference_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = np.linspace(-0.5, 0.5, num=np.prod(OUTPUT_SHAPE), dtype="<f4")
            device = (reference.astype(np.float16)).astype("<f4")
            reference.tofile(root / "reference.raw")
            device.tofile(root / "device.raw")
            device.tofile(root / "repeated.raw")
            (root / "speed.txt").write_text("0.25", encoding="ascii")
            report = analyze(
                root / "reference.raw", root / "device.raw", root / "repeated.raw", root / "speed.txt"
            )
            self.assertTrue(report["output_finite"])
            self.assertLess(report["max_abs_difference"], 1e-2)
            self.assertEqual(report["runner_average_inference_ms"], 0.0125)


if __name__ == "__main__":
    unittest.main()
