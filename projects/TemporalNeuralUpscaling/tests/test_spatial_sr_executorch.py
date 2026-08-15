"""Conditional portable ExecuTorch test for SpatialSR2x."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


EXECUTORCH_AVAILABLE = importlib.util.find_spec("executorch") is not None


@unittest.skipUnless(EXECUTORCH_AVAILABLE, "ExecuTorch is tested in .venv-executorch")
class SpatialSRExecuTorchTest(unittest.TestCase):
    def test_static_portable_export_runtime_parity(self) -> None:
        from tools.export_spatial_sr_portable import export_and_verify

        with tempfile.TemporaryDirectory() as directory:
            report = export_and_verify(Path(directory) / "spatial_sr_portable.pte")

        self.assertEqual(report["backend"], "portable")
        self.assertEqual(report["input_shape"], [1, 3, 64, 64])
        self.assertEqual(report["output_shape"], [1, 3, 128, 128])
        self.assertEqual(report["output_dtype"], "torch.float32")
        self.assertLessEqual(report["export_max_abs_difference"], 1e-6)
        self.assertLessEqual(report["runtime_max_abs_difference"], 1e-5)
        self.assertGreater(report["pte_size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
