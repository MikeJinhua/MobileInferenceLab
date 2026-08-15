"""Conditional XNNPACK delegation test for SpatialSR2x."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


EXECUTORCH_AVAILABLE = importlib.util.find_spec("executorch") is not None


@unittest.skipUnless(EXECUTORCH_AVAILABLE, "ExecuTorch is tested in .venv-executorch")
class SpatialSRXnnpackTest(unittest.TestCase):
    def test_static_export_is_fully_delegated_and_matches_eager(self) -> None:
        from tools.export_spatial_sr_xnnpack import export_and_verify

        with tempfile.TemporaryDirectory() as directory:
            report = export_and_verify(Path(directory) / "spatial_sr_xnnpack.pte")

        self.assertEqual(report["backend"], "xnnpack")
        self.assertEqual(report["delegate_count"], 1)
        self.assertEqual(report["delegate_backend_ids"], ["XnnpackBackend"])
        self.assertEqual(report["portable_fallback_operators"], {})
        self.assertTrue(report["fully_delegated"])
        self.assertEqual(report["output_shape"], [1, 3, 128, 128])
        self.assertLessEqual(report["runtime_max_abs_difference"], 1e-5)
        self.assertGreater(report["pte_size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
