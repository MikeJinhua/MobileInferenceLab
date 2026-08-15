"""Conditional bounded-dynamic ExecuTorch test for SpatialSR2x."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


EXECUTORCH_AVAILABLE = importlib.util.find_spec("executorch") is not None


@unittest.skipUnless(EXECUTORCH_AVAILABLE, "ExecuTorch is tested in .venv-executorch")
class SpatialSRDynamicTest(unittest.TestCase):
    def test_bounded_dynamic_portable_and_xnnpack(self) -> None:
        from tools.export_spatial_sr_dynamic import TEST_SHAPES, export_and_verify

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            report = export_and_verify(Path(directory))

        self.assertEqual(report["tensor_contract"]["height_min"], 16)
        self.assertEqual(report["tensor_contract"]["height_max"], 128)
        self.assertEqual(len(report["out_of_bounds_shapes_rejected_by_export"]), 2)
        for backend in ("portable", "xnnpack"):
            results = report["backends"][backend]["shape_results"]
            self.assertEqual(len(results), len(TEST_SHAPES))
            self.assertTrue(all(item["max_abs_difference"] <= 1e-5 for item in results))

        xnnpack = report["backends"]["xnnpack"]
        self.assertEqual(xnnpack["delegate_count"], 3)
        self.assertEqual(xnnpack["delegate_backend_ids"], ["XnnpackBackend"])
        self.assertEqual(sum(xnnpack["portable_fallback_operators"].values()), 2)


if __name__ == "__main__":
    unittest.main()
