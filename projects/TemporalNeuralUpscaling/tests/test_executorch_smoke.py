"""Conditional automated test for the isolated ExecuTorch environment."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


EXECUTORCH_AVAILABLE = importlib.util.find_spec("executorch") is not None


@unittest.skipUnless(EXECUTORCH_AVAILABLE, "ExecuTorch is tested in .venv-executorch")
class ExecuTorchSmokeTest(unittest.TestCase):
    def test_add_export_and_runtime_parity(self) -> None:
        from tools.smoke_executorch import export_and_run

        with tempfile.TemporaryDirectory() as directory:
            report = export_and_run(Path(directory) / "add_xnnpack.pte")

        self.assertEqual(report["backend"], "XNNPACK")
        self.assertEqual(report["method_names"], ["forward"])
        self.assertEqual(report["output_value"], 4.0)
        self.assertEqual(report["export_max_abs_difference"], 0.0)
        self.assertEqual(report["runtime_max_abs_difference"], 0.0)
        self.assertGreater(report["pte_size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
