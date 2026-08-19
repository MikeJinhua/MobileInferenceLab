"""Tests for the static QNN HTP FP16 export contract."""

import unittest

from tools.export_spatial_sr_qnn import validate_report


class SpatialSrQnnContractTest(unittest.TestCase):
    def test_valid_fully_delegated_fp16_report(self) -> None:
        report = {
            "target_soc": "SM8550",
            "target_htp": "v73",
            "precision": "fp16",
            "input_shape": [1, 3, 64, 64],
            "output_shape": [1, 3, 128, 128],
            "delegate_count": 1,
            "delegate_backend_ids": ["QnnBackend"],
            "portable_fallback_operators": {},
            "fully_delegated": True,
            "eager_output_finite": True,
            "repeat_max_abs_difference": 0.0,
            "export_max_abs_difference": 0.0,
        }
        validate_report(report)

    def test_fallback_is_rejected(self) -> None:
        report = {
            "target_soc": "SM8550",
            "target_htp": "v73",
            "precision": "fp16",
            "input_shape": [1, 3, 64, 64],
            "output_shape": [1, 3, 128, 128],
            "delegate_count": 1,
            "delegate_backend_ids": ["QnnBackend"],
            "portable_fallback_operators": {"aten.pixel_shuffle.default": 1},
            "fully_delegated": False,
            "eager_output_finite": True,
            "repeat_max_abs_difference": 0.0,
            "export_max_abs_difference": 0.0,
        }
        with self.assertRaisesRegex(AssertionError, "portable fallback"):
            validate_report(report)


if __name__ == "__main__":
    unittest.main()
