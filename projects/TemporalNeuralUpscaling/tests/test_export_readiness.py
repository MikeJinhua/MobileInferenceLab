"""Tests for the local export-readiness preflight."""

import unittest

import torch

from analysis import collect_aten_operators, trace_inference_core, verify_trace_parity
from pipeline import create_deterministic_model


class ExportReadinessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = create_deterministic_model()
        self.traced = trace_inference_core(self.model, torch.rand(1, 3, 8, 12))

    def test_operator_inventory_is_minimal_and_explicit(self) -> None:
        self.assertEqual(
            collect_aten_operators(self.traced),
            {"aten::_convolution": 2, "aten::pixel_shuffle": 1, "aten::relu": 1},
        )

    def test_trace_matches_eager_for_multiple_shapes(self) -> None:
        shapes = ((1, 3, 8, 12), (1, 3, 9, 13), (2, 3, 10, 14))
        differences = verify_trace_parity(self.model, self.traced, shapes)
        self.assertEqual(set(differences), set(shapes))
        self.assertTrue(all(difference == 0.0 for difference in differences.values()))


if __name__ == "__main__":
    unittest.main()
