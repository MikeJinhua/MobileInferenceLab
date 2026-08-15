"""Automated tests for the Phase 1 spatial model."""

import unittest

import torch

from model import SpatialSR2x


class SpatialSR2xTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = SpatialSR2x().eval()

    def test_doubles_spatial_dimensions_and_preserves_batch_rgb(self) -> None:
        image = torch.rand(2, 3, 12, 20)
        with torch.inference_mode():
            output = self.model(image)
        self.assertEqual(tuple(output.shape), (2, 3, 24, 40))
        self.assertEqual(output.dtype, image.dtype)

    def test_accepts_odd_spatial_dimensions(self) -> None:
        image = torch.rand(1, 3, 7, 11)
        with torch.inference_mode():
            output = self.model(image)
        self.assertEqual(tuple(output.shape), (1, 3, 14, 22))

    def test_rejects_non_nchw_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            self.model(torch.rand(3, 12, 20))

    def test_rejects_non_rgb_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "three RGB channels"):
            self.model(torch.rand(1, 1, 12, 20))

    def test_rejects_invalid_feature_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            SpatialSR2x(feature_channels=0)


if __name__ == "__main__":
    unittest.main()
