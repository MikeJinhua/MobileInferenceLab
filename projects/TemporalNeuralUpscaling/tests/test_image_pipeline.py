"""Tests for deterministic RGB image inference and persistence."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from pipeline import (
    create_deterministic_model,
    create_synthetic_rgb_image,
    image_to_tensor,
    run_image_inference,
    save_image_results,
    tensor_to_image,
)


class ImagePipelineTest(unittest.TestCase):
    def test_rgb_conversion_and_tensor_shape(self) -> None:
        grayscale = Image.new("L", (13, 9), color=127)
        tensor = image_to_tensor(grayscale)
        self.assertEqual(tuple(tensor.shape), (1, 3, 9, 13))
        self.assertEqual(tensor.dtype, torch.float32)
        restored = tensor_to_image(tensor)
        self.assertEqual(restored.mode, "RGB")
        self.assertEqual(restored.size, (13, 9))

    def test_pipeline_output_shapes_and_rgb_channels(self) -> None:
        outputs = run_image_inference(create_synthetic_rgb_image(17, 11), create_deterministic_model())
        self.assertEqual(outputs["input"].size, (17, 11))
        self.assertEqual(outputs["bilinear_2x"].size, (34, 22))
        self.assertEqual(outputs["neural_2x"].size, (34, 22))
        self.assertTrue(all(image.mode == "RGB" for image in outputs.values()))

    def test_model_and_output_are_deterministic(self) -> None:
        image = create_synthetic_rgb_image(16, 12)
        first = np.asarray(run_image_inference(image, create_deterministic_model())["neural_2x"])
        second = np.asarray(run_image_inference(image, create_deterministic_model())["neural_2x"])
        np.testing.assert_array_equal(first, second)

    def test_save_and_load_preserves_rgb_pixels(self) -> None:
        outputs = run_image_inference(create_synthetic_rgb_image(10, 8), create_deterministic_model())
        with tempfile.TemporaryDirectory() as directory:
            paths = save_image_results(outputs, Path(directory))
            self.assertEqual(set(paths), {"input", "bilinear_2x", "neural_2x"})
            for name, path in paths.items():
                self.assertTrue(path.is_file())
                with Image.open(path) as loaded:
                    self.assertEqual(loaded.mode, "RGB")
                    np.testing.assert_array_equal(np.asarray(loaded), np.asarray(outputs[name]))


if __name__ == "__main__":
    unittest.main()
