"""Deterministic RGB image inference utilities for Phase 1."""

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw
import torch

from model import SpatialSR2x


DEFAULT_MODEL_SEED = 20260815


def create_synthetic_rgb_image(width: int = 96, height: int = 64) -> Image.Image:
    """Create an original deterministic RGB pattern with no external assets."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = (x * 5 + y * 3) % 256
    pixels[..., 1] = (x * 2 + y * 7) % 256
    pixels[..., 2] = ((x // 8 + y // 8) % 2) * 192 + 32

    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.rectangle((width // 8, height // 8, width // 2, height // 2), outline=(255, 255, 255), width=2)
    draw.line((0, height - 1, width - 1, 0), fill=(255, 224, 0), width=2)
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a Pillow image to a normalized float32 NCHW RGB tensor."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert one normalized CHW/NCHW RGB tensor to a Pillow RGB image."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("NCHW tensor must contain exactly one image")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("tensor must have shape [3, H, W] or [1, 3, H, W]")

    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    pixels = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(pixels)


def create_deterministic_model(seed: int = DEFAULT_MODEL_SEED) -> SpatialSR2x:
    """Construct an untrained model with reproducible initial weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SpatialSR2x()
    return model.eval()


def run_image_inference(image: Image.Image, model: SpatialSR2x) -> Dict[str, Image.Image]:
    """Run bilinear and neural 2x paths and return RGB images."""
    source = image.convert("RGB")
    input_tensor = image_to_tensor(source)
    with torch.inference_mode():
        output_tensor = model(input_tensor)

    target_size = (source.width * 2, source.height * 2)
    return {
        "input": source.copy(),
        "bilinear_2x": source.resize(target_size, Image.Resampling.BILINEAR),
        "neural_2x": tensor_to_image(output_tensor),
    }


def save_image_results(images: Dict[str, Image.Image], output_dir: Path) -> Dict[str, Path]:
    """Save the three required PNG outputs and return their paths."""
    required = ("input", "bilinear_2x", "neural_2x")
    missing = [name for name in required if name not in images]
    if missing:
        raise ValueError(f"missing required images: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}.png" for name in required}
    for name, path in paths.items():
        images[name].convert("RGB").save(path, format="PNG")
    return paths
