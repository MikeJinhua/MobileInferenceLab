"""Run deterministic 2x SR image inference and save comparison outputs."""

import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image

from pipeline import create_deterministic_model, create_synthetic_rgb_image, run_image_inference, save_image_results


def image_size(value: str) -> Tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must use WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="input image path")
    source.add_argument("--synthetic-size", type=image_size, metavar="WIDTHxHEIGHT", help="generate an original deterministic test image")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is not None:
        with Image.open(args.input) as loaded:
            image = loaded.convert("RGB")
        source_description = str(args.input)
    else:
        image = create_synthetic_rgb_image(*args.synthetic_size)
        source_description = f"deterministic synthetic {args.synthetic_size[0]}x{args.synthetic_size[1]}"

    outputs = run_image_inference(image, create_deterministic_model())
    paths = save_image_results(outputs, args.output_dir)
    print(f"Source: {source_description}")
    print("Model: untrained deterministic weights; output validates the pipeline, not SR quality")
    for name, path in paths.items():
        print(f"{name}: {path} ({outputs[name].width}x{outputs[name].height}, {outputs[name].mode})")


if __name__ == "__main__":
    main()
