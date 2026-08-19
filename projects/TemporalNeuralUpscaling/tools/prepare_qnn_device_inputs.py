"""Create deterministic raw input/reference files for the QNN device runner."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pipeline import create_deterministic_model
from tools.export_spatial_sr_qnn import INPUT_SEED, INPUT_SHAPE, OUTPUT_SHAPE


def prepare(output_dir: Path) -> dict:
    generator = torch.Generator().manual_seed(INPUT_SEED)
    input_tensor = torch.rand(INPUT_SHAPE, generator=generator, dtype=torch.float32)
    model = create_deterministic_model().network.eval()
    with torch.inference_mode():
        reference = model(input_tensor)

    if tuple(reference.shape) != OUTPUT_SHAPE or not torch.isfinite(reference).all():
        raise AssertionError("invalid eager reference output")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.raw"
    reference_path = output_dir / "eager_reference.raw"
    input_list_path = output_dir / "input_list.txt"
    input_tensor.numpy().astype("<f4", copy=False).tofile(input_path)
    reference.numpy().astype("<f4", copy=False).tofile(reference_path)
    input_list_path.write_text("input.raw\n" * 20, encoding="ascii", newline="\n")

    report = {
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "dtype": "float32-little-endian",
        "input_seed": INPUT_SEED,
        "input_bytes": input_path.stat().st_size,
        "reference_bytes": reference_path.stat().st_size,
        "reference_sum": float(reference.double().sum().item()),
        "reference_min": float(reference.min().item()),
        "reference_max": float(reference.max().item()),
    }
    (output_dir / "reference_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/p4_4"))
    return parser.parse_args()


def main() -> None:
    report = prepare(parse_args().output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
