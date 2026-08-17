"""Generate the ignored static XNNPACK model asset used by the Android app."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import torch

from pipeline import create_deterministic_model
from tools.export_spatial_sr_xnnpack import export_and_verify


DEFAULT_OUTPUT = Path("android/app/src/main/assets/spatial_sr_xnnpack.pte")
DEFAULT_REPORT = Path("results/p3_3/android_model_report.json")


def reference_statistics() -> Dict[str, float]:
    values = torch.tensor(
        [((index * 37) % 256) / 255.0 for index in range(3 * 64 * 64)],
        dtype=torch.float32,
    ).reshape(1, 3, 64, 64)
    with torch.inference_mode():
        output = create_deterministic_model().network(values)
    return {
        "minimum": float(output.min().item()),
        "maximum": float(output.max().item()),
        "checksum_float64_accumulation": float(output.double().sum().item()),
    }


def prepare_model(output: Path) -> Dict[str, object]:
    report = export_and_verify(output)
    report["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report["android_asset_name"] = output.name
    report["deterministic_input_reference"] = reference_statistics()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_model(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2) + "\n"
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
