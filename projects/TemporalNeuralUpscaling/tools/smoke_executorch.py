"""Export and execute the official-style minimal Add model with ExecuTorch."""

import argparse
import importlib.resources
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import Dict

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


class Add(torch.nn.Module):
    """Minimal model used only to validate the ExecuTorch toolchain."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


def configure_bundled_flatc() -> Path:
    """Point ExecuTorch 1.3's Windows serializer at its bundled flatc.exe."""
    configured = os.getenv("FLATC_EXECUTABLE")
    if configured:
        return Path(configured)

    resource = importlib.resources.files("executorch.data.bin").joinpath("flatc.exe")
    if not resource.is_file():
        raise FileNotFoundError("ExecuTorch wheel does not contain executorch/data/bin/flatc.exe")
    flatc = Path(str(resource))
    os.environ["FLATC_EXECUTABLE"] = str(flatc)
    return flatc


def export_and_run(output: Path) -> Dict[str, object]:
    flatc = configure_bundled_flatc()
    model = Add().eval()
    inputs = (torch.tensor([1.25], dtype=torch.float32), torch.tensor([2.75], dtype=torch.float32))

    with torch.inference_mode():
        eager_output = model(*inputs)
        exported_program = torch.export.export(model, inputs)
        exported_output = exported_program.module()(*inputs)

    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(program.buffer)

    runtime_program = Runtime.get().load_program(output)
    runtime_method = runtime_program.load_method("forward")
    runtime_output = runtime_method.execute(inputs)[0]

    export_difference = float(torch.max(torch.abs(eager_output - exported_output)).item())
    runtime_difference = float(torch.max(torch.abs(eager_output - runtime_output)).item())
    if export_difference != 0.0 or runtime_difference != 0.0:
        raise AssertionError(
            f"Add parity failed: export={export_difference}, runtime={runtime_difference}"
        )

    return {
        "python_package": "executorch",
        "torch_version": torch.__version__,
        "executorch_version": version("executorch"),
        "backend": "XNNPACK",
        "flatc_path": str(flatc),
        "method_names": list(runtime_program.method_names),
        "input_values": [float(value.item()) for value in inputs],
        "output_value": float(runtime_output.item()),
        "export_max_abs_difference": export_difference,
        "runtime_max_abs_difference": runtime_difference,
        "pte_path": str(output),
        "pte_size_bytes": output.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/p2_2/add_xnnpack.pte"))
    parser.add_argument("--report", type=Path, default=Path("results/p2_2/smoke_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_and_run(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
