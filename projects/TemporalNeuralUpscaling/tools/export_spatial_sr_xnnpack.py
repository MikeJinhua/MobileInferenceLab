"""Export static SpatialSR2x with XNNPACK and verify delegation and parity."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
import operator
from pathlib import Path
from typing import Dict

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from pipeline import create_deterministic_model
from tools.executorch_utils import configure_bundled_flatc


INPUT_SHAPE = (1, 3, 64, 64)
DELEGATE_TARGET = torch.ops.higher_order.executorch_call_delegate


def operator_inventory(exported_program: torch.export.ExportedProgram) -> Dict[str, int]:
    operators = Counter(
        str(node.target)
        for node in exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    )
    return dict(sorted(operators.items()))


def partition_inventory(edge_program) -> Dict[str, object]:
    graph_module = edge_program.exported_program().graph_module
    delegate_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == DELEGATE_TARGET
    ]
    ignored_targets = {DELEGATE_TARGET, operator.getitem}
    fallback = Counter(
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target not in ignored_targets
    )
    backend_ids = sorted(
        {
            str(module.backend_id)
            for _, module in graph_module.named_modules()
            if hasattr(module, "backend_id")
        }
    )
    return {
        "delegate_count": len(delegate_nodes),
        "delegate_backend_ids": backend_ids,
        "portable_fallback_operators": dict(sorted(fallback.items())),
        "fully_delegated": len(delegate_nodes) > 0 and not fallback,
    }


def export_and_verify(output: Path) -> Dict[str, object]:
    flatc = configure_bundled_flatc()
    inference_core = create_deterministic_model().network
    generator = torch.Generator().manual_seed(20260815)
    input_tensor = torch.rand(INPUT_SHAPE, generator=generator, dtype=torch.float32)

    with torch.inference_mode():
        eager_output = inference_core(input_tensor)
        exported_program = torch.export.export(inference_core, (input_tensor,))
        exported_output = exported_program.module()(input_tensor)

    edge_program = to_edge_transform_and_lower(
        exported_program, partitioner=[XnnpackPartitioner()]
    )
    partition = partition_inventory(edge_program)
    program = edge_program.to_executorch()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(program.buffer)

    runtime_program = Runtime.get().load_program(output)
    runtime_output = runtime_program.load_method("forward").execute((input_tensor,))[0]
    export_difference = torch.abs(eager_output - exported_output)
    runtime_difference = torch.abs(eager_output - runtime_output)
    report = {
        "torch_version": torch.__version__,
        "executorch_version": version("executorch"),
        "backend": "xnnpack",
        "flatc_path": str(flatc),
        "model": "SpatialSR2x.network",
        "model_seed": 20260815,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(runtime_output.shape),
        "input_dtype": str(input_tensor.dtype),
        "output_dtype": str(runtime_output.dtype),
        "exported_operators": operator_inventory(exported_program),
        **partition,
        "method_names": list(runtime_program.method_names),
        "export_max_abs_difference": float(export_difference.max().item()),
        "runtime_max_abs_difference": float(runtime_difference.max().item()),
        "runtime_mean_abs_difference": float(runtime_difference.mean().item()),
        "pte_path": str(output),
        "pte_size_bytes": output.stat().st_size,
    }

    if not report["fully_delegated"] or report["delegate_backend_ids"] != ["XnnpackBackend"]:
        raise AssertionError(f"unexpected XNNPACK partition: {partition}")
    if tuple(runtime_output.shape) != (1, 3, 128, 128):
        raise AssertionError(f"unexpected runtime output shape: {tuple(runtime_output.shape)}")
    if runtime_output.dtype != torch.float32:
        raise AssertionError(f"unexpected runtime output dtype: {runtime_output.dtype}")
    if report["export_max_abs_difference"] > 1e-6:
        raise AssertionError(f"torch.export parity failed: {report['export_max_abs_difference']}")
    if report["runtime_max_abs_difference"] > 1e-5:
        raise AssertionError(f"XNNPACK runtime parity failed: {report['runtime_max_abs_difference']}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/p2_4/spatial_sr_xnnpack.pte"))
    parser.add_argument("--report", type=Path, default=Path("results/p2_4/xnnpack_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_and_verify(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
