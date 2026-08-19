"""Lower static SpatialSR2x to a QNN HTP FP16 ExecuTorch program."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import operator
from pathlib import Path
from typing import Dict, Mapping

import torch

from pipeline import create_deterministic_model


INPUT_SHAPE = (1, 3, 64, 64)
OUTPUT_SHAPE = (1, 3, 128, 128)
MODEL_SEED = 20260815
INPUT_SEED = 20260820
TARGET_SOC = "SM8550"
TARGET_HTP = "v73"


def operator_inventory(exported_program: torch.export.ExportedProgram) -> Dict[str, int]:
    operators = Counter(
        str(node.target)
        for node in exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    )
    return dict(sorted(operators.items()))


def partition_inventory(edge_program) -> Dict[str, object]:
    delegate_target = torch.ops.higher_order.executorch_call_delegate
    graph_module = edge_program.exported_program().graph_module
    delegate_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == delegate_target
    ]
    ignored_targets = {delegate_target, operator.getitem}
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


def validate_report(report: Mapping[str, object]) -> None:
    if report["target_soc"] != TARGET_SOC or report["target_htp"] != TARGET_HTP:
        raise AssertionError("unexpected QNN target")
    if report["precision"] != "fp16":
        raise AssertionError("QNN lowering must use HTP FP16")
    if tuple(report["input_shape"]) != INPUT_SHAPE:
        raise AssertionError(f"unexpected input shape: {report['input_shape']}")
    if tuple(report["output_shape"]) != OUTPUT_SHAPE:
        raise AssertionError(f"unexpected output shape: {report['output_shape']}")
    if report["delegate_count"] < 1 or report["delegate_backend_ids"] != ["QnnBackend"]:
        raise AssertionError("QNN delegation was not observed")
    if report["portable_fallback_operators"]:
        raise AssertionError(f"portable fallback observed: {report['portable_fallback_operators']}")
    if not report["fully_delegated"]:
        raise AssertionError("the static graph is not fully delegated")
    if not report["eager_output_finite"] or report["repeat_max_abs_difference"] != 0.0:
        raise AssertionError("deterministic eager output contract failed")
    if report["export_max_abs_difference"] > 1e-6:
        raise AssertionError(f"torch.export parity failed: {report['export_max_abs_difference']}")


def export_and_verify(output: Path) -> Dict[str, object]:
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    model = create_deterministic_model().network.eval()
    generator = torch.Generator().manual_seed(INPUT_SEED)
    input_tensor = torch.rand(INPUT_SHAPE, generator=generator, dtype=torch.float32)

    with torch.inference_mode():
        eager_output = model(input_tensor)
        repeated_output = model(input_tensor)
        exported_program = torch.export.export(model, (input_tensor,))
        exported_output = exported_program.module()(input_tensor)

    backend_options = generate_htp_compiler_spec(use_fp16=True)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8550,
        backend_options=backend_options,
    )
    edge_program = to_edge_transform_and_lower_to_qnn(
        model,
        (input_tensor,),
        compiler_specs,
    )
    partition = partition_inventory(edge_program)
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=True,
                alloc_graph_output=True,
            ),
            segment_alignment=get_qnn_context_binary_alignment(),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(executorch_program.buffer)

    export_difference = torch.abs(eager_output - exported_output)
    repeat_difference = torch.abs(eager_output - repeated_output)
    report = {
        "torch_version": torch.__version__,
        "executorch_version": version("executorch"),
        "backend": "qnn_htp",
        "precision": "fp16",
        "target_soc": TARGET_SOC,
        "target_htp": TARGET_HTP,
        "model": "SpatialSR2x.network",
        "model_seed": MODEL_SEED,
        "input_seed": INPUT_SEED,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(eager_output.shape),
        "input_dtype": str(input_tensor.dtype),
        "output_dtype": str(eager_output.dtype),
        "eager_output_finite": bool(torch.isfinite(eager_output).all().item()),
        "exported_operators": operator_inventory(exported_program),
        **partition,
        "export_max_abs_difference": float(export_difference.max().item()),
        "repeat_max_abs_difference": float(repeat_difference.max().item()),
        "host_qnn_runtime_executed": False,
        "host_runtime_note": "The offline SM8550 HTP context is device-targeted; execution is verified on Android in P4.4.",
        "pte_path": str(output),
        "pte_size_bytes": output.stat().st_size,
    }
    validate_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/p4_3/spatial_sr_qnn_htp_fp16.pte"))
    parser.add_argument("--report", type=Path, default=Path("results/p4_3/qnn_htp_fp16_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_and_verify(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
