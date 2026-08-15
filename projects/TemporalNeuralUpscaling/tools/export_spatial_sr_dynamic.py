"""Export bounded-dynamic SpatialSR2x artifacts and verify multiple shapes."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
import operator
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from pipeline import create_deterministic_model
from tools.executorch_utils import configure_bundled_flatc


EXAMPLE_SHAPE = (1, 3, 64, 64)
HEIGHT_BOUNDS = (16, 128)
WIDTH_BOUNDS = (16, 128)
TEST_SHAPES = (
    (1, 3, 16, 16),
    (1, 3, 17, 31),
    (1, 3, 64, 64),
    (1, 3, 96, 48),
    (1, 3, 128, 128),
)
DELEGATE_TARGET = torch.ops.higher_order.executorch_call_delegate


def _dynamic_shapes() -> tuple:
    height = torch.export.Dim("height", min=HEIGHT_BOUNDS[0], max=HEIGHT_BOUNDS[1])
    width = torch.export.Dim("width", min=WIDTH_BOUNDS[0], max=WIDTH_BOUNDS[1])
    return ({2: height, 3: width},)


def _partition_inventory(edge_program) -> Dict[str, object]:
    graph_module = edge_program.exported_program().graph_module
    call_targets = [
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    delegate_count = sum(target == DELEGATE_TARGET for target in call_targets)
    ignored = {
        DELEGATE_TARGET,
        operator.getitem,
        operator.mul,
        torch.ops.aten.sym_size.int,
    }
    fallback = Counter(str(target) for target in call_targets if target not in ignored)
    shape_ops = Counter(
        str(target)
        for target in call_targets
        if target in {operator.mul, torch.ops.aten.sym_size.int}
    )
    backend_ids = sorted(
        {
            str(module.backend_id)
            for _, module in graph_module.named_modules()
            if hasattr(module, "backend_id")
        }
    )
    return {
        "delegate_count": delegate_count,
        "delegate_backend_ids": backend_ids,
        "portable_fallback_operators": dict(sorted(fallback.items())),
        "symbolic_shape_operators": dict(sorted(shape_ops.items())),
    }


def _verify_shapes(model, method, shapes: Iterable[Tuple[int, ...]]) -> list:
    results = []
    for index, shape in enumerate(shapes):
        generator = torch.Generator().manual_seed(20260815 + index)
        input_tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
        with torch.inference_mode():
            eager_output = model(input_tensor)
        runtime_output = method.execute((input_tensor,))[0]
        difference = torch.abs(eager_output - runtime_output)
        expected_shape = (shape[0], 3, shape[2] * 2, shape[3] * 2)
        if tuple(runtime_output.shape) != expected_shape:
            raise AssertionError(
                f"unexpected output for {shape}: {tuple(runtime_output.shape)}"
            )
        maximum = float(difference.max().item())
        if maximum > 1e-5:
            raise AssertionError(f"runtime parity failed for {shape}: {maximum}")
        results.append(
            {
                "input_shape": list(shape),
                "output_shape": list(runtime_output.shape),
                "max_abs_difference": maximum,
                "mean_abs_difference": float(difference.mean().item()),
            }
        )
    return results


def _verify_export_bounds(exported) -> list:
    rejected = []
    for shape in ((1, 3, 15, 64), (1, 3, 64, 129)):
        try:
            exported.module()(torch.rand(shape))
        except RuntimeError:
            rejected.append(list(shape))
        else:
            raise AssertionError(f"torch.export accepted out-of-bounds shape: {shape}")
    return rejected


def export_and_verify(output_dir: Path) -> Dict[str, object]:
    flatc = configure_bundled_flatc()
    model = create_deterministic_model().network
    example = torch.rand(
        EXAMPLE_SHAPE, generator=torch.Generator().manual_seed(20260815)
    )
    exported = torch.export.export(
        model, (example,), dynamic_shapes=_dynamic_shapes()
    )
    rejected_shapes = _verify_export_bounds(exported)

    output_dir.mkdir(parents=True, exist_ok=True)
    backend_reports = {}
    for backend, partitioners in (
        ("portable", None),
        ("xnnpack", [XnnpackPartitioner()]),
    ):
        edge_program = to_edge_transform_and_lower(
            exported, partitioner=partitioners
        )
        partition = _partition_inventory(edge_program)
        output = output_dir / f"spatial_sr_dynamic_{backend}.pte"
        output.write_bytes(edge_program.to_executorch().buffer)
        runtime_program = Runtime.get().load_program(output)
        method = runtime_program.load_method("forward")
        shape_results = _verify_shapes(model, method, TEST_SHAPES)
        backend_reports[backend] = {
            **partition,
            "shape_results": shape_results,
            "pte_path": str(output),
            "pte_size_bytes": output.stat().st_size,
        }

    xnnpack = backend_reports["xnnpack"]
    if xnnpack["delegate_count"] != 3:
        raise AssertionError(f"unexpected dynamic delegate count: {xnnpack}")
    if xnnpack["delegate_backend_ids"] != ["XnnpackBackend"]:
        raise AssertionError(f"unexpected dynamic backend: {xnnpack}")
    return {
        "torch_version": torch.__version__,
        "executorch_version": version("executorch"),
        "flatc_path": str(flatc),
        "model": "SpatialSR2x.network",
        "model_seed": 20260815,
        "tensor_contract": {
            "batch": 1,
            "channels": 3,
            "dtype": "torch.float32",
            "height_min": HEIGHT_BOUNDS[0],
            "height_max": HEIGHT_BOUNDS[1],
            "width_min": WIDTH_BOUNDS[0],
            "width_max": WIDTH_BOUNDS[1],
        },
        "example_shape": list(EXAMPLE_SHAPE),
        "out_of_bounds_shapes_rejected_by_export": rejected_shapes,
        "backends": backend_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/p2_5"))
    parser.add_argument("--report", type=Path, default=Path("results/p2_5/dynamic_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_and_verify(args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
