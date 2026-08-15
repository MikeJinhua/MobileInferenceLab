"""Local export-readiness checks that do not require ExecuTorch."""

from collections import Counter
from typing import Dict, Iterable, Tuple

import torch

from model import SpatialSR2x


InputShape = Tuple[int, int, int, int]


def trace_inference_core(model: SpatialSR2x, example: torch.Tensor) -> torch.jit.ScriptModule:
    """Trace and freeze the tensor-only inference core for operator inspection."""
    model.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(model.network, example, strict=True)
    return torch.jit.freeze(traced)


def collect_aten_operators(traced: torch.jit.ScriptModule) -> Dict[str, int]:
    """Return a stable, sorted ATen operator frequency map."""
    counts = Counter(node.kind() for node in traced.inlined_graph.nodes() if node.kind().startswith("aten::"))
    return dict(sorted(counts.items()))


def verify_trace_parity(
    model: SpatialSR2x,
    traced: torch.jit.ScriptModule,
    shapes: Iterable[InputShape],
) -> Dict[InputShape, float]:
    """Compare eager and traced inference cores for representative shapes."""
    differences: Dict[InputShape, float] = {}
    generator = torch.Generator().manual_seed(20260815)
    with torch.inference_mode():
        for shape in shapes:
            image = torch.rand(shape, generator=generator)
            eager = model(image)
            exported = traced(image)
            if eager.shape != exported.shape:
                raise AssertionError(f"shape mismatch for {shape}: {eager.shape} != {exported.shape}")
            differences[shape] = float(torch.max(torch.abs(eager - exported)).item())
    return differences
