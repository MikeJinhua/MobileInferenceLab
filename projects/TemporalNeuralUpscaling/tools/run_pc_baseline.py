"""Run the Phase 1 PC CPU baseline and write JSON/Markdown reports."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from benchmark.pipeline_benchmark import run_pc_baseline


def parse_size(value: str) -> Tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must use WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("dimensions must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=[(64, 64), (320, 180), (960, 540)])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json-output", type=Path, default=Path("docs/PC_BASELINE.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/PC_BASELINE.md"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.threads <= 0:
        parser.error("warmup must be non-negative; iterations and threads must be positive")
    return args


def render_markdown(report: Dict[str, object]) -> str:
    environment = report["environment"]
    model = report["model"]
    rows: List[str] = []
    for result in report["results"]:
        size = f'{result["input_size"][0]}x{result["input_size"][1]}'
        for stage, stats in result["stages"].items():
            rows.append(
                f'| {size} | `{stage}` | {stats["mean_ms"]:.3f} | {stats["median_ms"]:.3f} | '
                f'{stats["p90_ms"]:.3f} | {stats["p95_ms"]:.3f} | {stats["min_ms"]:.3f} | {stats["max_ms"]:.3f} |'
            )
    table = "\n".join(rows)
    observations = []
    diagnostic_stages = ("rgb_to_tensor", "model_inference", "tensor_to_rgb")
    for result in report["results"]:
        width, height = result["input_size"]
        stages = result["stages"]
        dominant = max(diagnostic_stages, key=lambda name: stages[name]["median_ms"])
        end_to_end = stages["neural_end_to_end"]["median_ms"]
        share = stages[dominant]["median_ms"] / end_to_end * 100.0
        observations.append(
            f'- {width}x{height}: `{dominant}` is the largest independently measured neural stage '
            f'({stages[dominant]["median_ms"]:.3f} ms, {share:.1f}% of the directly measured end-to-end median).'
        )
    observation_text = "\n".join(observations)
    first = report["results"][0]
    return f"""# PC Baseline Report

This is a local Phase 1 CPU baseline for pipeline engineering. `SpatialSR2x` has deterministic random weights, so these measurements do not represent SR image quality or a trained final model.

## Environment

- OS: `{environment["os"]}`
- CPU: `{environment["cpu"]}`
- Logical CPUs: {environment["logical_cpu_count"]}
- Python: `{environment["python"]}`
- PyTorch: `{environment["pytorch"]}`
- Device / dtype / layout: `{environment["device"]}` / `{environment["dtype"]}` / `{environment["layout"]}`
- PyTorch threads: {environment["torch_threads"]}
- Model parameters: {model["parameter_count"]:,} ({model["parameter_bytes_fp32"]:,} FP32 bytes)
- Warmup / measured iterations per stage: {first["warmup"]} / {first["iterations"]}

## Results

| Input | Stage | Mean ms | Median ms | P90 ms | P95 ms | Min ms | Max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Stage definitions:

- `rgb_to_tensor`: Pillow RGB to normalized contiguous float32 NCHW tensor.
- `bilinear_2x`: Pillow bilinear baseline only.
- `model_inference`: eager PyTorch CPU model only, with a prepared tensor.
- `tensor_to_rgb`: prepared output tensor to Pillow RGB.
- `neural_end_to_end`: RGB-to-tensor + inference + tensor-to-RGB, measured directly.

PNG load/save, application UI, memory transfer to another device, synchronization, display, and sustained thermal/power behavior are excluded. Stages were measured independently; their distribution statistics must not be summed as a substitute for `neural_end_to_end`.

## Interpretation

This baseline is useful for regression checks and later platform comparisons. It is not a PC-versus-mobile performance claim: Android CPU, ExecuTorch, QNN/NPU, Vulkan transfer, precision changes, and trained-model effects remain unmeasured.

{observation_text}

The percentages compare independently measured stage medians with the direct end-to-end median, so they are diagnostic approximations rather than an additive latency breakdown.

Machine-readable data: [PC_BASELINE.json](PC_BASELINE.json).
"""


def main() -> None:
    args = parse_args()
    report = run_pc_baseline(args.sizes, args.warmup, args.iterations, args.threads)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON report: {args.json_output}")
    print(f"Markdown report: {args.markdown_output}")
    for result in report["results"]:
        width, height = result["input_size"]
        inference = result["stages"]["model_inference"]["median_ms"]
        end_to_end = result["stages"]["neural_end_to_end"]["median_ms"]
        print(f"{width}x{height}: inference median={inference:.3f} ms; end-to-end median={end_to_end:.3f} ms")


if __name__ == "__main__":
    main()
