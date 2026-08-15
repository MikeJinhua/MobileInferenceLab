"""Reusable PC benchmark implementation for the Phase 1 image pipeline."""

from dataclasses import asdict, dataclass
import os
import platform
import statistics
import time
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from PIL import Image
import torch

from pipeline import create_deterministic_model, create_synthetic_rgb_image, image_to_tensor, tensor_to_image


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def percentile(samples: Sequence[float], percentage: float) -> float:
    """Return a linearly interpolated percentile without extra dependencies."""
    if not samples:
        raise ValueError("samples must not be empty")
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples_ms: Sequence[float]) -> LatencyStats:
    if not samples_ms:
        raise ValueError("samples must not be empty")
    return LatencyStats(
        mean_ms=statistics.fmean(samples_ms),
        median_ms=statistics.median(samples_ms),
        p90_ms=percentile(samples_ms, 90.0),
        p95_ms=percentile(samples_ms, 95.0),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
    )


def measure(operation: Callable[[], object], warmup: int, iterations: int) -> LatencyStats:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    for _ in range(warmup):
        operation()
    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        operation()
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return summarize(samples_ms)


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _os_name() -> str:
    if platform.system() == "Windows":
        version = platform.version()
        parts = version.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            build = int(parts[2])
            product = "Windows 11" if build >= 22000 else "Windows 10"
            return f"{product} {version}"
    return platform.platform()


def environment_metadata(threads: int) -> Dict[str, object]:
    return {
        "os": _os_name(),
        "cpu": _cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "layout": "NCHW RGB",
        "torch_threads": threads,
    }


def benchmark_size(width: int, height: int, warmup: int, iterations: int) -> Dict[str, object]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    source = create_synthetic_rgb_image(width, height)
    model = create_deterministic_model()
    input_tensor = image_to_tensor(source)
    with torch.inference_mode():
        output_tensor = model(input_tensor)

        stages = {
            "rgb_to_tensor": measure(lambda: image_to_tensor(source), warmup, iterations),
            "bilinear_2x": measure(
                lambda: source.resize((width * 2, height * 2), Image.Resampling.BILINEAR),
                warmup,
                iterations,
            ),
            "model_inference": measure(lambda: model(input_tensor), warmup, iterations),
            "tensor_to_rgb": measure(lambda: tensor_to_image(output_tensor), warmup, iterations),
            "neural_end_to_end": measure(
                lambda: tensor_to_image(model(image_to_tensor(source))),
                warmup,
                iterations,
            ),
        }

    return {
        "input_size": [width, height],
        "output_size": [width * 2, height * 2],
        "warmup": warmup,
        "iterations": iterations,
        "stages": {name: asdict(stats) for name, stats in stages.items()},
    }


def run_pc_baseline(
    sizes: Iterable[Tuple[int, int]], warmup: int, iterations: int, threads: int
) -> Dict[str, object]:
    if threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    model = create_deterministic_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return {
        "schema_version": 1,
        "scope": "local PC CPU baseline; PNG I/O excluded; untrained deterministic model",
        "environment": environment_metadata(threads),
        "model": {
            "name": "SpatialSR2x",
            "scale_factor": 2,
            "parameter_count": parameter_count,
            "parameter_bytes_fp32": parameter_bytes,
        },
        "results": [benchmark_size(width, height, warmup, iterations) for width, height in sizes],
    }
