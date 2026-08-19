"""Compare QNN device output with the deterministic eager reference."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from tools.export_spatial_sr_qnn import OUTPUT_SHAPE


def _timestamp_seconds(line: str) -> float:
    match = re.match(r"[A-Z] (\d+):(\d+):(\d+\.\d+)", line)
    if not match:
        raise AssertionError(f"missing runner timestamp: {line}")
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


def analyze(
    reference_path: Path,
    device_path: Path,
    repeated_path: Path,
    speed_path: Path,
    runner_log_path: Path | None = None,
    memory_path: Path | None = None,
) -> dict:
    reference = np.fromfile(reference_path, dtype="<f4")
    device = np.fromfile(device_path, dtype="<f4")
    repeated = np.fromfile(repeated_path, dtype="<f4")
    expected_elements = int(np.prod(OUTPUT_SHAPE))
    if reference.size != expected_elements or device.size != expected_elements or repeated.size != expected_elements:
        raise AssertionError(
            f"expected {expected_elements} values, got reference={reference.size}, device={device.size}, repeated={repeated.size}"
        )
    if not np.isfinite(device).all():
        raise AssertionError("device output contains non-finite values")

    difference = np.abs(reference.astype(np.float64) - device.astype(np.float64))
    repeat_difference = np.abs(device.astype(np.float64) - repeated.astype(np.float64))
    timed_block_ms = float(speed_path.read_text(encoding="ascii").strip())
    report = {
        "output_shape": list(OUTPUT_SHAPE),
        "output_dtype": "float32-little-endian",
        "output_finite": True,
        "device_min": float(device.min()),
        "device_max": float(device.max()),
        "device_sum": float(device.astype(np.float64).sum()),
        "reference_sum": float(reference.astype(np.float64).sum()),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "repeat_max_abs_difference": float(repeat_difference.max()),
        "warmup_iterations": 5,
        "timed_iterations_per_sample": 20,
        "timed_samples": 20,
        "runner_average_timed_block_ms": timed_block_ms,
        "runner_average_inference_ms": timed_block_ms / 20.0,
    }
    if runner_log_path:
        lines = runner_log_path.read_text(encoding="utf-8").splitlines()
        model_line = next(line for line in lines if "Model file " in line and " is loaded." in line)
        method_line = next(line for line in lines if "Method loaded." in line)
        report["model_file_open_ms_from_process_start"] = _timestamp_seconds(model_line) * 1000.0
        report["method_ready_ms_from_process_start"] = _timestamp_seconds(method_line) * 1000.0
        report["backend_and_method_load_ms"] = (
            report["method_ready_ms_from_process_start"] - report["model_file_open_ms_from_process_start"]
        )
        sample_averages = [
            float(match.group(1))
            for line in lines
            if "qnn_executor_runner.cpp:621]" in line
            and (match := re.search(r"20 inference took [0-9.]+ ms, avg ([0-9.]+) ms", line))
        ]
        if len(sample_averages) != 20:
            raise AssertionError(f"expected 20 inference samples, found {len(sample_averages)}")
        report["inference_ms"] = {
            "samples": len(sample_averages),
            "mean": float(np.mean(sample_averages)),
            "median": float(np.median(sample_averages)),
            "p95": float(np.percentile(sample_averages, 95)),
            "minimum": float(np.min(sample_averages)),
            "maximum": float(np.max(sample_averages)),
        }
    if memory_path:
        report["peak_process_rss_kb"] = int(memory_path.read_text(encoding="ascii").strip())
    if report["max_abs_difference"] > 1e-2:
        raise AssertionError(f"QNN FP16 parity exceeded tolerance: {report['max_abs_difference']}")
    if report["repeat_max_abs_difference"] != 0.0:
        raise AssertionError("repeated QNN output is not deterministic")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("results/p4_4"))
    return parser.parse_args()


def main() -> None:
    result_dir = parse_args().result_dir
    report = analyze(
        result_dir / "eager_reference.raw",
        result_dir / "device_output_0.raw",
        result_dir / "device_output_1.raw",
        result_dir / "inference_speed.txt",
        result_dir / "device_runner.log",
        result_dir / "memory_peak_rss_kb.txt",
    )
    (result_dir / "device_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
