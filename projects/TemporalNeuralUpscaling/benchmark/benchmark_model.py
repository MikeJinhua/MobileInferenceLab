"""Inference-only CPU microbenchmark for the minimal spatial SR model."""

import argparse
import statistics
import time

import torch

from model import SpatialSR2x


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=positive_int, default=64)
    parser.add_argument("--width", type=positive_int, default=64)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--threads", type=positive_int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = SpatialSR2x().eval()
    image = torch.rand(1, 3, args.height, args.width)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        samples_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            output = model(image)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(samples_ms)
    print("TemporalNeuralUpscaling Phase 1 CPU microbenchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {args.threads}")
    print(f"Input:  {tuple(image.shape)} float32 NCHW")
    print(f"Output: {tuple(output.shape)}")
    print(f"Warmup: {args.warmup}; measured iterations: {args.iterations}")
    print(f"Latency ms: mean={mean_ms:.3f}, median={statistics.median(samples_ms):.3f}, min={min(samples_ms):.3f}")
    print(f"Throughput: {1000.0 / mean_ms:.2f} inferences/s")
    print("Scope: inference only; excludes image I/O, transfer, synchronization, and display")


if __name__ == "__main__":
    main()
