"""Tests for PC benchmark statistics and report structure."""

import unittest

from benchmark.pipeline_benchmark import percentile, run_pc_baseline, summarize
from tools.run_pc_baseline import render_markdown


class PipelineBenchmarkTest(unittest.TestCase):
    def test_percentile_and_summary(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(percentile(samples, 0), 1.0)
        self.assertEqual(percentile(samples, 50), 3.0)
        self.assertEqual(percentile(samples, 100), 5.0)
        stats = summarize(samples)
        self.assertEqual(stats.mean_ms, 3.0)
        self.assertEqual(stats.median_ms, 3.0)
        self.assertEqual(stats.min_ms, 1.0)
        self.assertEqual(stats.max_ms, 5.0)

    def test_tiny_baseline_has_all_stages_and_valid_shapes(self) -> None:
        report = run_pc_baseline([(8, 6)], warmup=1, iterations=2, threads=1)
        result = report["results"][0]
        self.assertEqual(result["input_size"], [8, 6])
        self.assertEqual(result["output_size"], [16, 12])
        self.assertEqual(
            set(result["stages"]),
            {"rgb_to_tensor", "bilinear_2x", "model_inference", "tensor_to_rgb", "neural_end_to_end"},
        )
        for stats in result["stages"].values():
            self.assertGreaterEqual(stats["min_ms"], 0.0)
            self.assertGreaterEqual(stats["max_ms"], stats["min_ms"])
        markdown = render_markdown(report)
        self.assertIn("8x6", markdown)
        self.assertIn("neural_end_to_end", markdown)


if __name__ == "__main__":
    unittest.main()
