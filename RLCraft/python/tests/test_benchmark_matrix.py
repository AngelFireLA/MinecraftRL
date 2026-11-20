import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark_matrix import (
    DEFAULT_CONFIG,
    load_config,
    summarize_gc_logs,
    summarize_profile,
    summarize_training_log,
    write_results,
)


class BenchmarkMatrixTest(unittest.TestCase):
    def test_default_matrix_uses_valid_regression_task_and_cpu(self) -> None:
        config = load_config(None)
        self.assertEqual(config["task"], "bed_defense_break_v2")
        self.assertEqual(config["devices"], ["cpu"])
        self.assertFalse(config["gc_logging"])
        self.assertEqual(DEFAULT_CONFIG["vec_envs"], ["batched", "shared"])

    def test_config_rejects_unknown_keys(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"surprise": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_config(path)

    def test_profile_summary_drops_warmup(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            rows = []
            for fps in (10.0, 100.0, 200.0):
                rows.append(
                    {
                        "collect_fps": fps,
                        "end_to_end_fps": fps / 2,
                        "vector_step_mean_ms": 1.0,
                        "vector_step_p95_ms": 2.0,
                        "worker_critical_mean_ms": 0.5,
                        "parent_policy_ipc_mean_ms": 0.5,
                        "optimize_ms": 3.0,
                        "reset_mean_ms": 4.0,
                    }
                )
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary = summarize_profile(path, 1)
            self.assertEqual(summary["collect_fps"], 150.0)
            self.assertEqual(summary["end_to_end_fps"], 75.0)

    def test_training_log_summary_reads_final_success_and_mastery(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.out.log"
            path.write_text(
                "[episodes] total=400 success_rate_100=25.0% "
                "mean_reward_100=6.7 mean_length_100=28.0 "
                "mean_completion_100=25.0% stage=exposed_fixed_aimed\n"
                "[mastery] stage=exposed_fixed_aimed eligible=400 "
                "window=100/100 success=25.0% required=90.0%\n",
                encoding="utf-8",
            )
            summary = summarize_training_log(path)
            self.assertEqual(summary["final_success_rate_100"], 25.0)
            self.assertEqual(summary["final_mastery_success_rate"], 25.0)
            self.assertEqual(summary["final_mastery_eligible"], 400)

    def test_gc_summary_extracts_pause_and_full_collection_durations(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gc-rl-0.log"
            path.write_text(
                "2026-09-19T10:00:00.000+0200: [GC (Allocation Failure) "
                "[PSYoungGen: 10K->2K(20K)] 10K->3K(40K), 0.0050000 secs] "
                "[Times: user=0.01 sys=0.00, real=0.01 secs]\n"
                "2026-09-19T10:00:01.000+0200: [Full GC "
                "[PSYoungGen: 2K->0K(20K)] 20K->3K(40K), 0.1200000 secs] "
                "[Times: user=0.12 sys=0.00, real=0.12 secs]\n",
                encoding="utf-8",
            )
            summary = summarize_gc_logs(Path(directory))
            self.assertEqual(summary["gc_pause_count"], 2.0)
            self.assertAlmostEqual(summary["gc_pause_total_ms"], 125.0)
            self.assertEqual(summary["gc_pause_p95_ms"], 120.0)
            self.assertEqual(summary["gc_full_collection_count"], 1.0)


    def test_results_rank_medians_across_seeds(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            common = {
                "status": "ok",
                "task": "pvp_benchmark",
                "vector_step_p95_ms": 2.0,
                "collect_fps": 200.0,
                "peak_rss_gib": 1.0,
            }
            results = [
                {
                    **common,
                    "run": "a1",
                    "count": 2,
                    "vec_env": "batched",
                    "device": "cpu",
                    "seed": 1,
                    "end_to_end_fps": 100.0,
                },
                {
                    **common,
                    "run": "a2",
                    "count": 2,
                    "vec_env": "batched",
                    "device": "cpu",
                    "seed": 2,
                    "end_to_end_fps": 120.0,
                },
            ]
            write_results(output, results)
            summary = (output / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("median_e2e=110.0", summary)
            self.assertIn("n2-batched-cpu", summary)

    def test_recommendation_rejects_fps_gain_that_breaks_latency_guard(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            common = {
                "status": "ok",
                "task": "pvp_benchmark",
                "vec_env": "batched",
                "device": "cpu",
                "peak_rss_gib": 1.0,
                "peak_system_memory_percent": 70.0,
                "collect_fps": 200.0,
            }
            results = [
                {
                    **common,
                    "run": "n10",
                    "count": 10,
                    "seed": 1,
                    "end_to_end_fps": 100.0,
                    "vector_step_p95_ms": 20.0,
                },
                {
                    **common,
                    "run": "n12",
                    "count": 12,
                    "seed": 1,
                    "end_to_end_fps": 110.0,
                    "vector_step_p95_ms": 25.1,
                },
                {
                    **common,
                    "run": "n14",
                    "count": 14,
                    "seed": 1,
                    "end_to_end_fps": 105.0,
                    "vector_step_p95_ms": 22.0,
                },
            ]
            write_results(output, results)
            summary = (output / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Raw maximum observed throughput: n12-batched-cpu", summary)
            self.assertIn(
                "Recommended maximum-throughput preset "
                "(3% FPS gain, p95 and memory guards): n10-batched-cpu",
                summary,
            )

    def test_results_keep_heap_variants_separate(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            common = {
                "status": "ok",
                "task": "bed_defense_break_v2",
                "count": 10,
                "vec_env": "shared",
                "device": "cpu",
                "seed": 1,
                "collect_fps": 200.0,
                "vector_step_p95_ms": 20.0,
                "peak_rss_gib": 4.0,
                "peak_system_memory_percent": 70.0,
            }
            results = [
                {
                    **common,
                    "heap_mb": 1024,
                    "run": "heap1024",
                    "end_to_end_fps": 100.0,
                },
                {
                    **common,
                    "heap_mb": 1536,
                    "run": "heap1536",
                    "end_to_end_fps": 101.0,
                },
            ]
            write_results(output, results)
            summary = (output / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("n10-shared-cpu-heap1024", summary)
            self.assertIn("n10-shared-cpu-heap1536", summary)

    def test_learning_guard_blocks_throughput_with_success_regression(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            common = {
                "status": "ok",
                "task": "bed_defense_break_v2",
                "count": 10,
                "device": "cpu",
                "vector_step_p95_ms": 20.0,
                "peak_rss_gib": 1.0,
                "peak_system_memory_percent": 70.0,
                "collect_fps": 200.0,
            }
            results = []
            for seed in (1, 2, 3):
                results.append(
                    {
                        **common,
                        "run": f"shared-{seed}",
                        "vec_env": "shared",
                        "seed": seed,
                        "end_to_end_fps": 100.0,
                        "final_success_rate_100": 30.0,
                        "final_mastery_success_rate": 30.0,
                    }
                )
                results.append(
                    {
                        **common,
                        "run": f"batched-{seed}",
                        "vec_env": "batched",
                        "seed": seed,
                        "end_to_end_fps": 120.0,
                        "final_success_rate_100": 15.0,
                        "final_mastery_success_rate": 15.0,
                    }
                )
            write_results(output, results)
            summary = (output / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Learning guard n10-batched-cpu: FAIL", summary)
            self.assertIn("Recommended maximum-throughput preset", summary)
            self.assertIn("n10-shared-cpu", summary.splitlines()[-2])


if __name__ == "__main__":
    unittest.main()
