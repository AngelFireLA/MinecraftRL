"""Run explicit, alternating throughput comparisons; preserve unrelated processes."""

import argparse
import json
import time
from pathlib import Path
from benchmark_matrix import DEFAULT_CONFIG, ensure_instances, run_configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runs",
        required=True,
        help="JSON list: [backend, idle mode, count, seed, timesteps]",
    )
    parser.add_argument("--budget-seconds", type=float, default=7200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    headless = root.parent / "HeadlessMC"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = json.loads(
        args.runs if args.runs.lstrip().startswith("[") else Path(args.runs).read_text()
    )
    ensure_instances(headless, max(r[2] for r in runs))
    deadline = time.monotonic() + args.budget_seconds
    results = []
    for run in runs:
        backend, idle, count, seed, steps = run[:5]
        profiling = run[5] if len(run) > 5 else True
        remaining = deadline - time.monotonic()
        if remaining < 120:
            break
        config = {
            **DEFAULT_CONFIG,
            "idle_wait_mode": idle,
            "profile_java_phases": profiling,
            "profiling": profiling,
            "total_timesteps": steps,
            "timeout_seconds": min(900, remaining),
            "stagger_seconds": 1,
        }
        print(
            f"[campaign] {backend}/{idle} n={count} seed={seed} steps={steps}",
            flush=True,
        )
        result = run_configuration(
            root, headless, output, config, count, backend, "cpu", seed
        )
        results.append(result)
        (output / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(json.dumps(result), flush=True)
        if result["status"] != "ok":
            break


if __name__ == "__main__":
    main()
