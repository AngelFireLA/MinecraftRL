"""Bounded confirmation and scaling; all comparisons use the continuing background load."""

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from benchmark_matrix import DEFAULT_CONFIG, run_configuration

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def median(rows, key):
    return statistics.median(r[key] for r in rows)


def campaign(name, runs):
    config = BASE / (name + ".json")
    config.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "python/optimization_campaign.py"),
            "--output",
            str(BASE / name),
            "--runs",
            str(config),
            "--budget-seconds",
            "1800",
        ],
        check=True,
    )
    rows = read(BASE / name / "results.json")
    if len(rows) != len(runs) or any(r["status"] != "ok" for r in rows):
        raise RuntimeError(f"Incomplete campaign: {name}")
    return rows


def main():
    screen = BASE / "optimization_screen_v2/results.json"
    deadline = time.monotonic() + 300
    while True:
        rows = read(screen) if screen.exists() else []
        if len(rows) == 4:
            break
        if time.monotonic() > deadline or any(r["status"] != "ok" for r in rows):
            raise RuntimeError("Screen did not complete successfully")
        time.sleep(2)
    baseline = [
        r for r in rows if r["vec_env"] == "shared" and r["idle_wait_mode"] == "off"
    ]
    candidate = [
        r
        for r in rows
        if r["vec_env"] == "lean-shared" and r["idle_wait_mode"] == "event"
    ]
    if candidate[0]["end_to_end_fps"] < baseline[0]["end_to_end_fps"] * 1.1:
        raise RuntimeError("Combined candidate did not qualify")
    print(
        "[followups] Confirming combined candidate with two additional paired seeds",
        flush=True,
    )
    confirmed = campaign(
        "optimization_confirm",
        [
            ["lean-shared", "event", 10, 302, 25600],
            ["shared", "off", 10, 302, 25600],
            ["lean-shared", "event", 10, 303, 25600],
            ["shared", "off", 10, 303, 25600],
        ],
    )
    baseline += [r for r in confirmed if r["vec_env"] == "shared"]
    candidate += [r for r in confirmed if r["vec_env"] == "lean-shared"]
    deltas = [
        r["final_success_rate_100"]
        - next(b["final_success_rate_100"] for b in baseline if b["seed"] == r["seed"])
        for r in candidate
    ]
    confirmed_ok = (
        median(candidate, "end_to_end_fps") >= median(baseline, "end_to_end_fps") * 1.1
        and median(candidate, "vector_step_p95_ms")
        <= median(baseline, "vector_step_p95_ms") * 1.25
        and statistics.median(deltas) >= -5
        and max(r["peak_system_memory_percent"] for r in candidate) < 90
    )
    print(
        "[followups] Confirmation",
        {
            "passed": confirmed_ok,
            "baseline_fps": median(baseline, "end_to_end_fps"),
            "candidate_fps": median(candidate, "end_to_end_fps"),
            "success_deltas_pp": deltas,
        },
        flush=True,
    )
    if not confirmed_ok:
        raise RuntimeError("Candidate failed confirmation guard")
    scale = campaign(
        "optimization_scale",
        [
            ["lean-shared", "event", 8, 301, 25600],
            ["lean-shared", "event", 12, 301, 25600],
            ["lean-shared", "event", 14, 301, 25600],
        ],
    )
    eligible = [
        r
        for r in scale
        if r["peak_system_memory_percent"] < 90
        and r["vector_step_p95_ms"] <= median(candidate, "vector_step_p95_ms") * 1.25
        and r["end_to_end_fps"] >= median(candidate, "end_to_end_fps") * 1.03
    ]
    selected = candidate
    if eligible:
        best = max(eligible, key=lambda r: r["end_to_end_fps"])
        more = campaign(
            "optimization_scale_confirm",
            [
                ["lean-shared", "event", best["count"], 302, 25600],
                ["lean-shared", "event", best["count"], 303, 25600],
            ],
        )
        scaled = [best, *more]
        deltas = [
            r["final_success_rate_100"]
            - next(
                b["final_success_rate_100"] for b in baseline if b["seed"] == r["seed"]
            )
            for r in scaled
        ]
        if (
            median(scaled, "end_to_end_fps")
            >= median(candidate, "end_to_end_fps") * 1.03
            and median(scaled, "vector_step_p95_ms")
            <= median(candidate, "vector_step_p95_ms") * 1.25
            and statistics.median(deltas) >= -5
            and max(r["peak_system_memory_percent"] for r in scaled) < 90
        ):
            selected = scaled
    count = selected[0]["count"]
    print(
        f"[followups] Selected {count} environments; checking profiling overhead",
        flush=True,
    )
    overhead = campaign(
        "optimization_profile_off", [["lean-shared", "event", count, 301, 25600, False]]
    )
    subprocess.run(
        [sys.executable, "-u", str(ROOT / "python/supervise_fixture_check.py")],
        check=True,
    )
    print("[followups] Starting 15-minute warmed stability run", flush=True)
    output = BASE / "optimization_sustained"
    output.mkdir(exist_ok=True)
    config = {
        **DEFAULT_CONFIG,
        "idle_wait_mode": "event",
        "profile_java_phases": True,
        "total_timesteps": 10000000,
        "max_training_seconds": 900,
        "timeout_seconds": 1200,
        "stagger_seconds": 1,
    }
    sustained = run_configuration(
        ROOT,
        ROOT.parent / "HeadlessMC",
        output,
        config,
        count,
        "lean-shared",
        "cpu",
        304,
    )
    (output / "results.json").write_text(
        json.dumps([sustained], indent=2), encoding="utf-8"
    )
    summary = {
        "baseline": baseline,
        "candidate10": candidate,
        "selected": selected,
        "profile_off": overhead,
        "sustained": sustained,
    }
    (BASE / "optimization_final.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[followups] Finished", json.dumps(sustained), flush=True)
    if sustained["status"] != "ok":
        raise RuntimeError("Sustained validation incomplete; results saved, defaults not promoted")


if __name__ == "__main__":
    main()
