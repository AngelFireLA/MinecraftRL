#!/usr/bin/env python3
"""Supervise HeadlessMC clients and benchmark MinecraftRL training configurations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import csv
import itertools
import math
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ModuleNotFoundError as error:
    raise SystemExit("Install benchmark dependencies with: pip install -r requirements.txt") from error


DEFAULT_CONFIG: dict[str, Any] = {
    "task": "bed_defense_break_v2",
    "counts": [1, 2, 4, 6, 8, 10, 12, 14],
    "vec_envs": ["batched", "shared"],
    "devices": ["cpu"],
    "seeds": [1, 2, 3],
    "total_timesteps": 25600,
    "warmup_rollouts": 2,
    "profile_java_phases": False,
    "timeout_seconds": 900,
    "heap_mb": 1024,
    "gc_logging": False,
    "idle_wait_mode": "off",
    "profiling": True,
    "max_training_seconds": None,
    "active_processor_count": 4,
    "garbage_collector": "default",
    "display_sleep_milliseconds": 0,
    "stagger_seconds": 2,
    "max_memory_percent": 92.0,
    "trainer_cpus": [],
    "minecraft_cpus": [],
}


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        supplied = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f"unknown benchmark config keys: {sorted(unknown)}")
        config.update(supplied)
    return config


def ensure_instances(headless: Path, count: int) -> None:
    for index in range(count):
        instance = headless / "instances" / f"rl-{index}"
        if instance.is_dir():
            continue
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(headless / "New-Instance.ps1"),
                "-Name",
                f"rl-{index}",
            ],
            cwd=headless,
            check=True,
        )


def set_affinity(process: subprocess.Popen[Any], cpus: list[int]) -> None:
    if cpus:
        psutil.Process(process.pid).cpu_affinity(cpus)


def launch_client(
    headless: Path,
    index: int,
    config: dict[str, Any],
    run_dir: Path,
) -> tuple[subprocess.Popen[str], Any, Any]:
    stdout = (run_dir / f"rl-{index}.out.log").open("w", encoding="utf-8")
    stderr = (run_dir / f"rl-{index}.err.log").open("w", encoding="utf-8")
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(headless / "Start-Instance.ps1"),
        "-Name",
        f"rl-{index}",
        "-HeapMb",
        str(config["heap_mb"]),
        "-ActiveProcessorCount",
        str(config["active_processor_count"]),
        "-GarbageCollector",
        str(config["garbage_collector"]),
        "-IdleWaitMode",
        str(config.get("idle_wait_mode", "off")),
        "-DisplaySleepMilliseconds",
        str(config["display_sleep_milliseconds"]),
    ]
    if bool(config.get("gc_logging", False)):
        command.extend(
            ["-GcLogPath", str((run_dir / f"gc-rl-{index}.log").resolve())]
        )
    process = subprocess.Popen(
        command,
        cwd=headless,
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    minecraft_cpus = [int(cpu) for cpu in config.get("minecraft_cpus", [])]
    if minecraft_cpus:
        # PowerShell and the Java processes it creates inherit this affinity.
        set_affinity(process, minecraft_cpus)
    return process, stdout, stderr


def process_tree_rss(process: subprocess.Popen[Any]) -> int:
    try:
        root = psutil.Process(process.pid)
        members = [root, *root.children(recursive=True)]
        return sum(member.memory_info().rss for member in members if member.is_running())
    except (psutil.Error, ProcessLookupError):
        return 0


def stop_clients(clients: list[tuple[subprocess.Popen[str], Any, Any]]) -> None:
    for process, _, _ in clients:
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write("quit\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
    deadline = time.monotonic() + 30.0
    for process, stdout, stderr in clients:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        stdout.close()
        stderr.close()


def summarize_profile(path: Path, warmup_rollouts: int) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    steady = rows[warmup_rollouts:] or rows
    if not steady:
        raise RuntimeError(f"no profiling rows were written to {path}")

    def mean(name: str) -> float:
        return statistics.fmean(float(row[name]) for row in steady)

    transitions = sum(float(row.get("transitions", 0)) for row in steady)
    collect_ms = sum(float(row.get("rollout_ms", 0)) for row in steady)
    total_ms = collect_ms + sum(float(row.get("optimize_ms", 0)) for row in steady)
    samples = sorted(value for row in steady for value in row.get("vector_step_samples_ms", []))
    return {
        "rollouts": float(len(steady)),
        "collect_fps": transitions * 1000 / collect_ms if collect_ms else mean("collect_fps"),
        "end_to_end_fps": transitions * 1000 / total_ms if total_ms and transitions else mean("end_to_end_fps"),
        "vector_step_mean_ms": mean("vector_step_mean_ms"),
        "vector_step_p95_ms": samples[max(0, math.ceil(len(samples)*0.95)-1)] if samples else mean("vector_step_p95_ms"),
        "worker_critical_ms": mean("worker_critical_mean_ms"),
        "parent_policy_ipc_ms": mean("parent_policy_ipc_mean_ms"),
        "optimize_ms": mean("optimize_ms"),
        "reset_ms": mean("reset_mean_ms"),
    }


def summarize_gc_logs(run_dir: Path) -> dict[str, float]:
    """Summarize HotSpot Java GC pause times from optional -Xloggc logs."""
    pauses_ms: list[float] = []
    full_gc_count = 0
    for path in sorted(run_dir.glob("gc-rl-*.log")):
        content = path.read_text(encoding="utf-8", errors="replace")
        full_gc_count += content.count("Full GC")
        for line in content.splitlines():
            gc_event = line.partition("[Times:")[0]
            matches = re.findall(r",\s*(\d+(?:\.\d+)?)\s*(?:secs|s)\]", gc_event)
            if matches:
                pauses_ms.append(float(matches[-1]) * 1000.0)
    if not pauses_ms:
        return {
            "gc_pause_count": 0.0,
            "gc_pause_total_ms": 0.0,
            "gc_pause_p95_ms": 0.0,
            "gc_pause_max_ms": 0.0,
            "gc_full_collection_count": float(full_gc_count),
        }
    return {
        "gc_pause_count": float(len(pauses_ms)),
        "gc_pause_total_ms": float(sum(pauses_ms)),
        "gc_pause_p95_ms": float(
            sorted(pauses_ms)[max(0, math.ceil(0.95 * len(pauses_ms)) - 1)]
        ),
        "gc_pause_max_ms": float(max(pauses_ms)),
        "gc_full_collection_count": float(full_gc_count),
    }


def summarize_training_log(path: Path) -> dict[str, Any]:
    """Keep final rolling episode and mastery windows for learning guards."""
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    episode_line = next(
        (line for line in reversed(lines) if line.startswith("[episodes] ")),
        None,
    )
    mastery_line = next(
        (line for line in reversed(lines) if line.startswith("[mastery] ")),
        None,
    )

    def token(line: str | None, key: str) -> str | None:
        if line is None:
            return None
        found = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", line)
        return found.group(1).rstrip("%") if found is not None else None

    def number(line: str | None, key: str) -> float | None:
        value = token(line, key)
        if value is None:
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if parsed == parsed and abs(parsed) != float("inf") else None

    metrics: dict[str, Any] = {}
    for source_key, result_key in (
        ("total", "final_episode_count"),
        ("success_rate_100", "final_success_rate_100"),
        ("mean_reward_100", "final_mean_reward_100"),
        ("mean_length_100", "final_mean_episode_length_100"),
        ("mean_completion_100", "final_mean_completion_100"),
    ):
        value = number(episode_line, source_key)
        if value is not None:
            metrics[result_key] = value
    stage = token(episode_line, "stage")
    if stage is not None:
        metrics["final_curriculum_stage"] = stage
    eligible = number(mastery_line, "eligible")
    if eligible is not None:
        metrics["final_mastery_eligible"] = int(eligible)
    for source_key, result_key in (
        ("success", "final_mastery_success_rate"),
        ("required", "final_mastery_required_rate"),
    ):
        value = number(mastery_line, source_key)
        if value is not None:
            metrics[result_key] = value
    return metrics


def run_configuration(
    rlcraft: Path,
    headless: Path,
    output: Path,
    config: dict[str, Any],
    count: int,
    vec_env: str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    run_name = f"n{count}-{vec_env}-{device}-seed{seed}"
    if config.get("idle_wait_mode", "off") != "off":
        run_name += "-idle-" + config["idle_wait_mode"]
    run_dir = output / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_path = run_dir / "profile.jsonl"
    train_stdout = (run_dir / "train.out.log").open("w", encoding="utf-8")
    train_stderr = (run_dir / "train.err.log").open("w", encoding="utf-8")
    train_command = [
        sys.executable,
        str(rlcraft / "python" / "train.py"),
        "--task",
        str(config["task"]),
        "--num-envs",
        str(count),
        "--vec-env",
        vec_env,
        "--device",
        device,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(config["total_timesteps"]),
        "--checkpoint-freq",
        "1000000000",
        "--output-dir",
        str(run_dir / "model"),
        "--profile",
        "--profile-output",
        str(profile_path),
    ]
    if config.get("max_training_seconds") is not None:
        train_command.extend(["--max-training-seconds", str(config["max_training_seconds"])])
    if bool(config.get("profile_java_phases", False)):
        train_command.append("--profile-java-phases")
    if not config.get("profiling", True):
        train_command.remove("--profile")
    manifest = {"config": config, "count": count, "backend": vec_env, "device": device,
                "seed": seed, "command": train_command,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "stable-baselines3", "numpy", "gymnasium")},
                "sha256": {str(path.relative_to(rlcraft.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in [*sorted((rlcraft / "python").glob("*.py")),
                                        rlcraft / "build/libs/minecraftrl-1.0.0.jar",
                                        headless / "Start-Instance.ps1"]}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.monotonic()
    trainer = subprocess.Popen(
        train_command,
        cwd=rlcraft,
        stdout=train_stdout,
        stderr=train_stderr,
    )
    set_affinity(trainer, [int(cpu) for cpu in config.get("trainer_cpus", [])])
    clients: list[tuple[subprocess.Popen[str], Any, Any]] = []
    peak_rss = 0
    peak_memory_percent = 0.0
    status = "ok"
    owned = {}
    resource_samples = []
    try:
        time.sleep(1.0)
        for index in range(count):
            clients.append(launch_client(headless, index, config, run_dir))
            if index + 1 < count:
                time.sleep(float(config["stagger_seconds"]))
        deadline = started + float(config["timeout_seconds"])
        while trainer.poll() is None:
            peak_rss = max(
                peak_rss,
                process_tree_rss(trainer)
                + sum(process_tree_rss(client[0]) for client in clients),
            )
            snapshot = {"elapsed_s": time.monotonic() - started, "processes": []}
            for process_root, role in [(trainer, "trainer"), *[(client[0], "minecraft") for client in clients]]:
                try:
                    parent = psutil.Process(process_root.pid)
                    for member in [parent, *parent.children(recursive=True)]:
                        owned[member.pid] = member
                        memory = member.memory_info()
                        cpu = member.cpu_times()
                        snapshot["processes"].append({"pid": member.pid, "role": role if member.pid == parent.pid or role == "minecraft" else "worker",
                            "rss": memory.rss, "private": getattr(memory, "private", 0),
                            "cpu_seconds": cpu.user + cpu.system, "threads": member.num_threads()})
                except psutil.Error:
                    pass
            snapshot["system_cpu_percent"] = psutil.cpu_percent()
            snapshot["system_memory_percent"] = psutil.virtual_memory().percent
            resource_samples.append(snapshot)
            memory_percent = float(psutil.virtual_memory().percent)
            peak_memory_percent = max(peak_memory_percent, memory_percent)
            if memory_percent >= float(config["max_memory_percent"]):
                status = "memory_limit"
                trainer.terminate()
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                trainer.terminate()
                break
            time.sleep(1.0)
        try:
            exit_code = trainer.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            trainer.kill()
            exit_code = trainer.wait(timeout=5.0)
        if exit_code != 0 and status == "ok":
            status = f"train_exit_{exit_code}"
    finally:
        # Capture descendants before a launcher exits; never enumerate/kill unrelated jobs.
        for process_root in [trainer, *[client[0] for client in clients]]:
            try:
                root_process = psutil.Process(process_root.pid)
                for member in [root_process, *root_process.children(recursive=True)]: owned[member.pid] = member
            except psutil.Error: pass
        stop_clients(clients)
        survivors = [member for member in owned.values() if member.is_running()]
        for member in survivors:
            try: member.terminate()
            except psutil.Error: pass
        _, survivors = psutil.wait_procs(survivors, timeout=5)
        for member in survivors:
            try: member.kill()
            except psutil.Error: pass
        (run_dir / "resources.json").write_text(json.dumps(resource_samples), encoding="utf-8")
        train_stdout.close()
        train_stderr.close()

    result: dict[str, Any] = {
        "run": run_name,
        "task": config["task"],
        "count": count,
        "vec_env": vec_env,
        "device": device,
        "heap_mb": int(config["heap_mb"]),
        "idle_wait_mode": config.get("idle_wait_mode", "off"),
        "seed": seed,
        "status": status,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_gib": peak_rss / (1024**3),
        "peak_system_memory_percent": peak_memory_percent,
    }
    if status == "ok":
        if config.get("profiling", True) and profile_path.is_file():
            result.update(summarize_profile(profile_path, int(config["warmup_rollouts"])))
        result.update(summarize_training_log(run_dir / "train.out.log"))
    if status == "ok" and bool(config.get("gc_logging", False)):
        result.update(summarize_gc_logs(run_dir))
    throughput_path = run_dir / "model" / "throughput.json"
    if status == "ok" and throughput_path.is_file():
        result["warmed_wall_fps"] = json.loads(throughput_path.read_text())["fps"]
    return result


def write_results(
    output: Path,
    results: list[dict[str, Any]],
    max_memory_percent: float = 92.0,
) -> None:
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    fields = sorted({key for row in results for key in row})
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    successful = [row for row in results if row["status"] == "ok"]
    groups: dict[tuple[int, str, str, int], list[dict[str, Any]]] = {}
    heap_values = {int(row.get("heap_mb", 0)) for row in successful}
    multiple_heaps = len(heap_values) > 1
    for row in successful:
        key = (
            int(row["count"]),
            str(row["vec_env"]),
            str(row["device"]),
            int(row.get("heap_mb", 0)),
        )
        groups.setdefault(key, []).append(row)
    aggregate = []
    for (count, vec_env, device, heap_mb), rows in groups.items():
        name = f"n{count}-{vec_env}-{device}"
        if multiple_heaps:
            name += f"-heap{heap_mb}"
        aggregate.append(
            {
                "name": name,
                "count": count,
                "vec_env": vec_env,
                "device": device,
                "heap_mb": heap_mb,
                "task": str(rows[0]["task"]),
                "runs": len(rows),
                "end_to_end_fps": statistics.median(
                    float(row["end_to_end_fps"]) for row in rows
                ),
                "collect_fps": statistics.median(
                    float(row["collect_fps"]) for row in rows
                ),
                "vector_step_p95_ms": statistics.median(
                    float(row["vector_step_p95_ms"]) for row in rows
                ),
                "peak_rss_gib": max(float(row["peak_rss_gib"]) for row in rows),
                "peak_system_memory_percent": max(
                    float(row.get("peak_system_memory_percent", 0.0))
                    for row in rows
                ),
                "final_success_rate_100": (
                    statistics.median(
                        float(row["final_success_rate_100"])
                        for row in rows
                        if row.get("final_success_rate_100") is not None
                    )
                    if any(row.get("final_success_rate_100") is not None for row in rows)
                    else None
                ),
                "final_mastery_success_rate": (
                    statistics.median(
                        float(row["final_mastery_success_rate"])
                        for row in rows
                        if row.get("final_mastery_success_rate") is not None
                    )
                    if any(row.get("final_mastery_success_rate") is not None for row in rows)
                    else None
                ),
            }
        )
    aggregate_by_key = {
        (
            int(row["count"]),
            str(row["vec_env"]),
            str(row["device"]),
            int(row["heap_mb"]),
        ): row
        for row in aggregate
    }
    for row in aggregate:
        if row["task"] != "bed_defense_break_v2":
            row["learning_guard_pass"] = True
            continue
        if row["vec_env"] == "shared":
            row["paired_success_delta_pp"] = 0.0
            row["paired_mastery_delta_pp"] = 0.0
            row["learning_guard_pass"] = True
            continue
        baseline = aggregate_by_key.get(
            (
                int(row["count"]),
                "shared",
                str(row["device"]),
                int(row["heap_mb"]),
            )
        )
        candidate_runs = groups[
            (
                int(row["count"]),
                str(row["vec_env"]),
                str(row["device"]),
                int(row["heap_mb"]),
            )
        ]
        baseline_runs = (
            groups.get(
                (
                    int(row["count"]),
                    "shared",
                    str(row["device"]),
                    int(row["heap_mb"]),
                ),
                [],
            )
            if baseline is not None
            else []
        )
        baseline_by_seed = {int(item["seed"]): item for item in baseline_runs}
        paired = [
            (item, baseline_by_seed[int(item["seed"])])
            for item in candidate_runs
            if int(item["seed"]) in baseline_by_seed
        ]
        success_deltas = [
            float(candidate["final_success_rate_100"])
            - float(reference["final_success_rate_100"])
            for candidate, reference in paired
            if candidate.get("final_success_rate_100") is not None
            and reference.get("final_success_rate_100") is not None
        ]
        mastery_deltas = [
            float(candidate["final_mastery_success_rate"])
            - float(reference["final_mastery_success_rate"])
            for candidate, reference in paired
            if candidate.get("final_mastery_success_rate") is not None
            and reference.get("final_mastery_success_rate") is not None
        ]
        if len(success_deltas) >= 3 and len(mastery_deltas) >= 3:
            success_delta = statistics.median(success_deltas)
            mastery_delta = statistics.median(mastery_deltas)
            row["paired_success_delta_pp"] = success_delta
            row["paired_mastery_delta_pp"] = mastery_delta
            row["learning_guard_pass"] = (
                success_delta >= -5.0 and mastery_delta >= -5.0
            )
        else:
            row["paired_success_delta_pp"] = None
            row["paired_mastery_delta_pp"] = None
            row["learning_guard_pass"] = None
    ranked = sorted(aggregate, key=lambda row: row["end_to_end_fps"], reverse=True)
    lines = ["MinecraftRL benchmark ranking", ""]
    for row in ranked:
        learning = ""
        if row["task"] == "bed_defense_break_v2":
            learning = (
                f", success100={row['final_success_rate_100']!s}%"
                f", mastery100={row['final_mastery_success_rate']!s}%"
            )
        lines.append(
            f"{row['name']} ({row['runs']} run(s)): "
            f"median_e2e={row['end_to_end_fps']:.1f} fps, "
            f"collect={row['collect_fps']:.1f} fps, "
            f"p95={row['vector_step_p95_ms']:.2f} ms, "
            f"rss={row['peak_rss_gib']:.2f} GiB{learning}"
        )
    for row in aggregate:
        if row["task"] == "bed_defense_break_v2" and row["vec_env"] != "shared":
            if row["learning_guard_pass"] is None:
                state = "insufficient paired seeds"
            else:
                state = "pass" if row["learning_guard_pass"] else "FAIL"
            lines.append(
                f"Learning guard {row['name']}: {state}; "
                f"paired median success delta={row['paired_success_delta_pp']} pp, "
                f"mastery delta={row['paired_mastery_delta_pp']} pp "
                "(require at least 3 paired seeds and no median drop below -5 pp)."
            )
    stable_frontier = []
    by_backend: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in aggregate:
        by_backend.setdefault(
            (str(row["vec_env"]), str(row["device"]), int(row["heap_mb"])), []
        ).append(row)
    for rows in by_backend.values():
        selected = None
        previous_candidate = None
        for candidate in sorted(rows, key=lambda row: int(row["count"])):
            if candidate["learning_guard_pass"] is not True:
                previous_candidate = candidate
                continue
            if candidate["peak_system_memory_percent"] > max_memory_percent:
                previous_candidate = candidate
                continue
            if selected is None:
                selected = candidate
                stable_frontier.append(candidate)
                previous_candidate = candidate
                continue
            comparison = previous_candidate or selected
            enough_fps_gain = candidate["end_to_end_fps"] >= (
                1.03 * comparison["end_to_end_fps"]
            )
            p95_within_guard = candidate["vector_step_p95_ms"] <= (
                1.25 * selected["vector_step_p95_ms"]
            )
            if enough_fps_gain and p95_within_guard:
                selected = candidate
                stable_frontier.append(candidate)
            previous_candidate = candidate

    if ranked:
        lines.extend(["", f"Raw maximum observed throughput: {ranked[0]['name']}"])
    if stable_frontier:
        stable_ranked = sorted(
            stable_frontier,
            key=lambda row: row["end_to_end_fps"],
            reverse=True,
        )
        lines.append(
            "Recommended maximum-throughput preset (3% FPS gain, p95 and memory guards): "
            + stable_ranked[0]["name"]
        )
        memory_ranked = sorted(
            (
                row
                for row in stable_ranked
                if row["end_to_end_fps"]
                >= 0.8 * stable_ranked[0]["end_to_end_fps"]
            ),
            key=lambda row: (row["peak_rss_gib"], -row["end_to_end_fps"]),
        )
        if memory_ranked:
            lines.append(
                "Recommended lower-memory/interactive preset (>=80% of stable peak): "
                + memory_ranked[0]["name"]
            )
    elif ranked:
        lines.append(
            "No configuration passed the learning, latency, and memory guards."
        )
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/matrix"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rlcraft = Path(__file__).resolve().parents[1]
    headless = rlcraft.parent / "HeadlessMC"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    maximum_count = max(int(value) for value in config["counts"])
    ensure_instances(headless, maximum_count)
    results: list[dict[str, Any]] = []
    combinations = itertools.product(
        config["counts"], config["vec_envs"], config["devices"], config["seeds"]
    )
    for count, vec_env, device, seed in combinations:
        print(f"[matrix] n={count} vec={vec_env} device={device} seed={seed}", flush=True)
        result = run_configuration(
            rlcraft,
            headless,
            output,
            config,
            int(count),
            str(vec_env),
            str(device),
            int(seed),
        )
        results.append(result)
        write_results(output, results, float(config["max_memory_percent"]))
        if result["status"] == "memory_limit":
            print("[matrix] stopping after memory safety limit", flush=True)
            break


if __name__ == "__main__":
    main()
