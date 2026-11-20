#!/usr/bin/env python3
"""Validate aim_break_tnt with a deterministic binned oracle, without a model."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parent.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from minecraft_env import MinecraftEnv
from tasks import AimBreakTntTask


def closest_bin(values: np.ndarray, desired_degrees: float) -> int:
    return int(np.argmin(np.abs(values - desired_degrees)))


def oracle_action(task: AimBreakTntTask, raw: Mapping[str, Any]) -> np.ndarray:
    if task.is_aimed(raw):
        return np.asarray([4, 4, 1], dtype=np.int64)
    return np.asarray(
        [
            closest_bin(task.YAW_DEGREES, float(raw["task_goal.yaw_delta"])),
            closest_bin(task.PITCH_DEGREES, float(raw["task_goal.pitch_delta"])),
            0,
        ],
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--visualized", action="store_true")
    parser.add_argument("--tps", type=float, default=20.0)
    parser.add_argument("--episode-pause", type=float, default=0.5)
    args = parser.parse_args()

    task = AimBreakTntTask(max_episode_steps=args.max_episode_steps)
    env = MinecraftEnv(
        task,
        port=args.port,
        visualized=args.visualized,
        target_tps=args.tps,
    )
    successes = 0
    try:
        for episode in range(1, args.episodes + 1):
            observation, info = env.reset(seed=args.seed + episode)
            del observation
            raw = info["raw_observation"]
            if args.visualized and args.episode_pause > 0.0:
                time.sleep(args.episode_pause)
            total_reward = 0.0
            for step in range(1, args.max_episode_steps + 1):
                action = oracle_action(task, raw)
                _, reward, terminated, truncated, info = env.step(action)
                raw = info["raw_observation"]
                total_reward += reward
                print(
                    f"episode={episode} step={step} {task.describe_action(action)} "
                    f"error={info['angular_error_degrees']:.2f} "
                    f"aimed={info['aimed']} reward={reward:+.3f}"
                )
                if terminated or truncated:
                    success = bool(info["success"])
                    successes += int(success)
                    print(
                        f"episode={episode} success={success} steps={step} "
                        f"return={total_reward:.3f}"
                    )
                    break
        if successes != args.episodes:
            raise RuntimeError(f"oracle solved {successes}/{args.episodes} episodes")
        print(f"aim_break_tnt validation passed: {successes}/{args.episodes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
