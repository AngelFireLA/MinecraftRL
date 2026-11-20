#!/usr/bin/env python3
"""Validate aim_tool_break with a deterministic aim/tool/attack oracle."""

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
from tasks import AimToolBreakTask


def closest_bin(values: np.ndarray, desired_degrees: float) -> int:
    return int(np.argmin(np.abs(values - desired_degrees)))


def oracle_action(task: AimToolBreakTask, raw: Mapping[str, Any]) -> np.ndarray:
    tool_slot = task.correct_slot(raw)
    if task.is_aimed(raw):
        return np.asarray(
            [
                closest_bin(task.YAW_DEGREES, 0.0),
                closest_bin(task.PITCH_DEGREES, 0.0),
                tool_slot,
                1,
            ],
            dtype=np.int64,
        )
    return np.asarray(
        [
            closest_bin(task.YAW_DEGREES, float(raw["task_goal.yaw_delta"])),
            closest_bin(task.PITCH_DEGREES, float(raw["task_goal.pitch_delta"])),
            tool_slot,
            0,
        ],
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--visualized", action="store_true")
    parser.add_argument("--tps", type=float, default=20.0)
    parser.add_argument("--episode-pause", type=float, default=0.5)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episode_pause < 0.0:
        parser.error("--episode-pause must be non-negative")

    task = AimToolBreakTask(max_episode_steps=args.max_episode_steps)
    env = MinecraftEnv(
        task,
        port=args.port,
        visualized=args.visualized,
        target_tps=args.tps,
    )
    successes = 0
    block_counts = {name: 0 for name in task.BLOCK_NAMES.values()}
    try:
        for episode in range(1, args.episodes + 1):
            _, info = env.reset(seed=args.seed + episode)
            raw = info["raw_observation"]
            block_name = str(info["target_block"])
            block_counts[block_name] += 1
            if args.visualized and args.episode_pause > 0.0:
                time.sleep(args.episode_pause)

            total_reward = 0.0
            for step in range(1, args.max_episode_steps + 1):
                action = oracle_action(task, raw)
                _, reward, terminated, truncated, info = env.step(action)
                raw = info["raw_observation"]
                total_reward += reward
                print(
                    f"episode={episode} block={block_name} step={step} "
                    f"{task.describe_action(action)} "
                    f"error={info['angular_error_degrees']:.2f} "
                    f"progress={info['break_progress']:.3f} reward={reward:+.3f}"
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
        print(
            f"aim_tool_break validation passed: {successes}/{args.episodes}; "
            f"blocks={block_counts}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
