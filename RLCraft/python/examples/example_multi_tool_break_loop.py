#!/usr/bin/env python3
"""Validate multi_tool_break with a deterministic nearest-target oracle."""

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
from tasks import MultiToolBreakTask


def closest_bin(values: np.ndarray, desired_degrees: float) -> int:
    return int(np.argmin(np.abs(values - desired_degrees)))


def choose_target(task: MultiToolBreakTask, raw: Mapping[str, Any]) -> int:
    aimed = task.aimed_index(raw)
    if aimed >= 0:
        return aimed
    return min(
        task.remaining_indices(raw),
        key=lambda index: abs(float(raw[task.key(index, "yaw_delta")]))
        + abs(float(raw[task.key(index, "pitch_delta")])),
    )


def oracle_action(task: MultiToolBreakTask, raw: Mapping[str, Any]) -> np.ndarray:
    target = choose_target(task, raw)
    tool_slot = task.correct_slot(raw, target)
    attack = task.is_aimed_at(raw, target)
    return np.asarray(
        [
            closest_bin(
                task.YAW_DEGREES,
                0.0 if attack else float(raw[task.key(target, "yaw_delta")]),
            ),
            closest_bin(
                task.PITCH_DEGREES,
                0.0 if attack else float(raw[task.key(target, "pitch_delta")]),
            ),
            tool_slot,
            int(attack),
        ],
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--visualized", action="store_true")
    parser.add_argument("--tps", type=float, default=20.0)
    parser.add_argument("--episode-pause", type=float, default=0.5)
    args = parser.parse_args()

    task = MultiToolBreakTask(max_episode_steps=args.max_episode_steps)
    env = MinecraftEnv(task, port=args.port, visualized=args.visualized, target_tps=args.tps)
    successes = 0
    try:
        for episode in range(1, args.episodes + 1):
            _, info = env.reset(seed=args.seed + episode)
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
                    f"episode={episode} step={step} remaining={info['remaining_blocks']} "
                    f"{task.describe_action(action)} reward={reward:+.3f}"
                )
                if terminated or truncated:
                    successes += int(bool(info["success"]))
                    print(
                        f"episode={episode} success={info['success']} steps={step} "
                        f"return={total_reward:.3f}"
                    )
                    break
        if successes != args.episodes:
            raise RuntimeError(f"oracle solved {successes}/{args.episodes} episodes")
        print(f"multi_tool_break validation passed: {successes}/{args.episodes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
