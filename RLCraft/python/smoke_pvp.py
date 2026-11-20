#!/usr/bin/env python3
"""Exercise deterministic no-attack or held-attack PvP steps against one client."""

from __future__ import annotations

import argparse

import numpy as np

from minecraft_env import MinecraftEnv
from task_registry import get_task_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--instance-id", default="rl-0")
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--attack", action="store_true", help="aim toward and attack the opponent"
    )
    if (args := parser.parse_args()).steps <= 0:
        parser.error("--steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    task = get_task_preset("pvp_benchmark").create_task(args.steps + 1)
    env = MinecraftEnv(
        task,
        host=args.host,
        port=args.port,
        instance_id=args.instance_id,
        include_raw_info=True,
        info_mode="full",
    )
    try:
        _, reset_info = env.reset(seed=args.seed)
        raw = reset_info["raw_observation"]
        print(
            "reset: "
            f"id={raw['task_entity.id']} alive={raw['task_entity.alive']} "
            f"health={raw['task_entity.health']:.1f} "
            f"distance={raw['task_entity.distance']:.2f}",
            flush=True,
        )
        for step in range(1, args.steps + 1):
            yaw_action = 4
            pitch_action = 3
            if args.attack:
                yaw_action = int(
                    np.argmin(
                        np.abs(
                            task.YAW_DEGREES
                            - float(raw["task_entity.yaw_delta"])
                        )
                    )
                )
                pitch_action = int(
                    np.argmin(
                        np.abs(
                            task.PITCH_DEGREES
                            - float(raw["task_entity.pitch_delta"])
                        )
                    )
                )
            action = np.asarray(
                [
                    yaw_action,
                    pitch_action,
                    int(args.attack),
                    0,
                    0,
                    int(args.attack),
                    0,
                ],
                dtype=np.int64,
            )
            _, reward, terminated, truncated, info = env.step(action)
            raw = info["raw_observation"]
            print(
                f"step={step} reward={reward:.3f} "
                f"opponent_alive={raw['task_entity.alive']} "
                f"opponent_health={raw['task_entity.health']:.1f} "
                f"player_health={raw['player.health']:.1f} "
                f"distance={raw['task_entity.distance']:.2f} "
                f"aim_delta=({raw['task_entity.yaw_delta']:.1f},"
                f"{raw['task_entity.pitch_delta']:.1f}) "
                f"raycast={raw['raycast.type']}:{raw['raycast.id']} "
                f"terminated={terminated} truncated={truncated}",
                flush=True,
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
