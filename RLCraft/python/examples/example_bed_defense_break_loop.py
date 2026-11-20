#!/usr/bin/env python3
"""Validate bed_defense_break with its deterministic cheapest-sightline oracle."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from minecraft_env import MinecraftEnv
from tasks import BedDefenseBreakTask


def oracle_failure_diagnostic(task: BedDefenseBreakTask, raw: dict) -> str:
    eye = (
        float(raw["player.x"]),
        float(raw["player.y"]) + task.EYE_HEIGHT,
        float(raw["player.z"]),
    )
    paths = []
    for bed_position in task.bed_positions(raw):
        cells = []
        for index in task._sightline_indices(raw, bed_position):
            category = task.category_at_index(raw, index)
            if category != task.CATEGORY_AIR:
                cells.append(
                    f"{task.index_to_position(raw, index)}:{task.CATEGORY_NAMES[category]}"
                )
        bed_index = task.position_to_index(raw, bed_position)
        center = tuple(coordinate + 0.5 for coordinate in bed_position)
        paths.append(
            f"bed={bed_position} center_distance={math.dist(eye, center):.3f} "
            f"within_reach={task._within_reach(raw, bed_index)} path={cells}"
        )
    return (
        f"player=({raw['player.x']:.3f},{raw['player.y']:.3f},"
        f"{raw['player.z']:.3f}) eye={eye} on_ground={raw['player.on_ground']} "
        f"motion_y={raw['player.motion_y']:.3f} raycast="
        f"{raw['raycast.type']}@({raw['raycast.x']},{raw['raycast.y']},"
        f"{raw['raycast.z']}) {'; '.join(paths)}"
    )


def oracle_aim_diagnostic(task: BedDefenseBreakTask, raw: dict) -> str:
    target = task.oracle_target_index(raw)
    eye = (
        float(raw["player.x"]),
        float(raw["player.y"]) + task.EYE_HEIGHT,
        float(raw["player.z"]),
    )
    aim_point = task.target_aim_point(raw, target)
    yaw_delta, pitch_delta = task.target_angle_deltas(raw, target)
    return (
        f"player=({raw['player.x']:.3f},{raw['player.y']:.3f},"
        f"{raw['player.z']:.3f}) eye={eye} rotation=({raw['player.yaw']:.3f},"
        f"{raw['player.pitch']:.3f}) target={task.index_to_position(raw, target)} "
        f"aim_point={aim_point} aim_distance={math.dist(eye, aim_point):.3f} "
        f"angle_delta=({yaw_delta:.3f},{pitch_delta:.3f}) raycast="
        f"{raw['raycast.type']}@({raw['raycast.x']},{raw['raycast.y']},"
        f"{raw['raycast.z']}) {oracle_failure_diagnostic(task, raw)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--instance-id", default="rl-0")
    parser.add_argument("--episodes", type=int, default=9)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--visualized", action="store_true")
    parser.add_argument("--tps", type=float, default=20.0)
    parser.add_argument("--episode-pause", type=float, default=0.5)
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        choices=range(len(BedDefenseBreakTask.CURRICULUM_STAGES)),
        help="validate one training reset stage instead of the final free task",
    )
    args = parser.parse_args()

    task = BedDefenseBreakTask(max_episode_steps=args.max_episode_steps)
    if args.curriculum_stage is not None:
        task.set_training_mode(True)
        task.set_curriculum_stage(args.curriculum_stage)
        stage_name = task.CURRICULUM_STAGES[args.curriculum_stage][0]
        print(f"validating curriculum stage {args.curriculum_stage}: {stage_name}")
    env = MinecraftEnv(
        task,
        port=args.port,
        instance_id=args.instance_id,
        visualized=args.visualized,
        target_tps=args.tps,
    )
    successes = 0
    try:
        for episode in range(1, args.episodes + 1):
            _, info = env.reset(seed=args.seed + episode)
            raw = info["raw_observation"]
            print(
                f"episode={episode} seed={args.seed + episode} reset "
                f"{task.describe_reset(raw)}"
            )
            if args.visualized and args.episode_pause > 0.0:
                time.sleep(args.episode_pause)
            total_reward = 0.0
            route_stall_steps = 0
            aim_stall_steps = 0
            for step in range(1, args.max_episode_steps + 1):
                action = task.oracle_action(raw)
                _, reward, terminated, truncated, info = env.step(action)
                raw = info["raw_observation"]
                total_reward += reward
                route_finite = math.isfinite(info["oracle_route_cost"])
                if not route_finite and info["bed_halves_remaining"] > 0:
                    route_stall_steps += 1
                    if route_stall_steps == 10:
                        print(
                            f"episode={episode} step={step} route_stall "
                            f"{oracle_failure_diagnostic(task, raw)}"
                        )
                else:
                    route_stall_steps = 0
                if route_finite and not bool(action[3]):
                    aim_stall_steps += 1
                    if aim_stall_steps == 10:
                        print(
                            f"episode={episode} step={step} aim_stall "
                            f"{oracle_aim_diagnostic(task, raw)}"
                        )
                else:
                    aim_stall_steps = 0
                if step == 1 or step % 20 == 0 or terminated or truncated:
                    print(
                        f"episode={episode} step={step} "
                        f"beds={info['bed_halves_remaining']} "
                        f"route_cost={info['oracle_route_cost']:.2f} "
                        f"broken={info['broken_blocks']} "
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
        print(f"bed_defense_break validation passed: {successes}/{args.episodes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
