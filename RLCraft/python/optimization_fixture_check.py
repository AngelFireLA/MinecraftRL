"""Live invariant checks on one already-running client; no task/reward changes."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from minecraft_env import MinecraftEnv
from task_factory import create_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    task = create_task("bed_defense_break_v2", 64)
    task.set_training_mode(True)
    task.set_curriculum_stage(6)
    env = MinecraftEnv(task, include_raw_info=True, strict_validation=True)
    try:
        task.action_space.seed(301)
        started = time.perf_counter()
        resets = 0
        for episode in range(5):
            obs, info = env.reset(seed=301 + episode)
            resets += 1
            for step in range(64):
                obs, reward, terminated, truncated, info = env.step(
                    task.action_space.sample()
                )
                if terminated or truncated:
                    break
        result["bed_v2_stage6"] = {
            "passed": True,
            "resets": resets,
            "seconds": time.perf_counter() - started,
        }
    finally:
        env.close()
    task = create_task("pvp_benchmark", 400)
    env = MinecraftEnv(task, include_raw_info=True, strict_validation=True)
    try:
        _, info = env.reset(seed=201)
        raw = info["raw_observation"]
        opponent_id = raw["task_entity.id"]
        for _ in range(12):
            _, _, done, _, info = env.step(np.array([4, 3, 0, 0, 0, 0, 0]))
            raw = info["raw_observation"]
            if done:
                break
        result["pvp_no_attack"] = {
            "passed": bool(raw["task_entity.alive"])
            and raw["task_entity.id"] == opponent_id,
            "health": raw["task_entity.health"],
        }
        _, info = env.reset(seed=201)
        raw = info["raw_observation"]
        trace = []
        for step in range(240):
            yaw = int(
                np.argmin(
                    np.abs(task.YAW_DEGREES - float(raw["task_entity.yaw_delta"]))
                )
            )
            pitch = int(
                np.argmin(
                    np.abs(task.PITCH_DEGREES - float(raw["task_entity.pitch_delta"]))
                )
            )
            move = 1 if float(raw["task_entity.distance"]) > 2.1 else 0
            _, _, done, _, info = env.step(np.array([yaw, pitch, move, 0, 0, 1, 0]))
            raw = info["raw_observation"]
            trace.append(
                {
                    k: raw[k]
                    for k in (
                        "task_entity.health",
                        "task_entity.alive",
                        "player.health",
                        "task_entity.distance",
                        "raycast.type",
                        "raycast.id",
                        "task_entity.id",
                    )
                }
            )
            if done:
                break
        damaged = any(
            row["task_entity.health"] < 20 and row["task_entity.alive"] for row in trace
        )
        killed = damaged and not raw["task_entity.alive"]
        _, reset = env.reset(seed=202)
        fresh = reset["raw_observation"]
        result["pvp_attack"] = {
            "damage_observed": damaged,
            "death_observed": killed,
            "fresh_alive": fresh["task_entity.alive"],
            "trace": trace,
        }
        result["pvp_speed_claim_allowed"] = bool(
            result["pvp_no_attack"]["passed"] and killed and fresh["task_entity.alive"]
        )
    finally:
        env.close()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in result.items() if k != "pvp_attack"}), flush=True
    )


if __name__ == "__main__":
    main()
