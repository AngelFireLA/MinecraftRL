#!/usr/bin/env python3
"""Run a few real reset/step exchanges against multiple Minecraft instances."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from batched_minecraft_vec_env import BatchedMinecraftVecEnv
from shared_memory_vec_env import SharedMemoryVecEnv
from task_registry import get_task_preset, task_names
from train import make_environment_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=task_names(), default="aim_break_tnt")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument(
        "--vec-env",
        choices=("batched", "shared", "subproc"),
        default="batched",
    )
    parser.add_argument("--first-instance-index", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="override the environment episode limit to exercise auto-reset paths",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="print every N vector steps; 0 prints only the final throughput",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("--num-envs and --steps must be positive")
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if args.report_every < 0:
        parser.error("--report-every must be non-negative")
    if args.first_instance_index < 0:
        parser.error("--first-instance-index must be non-negative")
    if args.base_port < 1 or (
        args.base_port + args.first_instance_index + args.num_envs - 1 > 65535
    ):
        parser.error("the requested port range must stay between 1 and 65535")
    return args


def main() -> None:
    args = parse_args()
    preset = get_task_preset(args.task)
    connections = [
        (f"rl-{index}", args.base_port + index)
        for index in range(
            args.first_instance_index,
            args.first_instance_index + args.num_envs,
        )
    ]
    factories = [
        make_environment_factory(
            task_id=preset.task_id,
            max_episode_steps=(
                args.max_episode_steps
                if args.max_episode_steps is not None
                else max(args.steps + 1, 10)
            ),
            host=args.host,
            port=port,
            instance_id=instance_id,
            visualized=False,
            target_tps=20.0,
        )
        for instance_id, port in connections
    ]
    if args.vec_env == "batched":
        env = BatchedMinecraftVecEnv(factories)
    elif args.num_envs == 1:
        env = DummyVecEnv(factories)
    elif args.vec_env == "shared":
        env = SharedMemoryVecEnv(factories, start_method="spawn")
    else:
        env = SubprocVecEnv(factories, start_method="spawn")
    env.seed(args.seed)
    print(
        "Servers ready; start HeadlessMC instances: "
        + ", ".join(f"{instance_id}@{args.host}:{port}" for instance_id, port in connections),
        flush=True,
    )
    try:
        observations = env.reset()
        observation_shapes = (
            {key: value.shape for key, value in observations.items()}
            if isinstance(observations, Mapping)
            else observations.shape
        )
        print(
            f"reset ok: observations={observation_shapes}, "
            f"instances={[info.get('instance_id') for info in env.reset_infos]}",
            flush=True,
        )
        benchmark_started = time.perf_counter()
        for step in range(args.steps):
            actions = np.asarray(
                [env.action_space.sample() for _ in range(args.num_envs)]
            )
            observations, rewards, dones, _ = env.step(actions)
            completed = step + 1
            if args.report_every and (
                completed % args.report_every == 0 or completed == args.steps
            ):
                print(
                    f"step {completed}/{args.steps}: rewards={rewards.tolist()} "
                    f"dones={dones.tolist()}",
                    flush=True,
                )
        elapsed = time.perf_counter() - benchmark_started
        transitions = args.steps * args.num_envs
        print(
            f"multi-instance smoke test passed: {transitions / elapsed:.1f} "
            f"transition/s ({args.steps / elapsed:.1f} vector-step/s, "
            f"{elapsed * 1000.0 / args.steps:.2f} ms/vector-step)",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
