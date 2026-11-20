#!/usr/bin/env python3
"""Evaluate or visualize a trained model for any registered MinecraftRL task."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from minecraft_env import MinecraftEnv
from seed_utils import evaluation_seed
from sb3_algorithms import checkpoint_custom_objects, get_algorithm_class
from task_registry import get_task_preset, task_names


def parse_args(default_task: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=task_names(),
        default=default_task,
        required=default_task is None,
        help="registered task id",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--instance-id", default="rl-0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--tps", type=float, default=20.0)
    parser.add_argument("--episode-pause", type=float, default=1.0)
    parser.add_argument("--fast", action="store_true", help="disable rendering and pacing")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--verbose-steps", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base reset seed for the selected seed schedule",
    )
    parser.add_argument(
        "--seed-schedule",
        choices=("mixed", "sequential"),
        default="mixed",
        help="mixed disperses nearby episode indices for Java Random (recommended)",
    )
    parser.add_argument(
        "--guidance",
        choices=("stage", "on", "off"),
        default="stage",
        help="use stage guidance probability or explicitly force it on/off",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        help=(
            "evaluate one training reset stage by zero-based index; "
            "omit to evaluate the task's normal final/free configuration"
        ),
    )
    args = parser.parse_args()

    preset = get_task_preset(args.task)
    if args.model is None:
        args.model = Path("models") / preset.task_id / "final_model.zip"
    if args.max_episode_steps is None:
        args.max_episode_steps = preset.default_max_episode_steps
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if args.episode_pause < 0.0:
        parser.error("--episode-pause must be non-negative")
    return args


def configure_curriculum_stage(task: object, stage_index: int | None) -> str:
    """Select a training reset stage without changing the learned policy contract."""

    if stage_index is None:
        return "default"
    stages = getattr(task, "CURRICULUM_STAGES", None)
    set_training_mode = getattr(task, "set_training_mode", None)
    set_curriculum_stage = getattr(task, "set_curriculum_stage", None)
    if stages is None or not callable(set_training_mode) or not callable(
        set_curriculum_stage
    ):
        raise ValueError("this task does not expose curriculum reset stages")
    if not 0 <= stage_index < len(stages):
        raise ValueError(
            f"curriculum stage must be between 0 and {len(stages) - 1}"
        )
    set_training_mode(True)
    set_curriculum_stage(stage_index)
    selected_stage = stages[stage_index]
    return str(
        selected_stage[0]
        if isinstance(selected_stage, (tuple, list))
        else selected_stage
    )


def configure_guidance(task: object, mode: str) -> None:
    """Optionally decouple guidance from the selected evaluation geometry."""
    if mode == "stage":
        return
    setter = getattr(task, "set_guidance_override", None)
    if not callable(setter):
        raise ValueError("this task does not support a guidance override")
    setter(mode == "on")


def main(default_task: str | None = None) -> None:
    args = parse_args(default_task)
    preset = get_task_preset(args.task)
    algorithm_class = get_algorithm_class(preset.algorithm)
    model_path = args.model.resolve()
    if not model_path.exists() and model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    task = preset.create_task(args.max_episode_steps)
    try:
        evaluation_stage = configure_curriculum_stage(
            task, args.curriculum_stage
        )
        configure_guidance(task, args.guidance)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    env = MinecraftEnv(
        task,
        port=args.port,
        instance_id=args.instance_id,
        visualized=not args.fast,
        target_tps=args.tps,
    )
    model = algorithm_class.load(
        model_path,
        device=args.device,
        custom_objects=checkpoint_custom_objects(
            task.observation_space, task.action_space
        ),
    )
    successes = 0
    episode_lengths: list[int] = []
    returns: list[float] = []

    print(
        f"Loaded {model_path}; task={preset.task_id}, episodes={args.episodes}, "
        f"stage={evaluation_stage}, visualized={not args.fast}, "
        f"deterministic={not args.stochastic}, guidance={args.guidance}, "
        f"base_seed={args.seed}, seed_schedule={args.seed_schedule}, "
        f"endpoint={args.instance_id}@127.0.0.1:{args.port}"
    )
    try:
        for episode in range(1, args.episodes + 1):
            if not args.fast and args.episode_pause > 0.0:
                time.sleep(args.episode_pause)
            episode_seed = (
                evaluation_seed(args.seed, episode - 1)
                if args.seed_schedule == "mixed"
                else args.seed + episode - 1
            )
            observation, info = env.reset(seed=episode_seed)
            raw = info["raw_observation"]
            print(
                f"episode={episode} seed={episode_seed} "
                f"{task.describe_reset(raw)}"
            )
            if not args.fast and args.episode_pause > 0.0:
                time.sleep(args.episode_pause)

            episode_return = 0.0
            for step in range(1, args.max_episode_steps + 1):
                action, _ = model.predict(
                    observation, deterministic=not args.stochastic
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if args.verbose_steps:
                    diagnostics = ""
                    describe_observation = getattr(
                        task, "describe_observation", None
                    )
                    if callable(describe_observation):
                        diagnostics = " " + describe_observation(
                            info["raw_observation"]
                        )
                    print(
                        f"  step={step:02d} {task.describe_action(action)} "
                        f"reward={reward:+.3f}{diagnostics}"
                    )
                if terminated or truncated:
                    success = bool(info.get("success", terminated and not truncated))
                    successes += int(success)
                    episode_lengths.append(step)
                    returns.append(episode_return)
                    print(
                        f"episode={episode} success={success} steps={step} "
                        f"return={episode_return:.3f}"
                    )
                    break

        print(
            f"evaluation: success_rate={successes / args.episodes:.1%}, "
            f"mean_steps={sum(episode_lengths) / len(episode_lengths):.1f}, "
            f"mean_return={sum(returns) / len(returns):.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
