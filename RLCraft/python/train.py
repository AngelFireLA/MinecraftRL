#!/usr/bin/env python3
"""Train a registered MinecraftRL task with its Stable-Baselines3 preset."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


# PPO's MLP batches are tiny compared with image models. Letting PyTorch and the
# BLAS backends create a thread per physical/logical core causes severe
# oversubscription once several Minecraft processes run beside the trainer.
_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _environment_name in _CPU_THREAD_ENV_VARS:
    os.environ.setdefault(_environment_name, "1")

try:
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.utils import FloatSchedule
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing training dependency {error.name!r}. Run: "
        "python -m pip install -r python/requirements.txt"
    ) from error

from minecraft_env import MinecraftEnv
from batched_minecraft_vec_env import BatchedMinecraftVecEnv
from seed_utils import evaluation_seed
from sb3_algorithms import checkpoint_custom_objects, get_algorithm_class
from shared_memory_vec_env import SharedMemoryVecEnv
from lean_shared_memory_vec_env import LeanSharedMemoryVecEnv
from task_registry import get_task_preset, task_names


def configure_torch_threads(thread_count: int) -> tuple[int, int]:
    """Bound PyTorch/BLAS CPU pools before creating models or worker processes."""
    if thread_count <= 0:
        raise ValueError("thread_count must be positive")

    value = str(thread_count)
    for environment_name in _CPU_THREAD_ENV_VARS:
        os.environ[environment_name] = value

    import torch

    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(thread_count)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != thread_count:
            raise RuntimeError(
                "PyTorch inter-op threads were initialized before training configuration"
            ) from error
    return torch.get_num_threads(), torch.get_num_interop_threads()


def resolve_training_device(requested_device: str) -> str:
    """Choose the measured fast device for the project's structured MLP policies.

    The 10-instance PvP A/B on the current target machine measured CPU at
    roughly 413 end-to-end FPS and CUDA at 268 FPS. These policies use small
    state-vector batches, so `auto` stays on CPU; CUDA remains an explicit A/B
    option for future architectures and workloads.
    """
    if requested_device == "auto":
        return "cpu"
    return requested_device


def initialize_bed_v2_action_biases(model: Any) -> None:
    """Give relative controls a conservative exploration prior.

    SB3 initializes every MultiDiscrete logit equally. For V2 that makes a
    large camera movement and an invalid attack much more common than holding
    the already-correct stage-zero aim. Bias only fresh policies: checkpoints
    must preserve whatever distribution they learned.
    """
    import torch

    action_sizes = tuple(int(size) for size in model.action_space.nvec)
    expected = (9, 9, 3, 2)
    if action_sizes != expected:
        raise ValueError(
            f"bed-defense V2 action initialization expected {expected}, "
            f"received {action_sizes}"
        )
    bias = model.policy.action_net.bias
    if bias is None or bias.numel() != sum(expected):
        raise ValueError("bed-defense V2 policy has an incompatible action head")

    yaw_offset = 0
    pitch_offset = expected[0]
    attack_offset = sum(expected[:-1])
    with torch.no_grad():
        bias.zero_()
        # exp(1.5) gives the zero delta about 36% probability on each camera
        # axis instead of 11%, while retaining every relative movement.
        bias[yaw_offset + 4] = 1.5
        bias[pitch_offset + 4] = 1.5
        # Begin with attack=True around 73%. Stage zero then receives enough
        # real progress samples before invalid attacks can suppress this head.
        bias[attack_offset] = -0.5
        bias[attack_offset + 1] = 0.5


def neutralize_bed_v2_tool_head(model: Any) -> None:
    """Remove learned tool preference without touching aim or attack logits."""
    import torch

    action_sizes = tuple(int(size) for size in model.action_space.nvec)
    expected = (9, 9, 3, 2)
    if action_sizes != expected:
        raise ValueError(
            f"bed-defense V2 tool neutralization expected {expected}, "
            f"received {action_sizes}"
        )
    action_net = model.policy.action_net
    if action_net.bias is None or action_net.bias.numel() != sum(expected):
        raise ValueError("bed-defense V2 policy has an incompatible action head")

    tool_start = expected[0] + expected[1]
    tool_rows = slice(tool_start, tool_start + expected[2])
    with torch.no_grad():
        action_net.weight[tool_rows].zero_()
        action_net.bias[tool_rows].zero_()

        # Adam momentum from an exposed stage would immediately restore the
        # arbitrary preference after release, so clear only these output rows.
        optimizer = getattr(model.policy, "optimizer", None)
        if optimizer is not None:
            for parameter in (action_net.weight, action_net.bias):
                for value in optimizer.state.get(parameter, {}).values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value[tool_rows].zero_()


def _find_subproc_vec_env(environment: Any) -> Any | None:
    """Find the subprocess owner below wrappers such as VecMonitor."""
    current = environment
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (SubprocVecEnv, SharedMemoryVecEnv)):
            return current
        current = getattr(current, "venv", None)
    return None


def _force_close_subproc_vec_env(environment: Any) -> None:
    """Close pipes and terminate workers without waiting for dead-worker replies."""
    for remote in environment.remotes:
        try:
            remote.close()
        except (BrokenPipeError, EOFError, OSError):
            pass
    for process in environment.processes:
        if process.is_alive():
            process.terminate()
    for process in environment.processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
    environment.waiting = False
    environment.closed = True
    close_shared = getattr(environment, "_close_shared", None)
    if callable(close_shared):
        close_shared()


def close_training_environment(environment: Any, *, force: bool = False) -> None:
    """Close normally after success, but never block on a failed SB3 worker."""
    subprocess_environment = _find_subproc_vec_env(environment)
    worker_failed = subprocess_environment is not None and any(
        process.exitcode not in (None, 0)
        for process in subprocess_environment.processes
    )
    if subprocess_environment is not None and (force or worker_failed):
        _force_close_subproc_vec_env(subprocess_environment)
        return
    try:
        environment.close()
    except (BrokenPipeError, EOFError, ConnectionError, OSError, RuntimeError):
        if subprocess_environment is None:
            raise
        _force_close_subproc_vec_env(subprocess_environment)


def resolve_algorithm_kwargs(
    defaults: dict[str, Any],
    *,
    num_envs: int,
    n_steps_override: int | None = None,
    batch_size_override: int | None = None,
) -> dict[str, Any]:
    """Keep PPO's total rollout near the single-environment preset size."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    result = dict(defaults)
    if batch_size_override is not None:
        result["batch_size"] = batch_size_override
    if n_steps_override is not None:
        result["n_steps"] = n_steps_override
        return result
    if num_envs == 1 or "n_steps" not in result:
        return result

    original_steps = int(result["n_steps"])
    batch_size = int(result.get("batch_size", 1))
    # Choose a per-environment length whose vector rollout divides cleanly into
    # minibatches. This avoids collecting N times more stale on-policy data just
    # because N Minecraft instances are running.
    quantum = batch_size // math.gcd(batch_size, num_envs)
    target_steps = original_steps / num_envs
    result["n_steps"] = max(
        quantum,
        int(math.floor(target_steps / quantum + 0.5)) * quantum,
    )
    return result


def apply_bridging_resume_safety(model: Any) -> None:
    """Apply the current conservative PPO settings to an older checkpoint."""
    learning_rate = 1.0e-4
    model.learning_rate = learning_rate
    model.lr_schedule = FloatSchedule(learning_rate)
    model.n_epochs = 4
    model.clip_range = FloatSchedule(0.15)
    model.target_kl = 0.02
    for parameter_group in model.policy.optimizer.param_groups:
        parameter_group["lr"] = learning_rate


class MasteryCurriculum:
    """Advance only after sustained success in the active task stage."""

    CONFIGS = {
        "multi_tool_break": (
            (
                "one_guided",
                "two_guided",
                "three_guided",
                "multi_guided",
                "multi_free",
            ),
            (0.70, 0.65, 0.60, 0.55),
        ),
        "bed_defense_break": (
            (
                "exposed_near",
                "exposed_random",
                "single_mixed_aimed",
                "single_mixed",
                "tunnel_guided",
                "layer_one",
                "layer_two",
                "varied_pose",
                "obsidian_guided",
                "layered_hint_75",
                "layered_hint_50",
                "layered_hint_25",
                "layered_free",
            ),
            (0.90, 0.85, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.35),
        ),
        "bed_defense_break_v2": (
            (
                "exposed_fixed_aimed",
                "exposed_random",
                "one_layer_fixed_aimed",
                "one_layer_random",
                "two_layers_random",
                "obsidian_inner_random",
                "full_random",
            ),
            (0.90, 0.90, 0.90, 0.90, 0.90, 0.80),
        ),
        "bridging": (
            (
                "one_gap_edge_aimed",
                "one_gap_random",
                "five_gap_edge_aimed",
                "five_gap_random",
                "eight_gap_random",
                "eleven_gap_random",
                "fifteen_gap_random",
                "twenty_two_gap_random",
                "thirty_gap_random",
                "variable_16_64_random",
            ),
            (0.90, 0.90, 0.85, 0.85, 0.82, 0.80, 0.75, 0.77, 0.75, 0.90),
        ),
    }

    def __init__(
        self,
        stage_index: int = 0,
        min_episodes: int = 100,
        task_id: str = "multi_tool_break",
    ) -> None:
        if task_id not in self.CONFIGS:
            raise ValueError(f"task {task_id!r} has no mastery curriculum")
        self.stages, self.success_thresholds = self.CONFIGS[task_id]
        if not 0 <= stage_index < len(self.stages):
            raise ValueError("invalid initial curriculum stage")
        if min_episodes <= 0:
            raise ValueError("min_episodes must be positive")
        self.stage_index = stage_index
        self.min_episodes = min_episodes
        self.successes: deque[float] = deque(maxlen=min_episodes)

    @property
    def stage(self) -> str:
        return self.stages[self.stage_index]

    def ready(self) -> bool:
        if self.stage_index >= len(self.stages) - 1:
            return False
        if len(self.successes) < self.min_episodes:
            return False
        threshold = self.success_thresholds[self.stage_index]
        return sum(self.successes) / len(self.successes) >= threshold

    def advance(self) -> None:
        if not self.ready():
            raise RuntimeError("curriculum cannot advance before mastery")
        self.stage_index += 1
        self.successes.clear()

    def observe(
        self,
        stage: str,
        success: bool,
        *,
        eligible: bool = True,
        advance: bool = True,
    ) -> bool:
        if not eligible:
            return False
        if stage != self.stage:
            return False
        self.successes.append(float(success))
        if self.stage_index >= len(self.stages) - 1:
            return False
        if not self.ready() or not advance:
            return False
        self.advance()
        return True


class EpisodeConsoleCallback(BaseCallback):
    BED_V2_DIAGNOSTIC_KEYS = (
        "attack_fraction",
        "block_attack_fraction",
        "correct_tool_attack_fraction",
        "wrong_tool_attack_fraction",
        "air_attack_fraction",
        "bed_attack_fraction",
        "defense_attack_fraction",
        "other_attack_fraction",
        "bed_focus_fraction",
        "breakable_focus_fraction",
        "camera_noop_fraction",
        "mean_abs_yaw_delta_degrees",
        "mean_abs_pitch_delta_degrees",
        "tool_switch_fraction",
        "pickaxe_selection_fraction",
        "axe_selection_fraction",
        "shears_selection_fraction",
        "max_break_progress",
        "max_observed_break_progress",
        "progress_reward_total",
    )

    def __init__(self, log_every_episodes: int) -> None:
        super().__init__(verbose=0)
        self.log_every_episodes = log_every_episodes
        self.episodes = 0
        self.successes = 0
        self.recent_rewards: deque[float] = deque(maxlen=100)
        self.recent_lengths: deque[int] = deque(maxlen=100)
        self.recent_successes: deque[float] = deque(maxlen=100)
        self.recent_broken: deque[float] = deque(maxlen=100)
        self.recent_completion: deque[float] = deque(maxlen=100)
        self.recent_focus: deque[float] = deque(maxlen=100)
        self.recent_attack: deque[float] = deque(maxlen=100)
        self.recent_valid_attack: deque[float] = deque(maxlen=100)
        self.recent_max_progress: deque[float] = deque(maxlen=100)
        self.recent_angular_error: deque[float] = deque(maxlen=100)
        self.recent_material_outcomes: deque[tuple[str, float]] = deque(maxlen=100)
        self.recent_variant_outcomes: deque[tuple[str, float]] = deque(maxlen=100)
        self.recent_stage_outcomes: deque[tuple[str, float]] = deque(maxlen=100)
        self.recent_rehearsal_outcomes: deque[tuple[str, float]] = deque(maxlen=100)
        self.recent_excess_blocks: deque[float] = deque(maxlen=100)
        self.recent_correct_tool_attack: deque[float] = deque(maxlen=100)
        self.recent_wrong_tool_attack: deque[float] = deque(maxlen=100)
        self.recent_progress_reward: deque[float] = deque(maxlen=100)
        self.recent_wrong_tool_penalty: deque[float] = deque(maxlen=100)
        self.recent_excess_block_penalty: deque[float] = deque(maxlen=100)
        self.recent_bed_v2: deque[dict[str, float | str]] = deque(maxlen=100)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.episodes += 1
            success = float(bool(info.get("success", False)))
            self.successes += int(success)
            self.recent_successes.append(success)
            episode_stage = str(info.get("curriculum_stage", "n/a"))
            curriculum_eligible = bool(info.get("curriculum_eligible", True))
            if curriculum_eligible:
                self.recent_stage_outcomes.append((episode_stage, success))
            elif bool(info.get("curriculum_rehearsal", False)):
                self.recent_rehearsal_outcomes.append((episode_stage, success))
            for key, values in (
                ("excess_blocks_broken", self.recent_excess_blocks),
                ("correct_tool_attack_fraction", self.recent_correct_tool_attack),
                ("wrong_tool_attack_fraction", self.recent_wrong_tool_attack),
                ("progress_reward_total", self.recent_progress_reward),
                ("wrong_tool_penalty_total", self.recent_wrong_tool_penalty),
                ("excess_block_penalty_total", self.recent_excess_block_penalty),
            ):
                if key in info:
                    values.append(float(info[key]))
            if "bed_focus_fraction" in info:
                record: dict[str, float | str] = {
                    "stage": episode_stage,
                    "layer_materials": str(info.get("layer_materials", "")),
                    "initial_selected_tool": str(
                        info.get("initial_selected_tool", "unknown")
                    ),
                    "tool_control_mode": str(
                        info.get("tool_control_mode", "policy")
                    ),
                    "rehearsal_kind": str(
                        info.get("curriculum_rehearsal_kind", "none")
                    ),
                    "success": success,
                    "mastery_eligible": float(curriculum_eligible),
                }
                for key in self.BED_V2_DIAGNOSTIC_KEYS:
                    record[key] = float(info.get(key, 0.0))
                self.recent_bed_v2.append(record)
            material = str(info.get("initial_target_material", ""))
            if material and material != "unknown":
                self.recent_material_outcomes.append((material, success))
            variant = str(info.get("curriculum_variant", ""))
            # Empty variants must also enter the fixed-size window so tunnel
            # depth diagnostics expire after the curriculum advances.
            self.recent_variant_outcomes.append((variant, success))
            if "broken_blocks" in info:
                self.recent_broken.append(float(info["broken_blocks"]))
            if "completion_fraction" in info:
                self.recent_completion.append(float(info["completion_fraction"]))
            for key, values in (
                ("focus_fraction", self.recent_focus),
                ("attack_fraction", self.recent_attack),
                ("valid_attack_fraction", self.recent_valid_attack),
                ("max_break_progress", self.recent_max_progress),
                ("mean_angular_error_degrees", self.recent_angular_error),
            ):
                if key in info:
                    values.append(float(info[key]))
            episode = info.get("episode", {})
            if "r" in episode:
                self.recent_rewards.append(float(episode["r"]))
            if "l" in episode:
                self.recent_lengths.append(int(episode["l"]))
            if self.episodes % self.log_every_episodes == 0:
                mean_reward = (
                    sum(self.recent_rewards) / len(self.recent_rewards)
                    if self.recent_rewards
                    else float("nan")
                )
                mean_length = (
                    sum(self.recent_lengths) / len(self.recent_lengths)
                    if self.recent_lengths
                    else float("nan")
                )
                success_rate = sum(self.recent_successes) / len(self.recent_successes)
                mean_broken = (
                    sum(self.recent_broken) / len(self.recent_broken)
                    if self.recent_broken
                    else float("nan")
                )
                mean_completion = (
                    sum(self.recent_completion) / len(self.recent_completion)
                    if self.recent_completion
                    else float("nan")
                )
                stage = info.get("curriculum_stage", "n/a")
                task_diagnostics = ""
                if self.recent_focus:
                    task_diagnostics = (
                        f" mean_focus_100="
                        f"{sum(self.recent_focus) / len(self.recent_focus):.1%} "
                        f"mean_attack_100="
                        f"{sum(self.recent_attack) / len(self.recent_attack):.1%} "
                        f"mean_valid_attack_100="
                        f"{sum(self.recent_valid_attack) / len(self.recent_valid_attack):.1%} "
                        f"mean_max_progress_100="
                        f"{sum(self.recent_max_progress) / len(self.recent_max_progress):.1%} "
                        f"mean_angle_100="
                        f"{sum(self.recent_angular_error) / len(self.recent_angular_error):.1f}deg"
                    )
                if self.recent_material_outcomes:
                    material_rates = []
                    for material in sorted(
                        {name for name, _ in self.recent_material_outcomes}
                    ):
                        outcomes = [
                            outcome
                            for name, outcome in self.recent_material_outcomes
                            if name == material
                        ]
                        material_rates.append(
                            f"{material}:{sum(outcomes) / len(outcomes):.0%}"
                            f"(n={len(outcomes)})"
                        )
                    task_diagnostics += (
                        " material_success_100=" + ",".join(material_rates)
                    )
                if self.recent_variant_outcomes:
                    variant_rates = []
                    for variant in sorted(
                        {
                            name
                            for name, _ in self.recent_variant_outcomes
                            if name
                        }
                    ):
                        outcomes = [
                            outcome
                            for name, outcome in self.recent_variant_outcomes
                            if name == variant
                        ]
                        variant_rates.append(
                            f"{variant}:{sum(outcomes) / len(outcomes):.0%}"
                            f"(n={len(outcomes)})"
                        )
                    if variant_rates:
                        task_diagnostics += (
                            " variant_success_100=" + ",".join(variant_rates)
                        )
                if self.recent_excess_blocks:
                    task_diagnostics += (
                        f" mean_excess_blocks_100="
                        f"{sum(self.recent_excess_blocks) / len(self.recent_excess_blocks):.2f}"
                        f" correct_tool_attack_100="
                        f"{sum(self.recent_correct_tool_attack) / len(self.recent_correct_tool_attack):.1%}"
                        f" wrong_tool_attack_100="
                        f"{sum(self.recent_wrong_tool_attack) / len(self.recent_wrong_tool_attack):.1%}"
                        f" progress_reward_100="
                        f"{sum(self.recent_progress_reward) / len(self.recent_progress_reward):.2f}"
                        f" wrong_tool_cost_100="
                        f"{sum(self.recent_wrong_tool_penalty) / len(self.recent_wrong_tool_penalty):.2f}"
                        f" excess_block_cost_100="
                        f"{sum(self.recent_excess_block_penalty) / len(self.recent_excess_block_penalty):.2f}"
                    )
                stage_rates = []
                for stage_name in sorted(
                    {name for name, _ in self.recent_stage_outcomes}
                ):
                    outcomes = [
                        outcome
                        for name, outcome in self.recent_stage_outcomes
                        if name == stage_name
                    ]
                    stage_rates.append(
                        f"{stage_name}:{sum(outcomes) / len(outcomes):.0%}"
                        f"(n={len(outcomes)})"
                    )
                if stage_rates:
                    task_diagnostics += " stage_success_100=" + ",".join(stage_rates)
                if self.recent_rehearsal_outcomes:
                    rehearsal_rates = []
                    for rehearsal_stage in sorted(
                        {name for name, _ in self.recent_rehearsal_outcomes}
                    ):
                        outcomes = [
                            outcome
                            for name, outcome in self.recent_rehearsal_outcomes
                            if name == rehearsal_stage
                        ]
                        rehearsal_rates.append(
                            f"{rehearsal_stage}:{sum(outcomes) / len(outcomes):.0%}"
                            f"(n={len(outcomes)})"
                        )
                    task_diagnostics += (
                        " rehearsal_success_100=" + ",".join(rehearsal_rates)
                    )
                print(
                    f"[episodes] total={self.episodes} "
                    f"success_rate_100={success_rate:.1%} "
                    f"mean_reward_100={mean_reward:.3f} "
                    f"mean_length_100={mean_length:.1f} "
                    f"mean_broken_100={mean_broken:.2f} "
                    f"mean_completion_100={mean_completion:.1%} "
                    f"stage={stage}{task_diagnostics}",
                    flush=True,
                )
                stage_v2 = [
                    record
                    for record in self.recent_bed_v2
                    if record["stage"] == stage
                ]
                if stage_v2:
                    def mean_v2(key: str) -> float:
                        return sum(float(record[key]) for record in stage_v2) / len(
                            stage_v2
                        )

                    material_rates = []
                    for material in sorted(
                        {str(record["layer_materials"]) or "none" for record in stage_v2}
                    ):
                        outcomes = [
                            float(record["success"])
                            for record in stage_v2
                            if (str(record["layer_materials"]) or "none") == material
                        ]
                        material_rates.append(
                            f"{material}:{sum(outcomes) / len(outcomes):.0%}"
                            f"(n={len(outcomes)})"
                        )
                    start_tool_rates = []
                    for tool in ("diamond_pickaxe", "diamond_axe", "shears"):
                        count = sum(
                            str(record["initial_selected_tool"]) == tool
                            for record in stage_v2
                        )
                        start_tool_rates.append(f"{tool}:{count / len(stage_v2):.0%}")
                    rehearsal_rates = []
                    for kind in ("partial_defense", "no_obsidian"):
                        count = sum(
                            str(record["rehearsal_kind"]) == kind
                            for record in stage_v2
                        )
                        if count:
                            rehearsal_rates.append(
                                f"{kind}:{count / len(stage_v2):.0%}"
                            )
                    print(
                        f"[bed-v2] stage={stage} n={len(stage_v2)} "
                        f"tool_control={stage_v2[-1]['tool_control_mode']} "
                        f"rehearsal={1.0 - mean_v2('mastery_eligible'):.1%} "
                        f"rehearsal_kinds={','.join(rehearsal_rates) or 'none'} "
                        f"attack={mean_v2('attack_fraction'):.1%} "
                        f"block_hit={mean_v2('block_attack_fraction'):.1%} "
                        f"focus_bed={mean_v2('bed_focus_fraction'):.1%} "
                        f"focus_breakable={mean_v2('breakable_focus_fraction'):.1%} "
                        f"attack_bed={mean_v2('bed_attack_fraction'):.1%} "
                        f"attack_defense={mean_v2('defense_attack_fraction'):.1%} "
                        f"attack_air={mean_v2('air_attack_fraction'):.1%} "
                        f"attack_other={mean_v2('other_attack_fraction'):.1%} "
                        f"correct_tool={mean_v2('correct_tool_attack_fraction'):.1%} "
                        f"wrong_tool={mean_v2('wrong_tool_attack_fraction'):.1%} "
                        f"max_progress={mean_v2('max_break_progress'):.1%} "
                        f"max_seen_progress={mean_v2('max_observed_break_progress'):.1%} "
                        f"camera_noop={mean_v2('camera_noop_fraction'):.1%} "
                        f"mean_camera=({mean_v2('mean_abs_yaw_delta_degrees'):.2f}deg,"
                        f"{mean_v2('mean_abs_pitch_delta_degrees'):.2f}deg) "
                        f"tool_switch={mean_v2('tool_switch_fraction'):.1%} "
                        f"selected=(pickaxe:{mean_v2('pickaxe_selection_fraction'):.0%},"
                        f"axe:{mean_v2('axe_selection_fraction'):.0%},"
                        f"shears:{mean_v2('shears_selection_fraction'):.0%}) "
                        f"starts={','.join(start_tool_rates)} "
                        f"material_success={','.join(material_rates)}",
                        flush=True,
                    )
                    for material in sorted(
                        {str(record["layer_materials"]) or "none" for record in stage_v2}
                    ):
                        material_records = [
                            record
                            for record in stage_v2
                            if (str(record["layer_materials"]) or "none") == material
                        ]

                        def material_mean(key: str) -> float:
                            return sum(
                                float(record[key]) for record in material_records
                            ) / len(material_records)

                        print(
                            f"[bed-v2-material] stage={stage} material={material} "
                            f"n={len(material_records)} "
                            f"success={material_mean('success'):.1%} "
                            f"defense_attack={material_mean('defense_attack_fraction'):.1%} "
                            f"correct_tool={material_mean('correct_tool_attack_fraction'):.1%} "
                            f"wrong_tool={material_mean('wrong_tool_attack_fraction'):.1%} "
                            f"max_progress={material_mean('max_break_progress'):.1%} "
                            f"progress_reward={material_mean('progress_reward_total'):.2f} "
                            f"selected=(pickaxe:"
                            f"{material_mean('pickaxe_selection_fraction'):.0%},"
                            f"axe:{material_mean('axe_selection_fraction'):.0%},"
                            f"shears:{material_mean('shears_selection_fraction'):.0%})",
                            flush=True,
                        )
        return True


def _profile_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _profile_percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


class SteadyThroughputCallback(BaseCallback):
    """Low-overhead warmed wall throughput, also usable without detailed profiling."""
    def __init__(self, path: Path, warmup_rollouts: int = 2, max_seconds: float | None = None):
        super().__init__()
        self.path = path
        self.warmup_rollouts = warmup_rollouts
        self.max_seconds = max_seconds
        self.rollouts = 0
        self.started = None
        self.start_steps = 0

    def _on_step(self):
        return self.max_seconds is None or self.started is None or time.perf_counter() - self.started < self.max_seconds

    def _on_rollout_start(self):
        if self.rollouts == self.warmup_rollouts:
            self.started = time.perf_counter()
            self.start_steps = self.num_timesteps
        self.rollouts += 1

    def _on_training_end(self):
        if self.started is not None:
            seconds = time.perf_counter() - self.started
            transitions = self.num_timesteps - self.start_steps
            data = {"transitions": transitions, "seconds": seconds,
                    "fps": transitions / max(seconds, 1e-9), "warmup_rollouts": self.warmup_rollouts}
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[throughput] warmed training FPS={data['fps']:.1f}", flush=True)


class TrainingProfilerCallback(BaseCallback):
    """Attribute real SB3 wall time to rollout, workers, resets, and PPO."""

    PROFILE_FIELDS = (
        "env_step_ms",
        "action_encode_ms",
        "pace_ms",
        "send_ms",
        "bridge_wait_ms",
        "protocol_ms",
        "socket_read_ms",
        "json_decode_ms",
        "task_ms",
        "vector_dispatch_ms",
        "vector_wait_ms",
        "action_client_ms",
        "server_tick_ms",
        "post_client_ms",
        "observation_serialize_ms",
        "java_total_ms",
    )

    def __init__(
        self, every_rollouts: int = 1, output_path: Path | None = None
    ) -> None:
        super().__init__(verbose=0)
        if every_rollouts <= 0:
            raise ValueError("profile interval must be positive")
        self.every_rollouts = every_rollouts
        self.output_path = output_path
        self.rollout_index = 0
        self._rollout_started = 0.0
        self._last_step_at = 0.0
        self._pending: dict[str, Any] | None = None
        self._seen_resets: set[tuple[str, int]] = set()
        self._clear_samples()

    def _clear_samples(self) -> None:
        self._vector_step_ms: list[float] = []
        self._worker_critical_ms: list[float] = []
        self._worker_spread_ms: list[float] = []
        self._field_samples: dict[str, list[float]] = {
            field: [] for field in self.PROFILE_FIELDS
        }
        self._reset_ms: list[float] = []
        self._reset_bridge_ms: list[float] = []
        self._instance_worker_ms: dict[str, list[float]] = {}

    def _base_vector_environment(self) -> Any:
        current = self.training_env
        visited: set[int] = set()
        while hasattr(current, "venv") and id(current) not in visited:
            visited.add(id(current))
            current = current.venv
        return current

    def _reset_infos(self) -> list[dict[str, Any]]:
        reset_infos = getattr(
            self._base_vector_environment(), "reset_infos", []
        )
        return list(reset_infos)

    @staticmethod
    def _reset_key(info: dict[str, Any]) -> tuple[str, int] | None:
        if "instance_id" not in info or "episode" not in info:
            return None
        return str(info["instance_id"]), int(info["episode"])

    def _on_training_start(self) -> None:
        from phase_profiling import PhaseTimers
        self._phase_timers = PhaseTimers(self.model, self._base_vector_environment())
        callback = self.locals.get("callback")
        if callback is not None:
            self._phase_timers.wrap(callback, "on_step", "callbacks_ms")
        for info in self._reset_infos():
            key = self._reset_key(info)
            if key is not None:
                self._seen_resets.add(key)

    def _on_rollout_start(self) -> None:
        now = time.perf_counter()
        if self._pending is not None:
            self._report_pending((now - self._pending["ended_at"]) * 1000.0)
            self._pending = None
        self._clear_samples()
        self._rollout_started = now
        self._last_step_at = now

    def _on_step(self) -> bool:
        now = time.perf_counter()
        self._vector_step_ms.append((now - self._last_step_at) * 1000.0)
        self._last_step_at = now

        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        reset_infos = self._reset_infos()
        effective_worker_times: list[float] = []
        for index, info in enumerate(infos):
            profile = info.get("training_profile", {})
            if not isinstance(profile, dict) or "env_step_ms" not in profile:
                continue
            for field in self.PROFILE_FIELDS:
                if field in profile:
                    self._field_samples[field].append(float(profile[field]))

            instance_id = str(profile.get("instance_id", f"env-{index}"))
            worker_ms = float(profile["env_step_ms"])
            if index < len(dones) and bool(dones[index]) and index < len(reset_infos):
                reset_info = reset_infos[index]
                key = self._reset_key(reset_info)
                reset_profile = reset_info.get("training_profile", {})
                if (
                    key is not None
                    and key not in self._seen_resets
                    and isinstance(reset_profile, dict)
                    and "reset_ms" in reset_profile
                ):
                    self._seen_resets.add(key)
                    reset_ms = float(reset_profile["reset_ms"])
                    worker_ms += reset_ms
                    self._reset_ms.append(reset_ms)
                    if "reset_bridge_ms" in reset_profile:
                        self._reset_bridge_ms.append(
                            float(reset_profile["reset_bridge_ms"])
                        )
            effective_worker_times.append(worker_ms)
            self._instance_worker_ms.setdefault(instance_id, []).append(worker_ms)

        if effective_worker_times:
            self._worker_critical_ms.append(max(effective_worker_times))
            self._worker_spread_ms.append(
                max(effective_worker_times) - min(effective_worker_times)
            )
        return True

    def _on_rollout_end(self) -> None:
        ended_at = time.perf_counter()
        self.rollout_index += 1
        self._pending = {
            "ended_at": ended_at,
            "rollout_ms": (ended_at - self._rollout_started) * 1000.0,
            "vector_step_ms": list(self._vector_step_ms),
            "worker_critical_ms": list(self._worker_critical_ms),
            "worker_spread_ms": list(self._worker_spread_ms),
            "field_samples": {
                key: list(values) for key, values in self._field_samples.items()
            },
            "reset_ms": list(self._reset_ms),
            "reset_bridge_ms": list(self._reset_bridge_ms),
            "instance_worker_ms": {
                key: list(values)
                for key, values in self._instance_worker_ms.items()
            },
        }

    def _on_training_end(self) -> None:
        if self._pending is not None:
            now = time.perf_counter()
            self._report_pending((now - self._pending["ended_at"]) * 1000.0)
            self._pending = None
        if hasattr(self, "_phase_timers"):
            self._phase_timers.close()

    def _report_pending(self, optimize_ms: float) -> None:
        phase_totals = self._phase_timers.take() if hasattr(self, "_phase_timers") else {}
        if self.rollout_index % self.every_rollouts != 0:
            return
        assert self._pending is not None
        rollout_ms = float(self._pending["rollout_ms"])
        vector_steps = len(self._pending["vector_step_ms"])
        num_envs = int(getattr(self.training_env, "num_envs", 1))
        transitions = vector_steps * num_envs
        collect_fps = transitions / max(rollout_ms / 1000.0, 1e-9)
        end_to_end_fps = transitions / max(
            (rollout_ms + optimize_ms) / 1000.0, 1e-9
        )
        vector_mean = _profile_mean(self._pending["vector_step_ms"])
        critical_mean = _profile_mean(self._pending["worker_critical_ms"])
        parent_overhead = max(vector_mean - critical_mean, 0.0)
        fields = self._pending["field_samples"]
        reset_ms = self._pending["reset_ms"]

        print(
            f"[profile] rollout={self.rollout_index} transitions={transitions} "
            f"collect={rollout_ms / 1000.0:.3f}s "
            f"optimize+between={optimize_ms / 1000.0:.3f}s "
            f"collect_fps={collect_fps:.1f} end_to_end_fps={end_to_end_fps:.1f}",
            flush=True,
        )
        print(
            f"[profile] vector_step mean={vector_mean:.3f}ms "
            f"p95={_profile_percentile(self._pending['vector_step_ms'], 0.95):.3f}ms "
            f"worker_critical={critical_mean:.3f}ms "
            f"parent_policy_ipc~={parent_overhead:.3f}ms "
            f"worker_spread={_profile_mean(self._pending['worker_spread_ms']):.3f}ms",
            flush=True,
        )
        print(
            f"[profile] per_env env={_profile_mean(fields['env_step_ms']):.3f}ms "
            f"bridge={_profile_mean(fields['bridge_wait_ms']):.3f}ms "
            f"task={_profile_mean(fields['task_ms']):.3f}ms "
            f"encode={_profile_mean(fields['action_encode_ms']):.3f}ms "
            f"send={_profile_mean(fields['send_ms']):.3f}ms "
            f"protocol={_profile_mean(fields['protocol_ms']):.3f}ms "
            f"resets={len(reset_ms)} mean_reset={_profile_mean(reset_ms):.3f}ms "
            f"p95_reset={_profile_percentile(reset_ms, 0.95):.3f}ms "
            f"reset_bridge={_profile_mean(self._pending['reset_bridge_ms']):.3f}ms",
            flush=True,
        )
        java_fields = (
            "action_client_ms",
            "server_tick_ms",
            "post_client_ms",
            "observation_serialize_ms",
            "java_total_ms",
        )
        if any(fields[field] for field in java_fields):
            print(
                "[profile] java_phase "
                + " ".join(
                    f"{field}={_profile_mean(fields[field]):.3f}ms"
                    for field in java_fields
                    if fields[field]
                ),
                flush=True,
            )
        instance_worker_ms = self._pending["instance_worker_ms"]
        slowest = sorted(
            (
                (_profile_mean(values), instance_id)
                for instance_id, values in instance_worker_ms.items()
            ),
            reverse=True,
        )[:3]
        if slowest:
            print(
                "[profile] slowest_workers="
                + ",".join(
                    f"{instance_id}:{average:.3f}ms"
                    for average, instance_id in slowest
                ),
                flush=True,
            )
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "rollout": self.rollout_index,
                "phase_totals_ms": phase_totals,
                "phase_resources": dict(self._phase_timers.resources) if hasattr(self, "_phase_timers") else {},
                "vector_step_samples_ms": self._pending["vector_step_ms"],
                "transitions": transitions,
                "rollout_ms": rollout_ms,
                "optimize_ms": optimize_ms,
                "collect_fps": collect_fps,
                "end_to_end_fps": end_to_end_fps,
                "vector_step_mean_ms": vector_mean,
                "vector_step_p95_ms": _profile_percentile(
                    self._pending["vector_step_ms"], 0.95
                ),
                "worker_critical_mean_ms": critical_mean,
                "parent_policy_ipc_mean_ms": parent_overhead,
                "worker_spread_mean_ms": _profile_mean(
                    self._pending["worker_spread_ms"]
                ),
                "per_env": {
                    field: _profile_mean(values)
                    for field, values in fields.items()
                },
                "reset_count": len(reset_ms),
                "reset_mean_ms": _profile_mean(reset_ms),
                "reset_p95_ms": _profile_percentile(reset_ms, 0.95),
                "reset_bridge_mean_ms": _profile_mean(
                    self._pending["reset_bridge_ms"]
                ),
                "slowest_workers": [
                    {"instance_id": instance_id, "mean_ms": average}
                    for average, instance_id in slowest
                ],
            }
            with self.output_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, separators=(",", ":")) + "\n")


class TrainingProgressCallback(BaseCallback):
    """Keep progress aligned, preserve mastered policies, and advance stages."""

    BRIDGING_MASTERY_EPISODES = 500
    BRIDGING_VALIDATION_EPISODES = 100
    BRIDGING_FIXED_VALIDATION_EPISODES = 20
    BRIDGING_VALIDATION_RETRY_EPISODES = 100
    BRIDGING_FINAL_REVALIDATION_EPISODES = 500
    BRIDGING_INITIAL_REHEARSAL_PERCENT = 50
    BRIDGING_MIN_REHEARSAL_PERCENT = 20
    BRIDGING_MAX_REHEARSAL_PERCENT = 90
    BRIDGING_REHEARSAL_TRANSITION_TARGET = 0.30
    BRIDGING_RETENTION_TRANSITION_TARGET = 0.50
    BRIDGING_RETENTION_WINDOW = 50
    BRIDGING_RETENTION_MIN_EPISODES = 30
    BRIDGING_RETENTION_FAILURE_MARGIN = 0.10

    def __init__(
        self,
        task_id: str,
        initial_stage: int = 0,
        *,
        stage_checkpoint_dir: Path | None = None,
        checkpoint_name_prefix: str | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.task_id = task_id
        mastery_episodes = (
            self.BRIDGING_MASTERY_EPISODES if task_id == "bridging" else 100
        )
        self.curriculum = (
            MasteryCurriculum(
                initial_stage,
                min_episodes=mastery_episodes,
                task_id=task_id,
            )
            if task_id in MasteryCurriculum.CONFIGS
            else None
        )
        self.stage_eligible_episodes = 0
        self.stage_checkpoint_dir = stage_checkpoint_dir
        self.checkpoint_name_prefix = checkpoint_name_prefix or task_id
        self._tool_neutrality_reported = False
        self._bridging_validation_requested = False
        self._bridging_last_validation_eligible = -self.BRIDGING_VALIDATION_RETRY_EPISODES
        self._bridging_exploration_boost_until = 0
        self._bridging_last_ent_coef: float | None = None
        self._bridging_active_lengths: deque[float] = deque(maxlen=100)
        self._bridging_rehearsal_lengths: deque[float] = deque(maxlen=100)
        self._bridging_retention_outcomes: dict[int, deque[float]] = {}
        self._bridging_rehearsal_percent = (
            self.BRIDGING_INITIAL_REHEARSAL_PERCENT if initial_stage > 0 else 0
        )
        self._bridging_retention_focus_stage = -1
        self._bridging_last_rehearsal_report: tuple[int, int] | None = None

    def _restore_bridging_state(self) -> None:
        if self.task_id != "bridging" or self.curriculum is None:
            return
        saved_stage = int(
            getattr(self.model, "bridging_curriculum_evidence_stage", -1)
        )
        if saved_stage == self.curriculum.stage_index:
            saved_successes = list(
                getattr(self.model, "bridging_curriculum_successes", [])
            )
            self.curriculum.successes.extend(
                float(value) for value in saved_successes[-self.curriculum.min_episodes :]
            )
            self.stage_eligible_episodes = int(
                getattr(self.model, "bridging_stage_eligible_episodes", 0)
            )
            self._bridging_rehearsal_percent = int(
                getattr(
                    self.model,
                    "bridging_rehearsal_percent",
                    self._bridging_rehearsal_percent,
                )
            )
            self._bridging_retention_focus_stage = int(
                getattr(self.model, "bridging_retention_focus_stage", -1)
            )
            self._bridging_active_lengths.extend(
                float(value)
                for value in getattr(self.model, "bridging_active_lengths", [])
            )
            self._bridging_rehearsal_lengths.extend(
                float(value)
                for value in getattr(self.model, "bridging_rehearsal_lengths", [])
            )
            saved_outcomes = getattr(
                self.model, "bridging_retention_outcomes", {}
            )
            for stage, outcomes in dict(saved_outcomes).items():
                history: deque[float] = deque(
                    (float(value) for value in outcomes),
                    maxlen=self.BRIDGING_RETENTION_WINDOW,
                )
                self._bridging_retention_outcomes[int(stage)] = history
        self._persist_bridging_state()

    def _persist_bridging_state(self) -> None:
        if self.task_id != "bridging" or self.curriculum is None:
            return
        self.model.bridging_curriculum_evidence_stage = self.curriculum.stage_index
        self.model.bridging_curriculum_successes = list(self.curriculum.successes)
        self.model.bridging_stage_eligible_episodes = self.stage_eligible_episodes
        self.model.bridging_rehearsal_percent = self._bridging_rehearsal_percent
        self.model.bridging_retention_focus_stage = (
            self._bridging_retention_focus_stage
        )
        self.model.bridging_active_lengths = list(self._bridging_active_lengths)
        self.model.bridging_rehearsal_lengths = list(
            self._bridging_rehearsal_lengths
        )
        self.model.bridging_retention_outcomes = {
            int(stage): list(outcomes)
            for stage, outcomes in self._bridging_retention_outcomes.items()
        }

    def _record_bridging_episode(self, info: dict[str, Any]) -> None:
        if self.task_id != "bridging" or self.curriculum is None:
            return
        episode = info.get("episode", {})
        length = float(episode.get("l", info.get("episode_steps", 0.0)))
        stage_index = int(info.get("curriculum_stage_index", -1))
        if bool(info.get("curriculum_rehearsal", False)):
            if length > 0.0:
                self._bridging_rehearsal_lengths.append(length)
            if 0 <= stage_index < self.curriculum.stage_index:
                history = self._bridging_retention_outcomes.setdefault(
                    stage_index,
                    deque(maxlen=self.BRIDGING_RETENTION_WINDOW),
                )
                history.append(float(bool(info.get("success", False))))
        elif stage_index == self.curriculum.stage_index and length > 0.0:
            self._bridging_active_lengths.append(length)

    def _update_bridging_retention_focus(self) -> None:
        assert self.curriculum is not None
        focus = self._bridging_retention_focus_stage
        if focus >= 0:
            history = self._bridging_retention_outcomes.get(focus, deque())
            threshold = self.curriculum.success_thresholds[focus]
            if (
                len(history) < self.BRIDGING_RETENTION_WINDOW
                or sum(history) / len(history) < threshold
            ):
                return
            self._bridging_retention_focus_stage = -1

        failures: list[tuple[float, int]] = []
        for stage, history in self._bridging_retention_outcomes.items():
            if len(history) < self.BRIDGING_RETENTION_MIN_EPISODES:
                continue
            threshold = self.curriculum.success_thresholds[stage]
            rate = sum(history) / len(history)
            if rate < threshold - self.BRIDGING_RETENTION_FAILURE_MARGIN:
                failures.append((rate - threshold, stage))
        if failures:
            _, failed_stage = min(failures)
            self._bridging_retention_focus_stage = failed_stage

    def _bridging_rehearsal_configuration(self) -> tuple[int, int]:
        assert self.curriculum is not None
        if self.curriculum.stage_index == 0:
            return 0, -1
        self._update_bridging_retention_focus()
        focus = self._bridging_retention_focus_stage
        if not self._bridging_active_lengths or not self._bridging_rehearsal_lengths:
            percent = self.BRIDGING_INITIAL_REHEARSAL_PERCENT
        else:
            active_length = sum(self._bridging_active_lengths) / len(
                self._bridging_active_lengths
            )
            rehearsal_length = sum(self._bridging_rehearsal_lengths) / len(
                self._bridging_rehearsal_lengths
            )
            transition_target = (
                self.BRIDGING_RETENTION_TRANSITION_TARGET
                if focus >= 0
                else self.BRIDGING_REHEARSAL_TRANSITION_TARGET
            )
            denominator = (
                rehearsal_length * (1.0 - transition_target)
                + transition_target * active_length
            )
            raw_percent = (
                100.0 * transition_target * active_length / denominator
                if denominator > 0.0
                else self.BRIDGING_INITIAL_REHEARSAL_PERCENT
            )
            percent = int(round(raw_percent / 5.0) * 5)
        if focus >= 0:
            percent = max(percent, 70)
        percent = max(
            self.BRIDGING_MIN_REHEARSAL_PERCENT,
            min(self.BRIDGING_MAX_REHEARSAL_PERCENT, percent),
        )
        return percent, focus

    def _apply_bridging_rehearsal(self, *, force: bool = False) -> None:
        if self.task_id != "bridging" or self.curriculum is None:
            return
        previous_percent = self._bridging_rehearsal_percent
        previous_focus = self._bridging_retention_focus_stage
        percent, focus = self._bridging_rehearsal_configuration()
        changed = (
            percent != previous_percent or focus != previous_focus
        )
        self._bridging_rehearsal_percent = percent
        self._bridging_retention_focus_stage = focus
        self.training_env.env_method(
            "set_curriculum_rehearsal", percent, focus
        )
        report = (percent, focus)
        if force or changed or report != self._bridging_last_rehearsal_report:
            focus_name = (
                self.curriculum.stages[focus] if focus >= 0 else "none"
            )
            active_mean = (
                sum(self._bridging_active_lengths)
                / len(self._bridging_active_lengths)
                if self._bridging_active_lengths
                else 0.0
            )
            rehearsal_mean = (
                sum(self._bridging_rehearsal_lengths)
                / len(self._bridging_rehearsal_lengths)
                if self._bridging_rehearsal_lengths
                else 0.0
            )
            print(
                f"[rehearsal] active={self.curriculum.stage} "
                f"episodes={percent}% retention_focus={focus_name} "
                f"mean_length_active={active_mean:.1f} "
                f"mean_length_rehearsal={rehearsal_mean:.1f}",
                flush=True,
            )
        self._bridging_last_rehearsal_report = report
        self._persist_bridging_state()

    def _reset_bridging_retention_tracking(self) -> None:
        self._bridging_active_lengths.clear()
        self._bridging_rehearsal_lengths.clear()
        self._bridging_retention_outcomes.clear()
        self._bridging_retention_focus_stage = -1
        self._bridging_rehearsal_percent = (
            self.BRIDGING_INITIAL_REHEARSAL_PERCENT
        )
        self._bridging_last_rehearsal_report = None

    def _bridging_target_ent_coef(self) -> float:
        assert self.curriculum is not None
        if self.stage_eligible_episodes < self._bridging_exploration_boost_until:
            return 0.005
        count = len(self.curriculum.successes)
        if count < 100:
            return 0.005
        success_rate = sum(self.curriculum.successes) / count
        stage_index = self.curriculum.stage_index
        threshold = self.curriculum.success_thresholds[stage_index]
        if success_rate < 0.60:
            return 0.005
        if success_rate < 0.80 * threshold:
            return 0.005
        if success_rate < threshold:
            return 0.001
        return 0.0

    def _bridging_window_ready(self) -> bool:
        assert self.curriculum is not None
        if len(self.curriculum.successes) < self.curriculum.min_episodes:
            return False
        threshold = self.curriculum.success_thresholds[
            self.curriculum.stage_index
        ]
        return (
            sum(self.curriculum.successes) / len(self.curriculum.successes)
            >= threshold
        )

    def _update_bridging_entropy(self, *, force: bool = False) -> None:
        if self.task_id != "bridging" or self.curriculum is None:
            return
        target = self._bridging_target_ent_coef()
        previous = float(getattr(self.model, "ent_coef", target))
        self.model.ent_coef = target
        if force or self._bridging_last_ent_coef is None or previous != target:
            count = len(self.curriculum.successes)
            rate = sum(self.curriculum.successes) / count if count else 0.0
            print(
                f"[entropy] stage={self.curriculum.stage} "
                f"eligible_window={count}/{self.curriculum.min_episodes} "
                f"success={rate:.1%} ent_coef={target:g}",
                flush=True,
            )
        self._bridging_last_ent_coef = target

    def _advance_curriculum(self, previous_stage: str) -> None:
        assert self.curriculum is not None
        mastered_stage_index = self.curriculum.stage_index - 1
        required_rate = self.curriculum.success_thresholds[mastered_stage_index]
        stage_checkpoint = self._save_mastered_stage(
            mastered_stage_index, previous_stage
        )
        if stage_checkpoint is not None:
            print(
                f"[checkpoint] mastered stage saved before switch: "
                f"{stage_checkpoint}.zip",
                flush=True,
            )
        eligible_episodes = self.stage_eligible_episodes
        self.stage_eligible_episodes = 0
        self._bridging_validation_requested = False
        self._bridging_last_validation_eligible = (
            -self.BRIDGING_VALIDATION_RETRY_EPISODES
        )
        self._bridging_exploration_boost_until = 0
        self._reset_bridging_retention_tracking()
        self._persist_bridging_state()
        self._synchronize()
        self._apply_bridging_rehearsal(force=True)
        self._prepare_bed_v2_tool_head()
        self._update_bridging_entropy(force=True)
        print(
            f"[curriculum] mastered {previous_stage}; "
            f"advancing to {self.curriculum.stage} "
            f"after {eligible_episodes} eligible episodes "
            f"(required success >= {required_rate:.1%})",
            flush=True,
        )

    def _bridging_validation_plan(self) -> list[tuple[int, int, float]]:
        assert self.curriculum is not None
        return [
            (
                stage_index,
                self.BRIDGING_FIXED_VALIDATION_EPISODES
                if stage_index in (0, 2)
                else self.BRIDGING_VALIDATION_EPISODES,
                self.curriculum.success_thresholds[stage_index],
            )
            for stage_index in range(self.curriculum.stage_index + 1)
        ]

    def _save_final_bridging_certification(self) -> Path | None:
        assert self.curriculum is not None
        self.model.bridging_final_certified = True
        self.model.bridging_final_certified_at = self.num_timesteps
        self._persist_bridging_state()
        return self._save_mastered_stage(
            self.curriculum.stage_index, self.curriculum.stage
        )

    def _prepare_bed_v2_tool_head(self) -> None:
        if self.task_id != "bed_defense_break_v2" or self.curriculum is None:
            return
        stage_index = self.curriculum.stage_index
        if stage_index <= 1:
            neutralize_bed_v2_tool_head(self.model)
            self.model.bed_v2_tool_head_released = False
            if not self._tool_neutrality_reported:
                print(
                    "[curriculum] bed-defense V2 tool head held uniform "
                    "through exposed-bed stages",
                    flush=True,
                )
                self._tool_neutrality_reported = True
        elif stage_index == 2 and not bool(
            getattr(self.model, "bed_v2_tool_head_released", False)
        ):
            neutralize_bed_v2_tool_head(self.model)
            self.model.bed_v2_tool_head_released = True
            print(
                "[curriculum] bed-defense V2 tool head reset to uniform and "
                "released for material learning",
                flush=True,
            )
        elif stage_index > 2:
            self.model.bed_v2_tool_head_released = True

    def _save_mastered_stage(self, stage_index: int, stage_name: str) -> Path | None:
        if self.stage_checkpoint_dir is None:
            return None
        self.stage_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_stage_name = re.sub(r"[^A-Za-z0-9._-]+", "_", stage_name)
        checkpoint_path = self.stage_checkpoint_dir / (
            f"{self.checkpoint_name_prefix}_stage_{stage_index:02d}_"
            f"{safe_stage_name}_mastered_{self.num_timesteps}_steps"
        )
        # For an intermediate stage the curriculum has already advanced, so
        # saved metadata must still identify the policy as belonging to the
        # stage it just mastered. The final certification is already on the
        # final stage and writes the same value.
        self.model.curriculum_stage_index = stage_index
        self.model.save(checkpoint_path)
        return checkpoint_path

    def _synchronize(self) -> None:
        self.training_env.env_method("set_training_progress", self.num_timesteps)
        if self.curriculum is not None:
            self.training_env.env_method(
                "set_curriculum_stage", self.curriculum.stage_index
            )
            self.model.curriculum_stage_index = self.curriculum.stage_index

    def _on_training_start(self) -> None:
        self._restore_bridging_state()
        self._synchronize()
        self._apply_bridging_rehearsal(force=True)
        self._prepare_bed_v2_tool_head()
        self._update_bridging_entropy(force=True)

    def _on_rollout_start(self) -> None:
        self._synchronize()
        self._apply_bridging_rehearsal()
        self._prepare_bed_v2_tool_head()
        self._update_bridging_entropy()

    def _on_step(self) -> bool:
        if self.curriculum is None:
            return True
        dones = self.locals.get("dones", [])
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self._record_bridging_episode(info)
            previous_stage = self.curriculum.stage
            episode_stage = str(info.get("curriculum_stage", ""))
            eligible = bool(info.get("curriculum_eligible", True))
            can_advance = self.curriculum.stage_index < len(self.curriculum.stages) - 1
            recorded = eligible and episode_stage == previous_stage and (
                can_advance or self.task_id == "bridging"
            )
            success = bool(info.get("success", False))
            if recorded:
                self.stage_eligible_episodes += 1
            advanced = self.curriculum.observe(
                episode_stage,
                success,
                eligible=eligible,
                advance=self.task_id != "bridging",
            )
            if recorded and self.task_id == "bridging":
                self._persist_bridging_state()
                validation_interval = (
                    self.BRIDGING_VALIDATION_RETRY_EPISODES
                    if can_advance
                    else self.BRIDGING_FINAL_REVALIDATION_EPISODES
                )
                validation_due = (
                    self._bridging_window_ready()
                    and self.stage_eligible_episodes
                    - self._bridging_last_validation_eligible
                    >= validation_interval
                )
                if validation_due:
                    self._bridging_validation_requested = True
            if advanced:
                self._advance_curriculum(previous_stage)
            elif recorded and self.stage_eligible_episodes % 20 == 0:
                window_rate = sum(self.curriculum.successes) / len(
                    self.curriculum.successes
                )
                threshold = self.curriculum.success_thresholds[
                    self.curriculum.stage_index
                ]
                print(
                    f"[mastery] stage={previous_stage} "
                    f"eligible={self.stage_eligible_episodes} "
                    f"window={len(self.curriculum.successes)}/"
                    f"{self.curriculum.min_episodes} "
                    f"success={window_rate:.1%} required={threshold:.1%}",
                    flush=True,
                )
        return True

    def _run_bridging_validation(self) -> bool:
        assert self.curriculum is not None
        environment = self.training_env
        active_stage = self.curriculum.stage_index
        stage_results: list[tuple[int, int, int, float, float]] = []
        restored_observation: Any | None = None
        environment.env_method("set_curriculum_validation_mode", True)
        try:
            for stage_index, target_episodes, threshold in (
                self._bridging_validation_plan()
            ):
                environment.env_method("set_curriculum_stage", stage_index)
                environment.seed(evaluation_seed(0x4D43524C, stage_index))
                observation = environment.reset()
                successes = 0
                episodes = 0
                while episodes < target_episodes:
                    actions, _ = self.model.predict(
                        observation, deterministic=True
                    )
                    observation, _, dones, infos = environment.step(actions)
                    for done, info in zip(dones, infos):
                        if not done or episodes >= target_episodes:
                            continue
                        episodes += 1
                        successes += int(bool(info.get("success", False)))
                success_rate = successes / episodes
                stage_results.append(
                    (stage_index, successes, episodes, success_rate, threshold)
                )
        finally:
            environment.env_method("set_curriculum_stage", active_stage)
            environment.env_method("set_curriculum_validation_mode", False)
            environment.seed(evaluation_seed(0x524553554D45, self.num_timesteps))
            restored_observation = environment.reset()
            self.model._last_obs = restored_observation
            self.model._last_episode_starts = np.ones(
                (environment.num_envs,), dtype=bool
            )

        passed = all(rate >= threshold for _, _, _, rate, threshold in stage_results)
        rendered = ", ".join(
            f"{self.curriculum.stages[index]}={successes}/{episodes}"
            f"({rate:.0%},need {threshold:.0%})"
            for index, successes, episodes, rate, threshold in stage_results
        )
        print(
            f"[deterministic-validation] active={self.curriculum.stage} "
            f"passed={passed} {rendered}",
            flush=True,
        )
        return passed

    def _on_rollout_end(self) -> None:
        if self.task_id != "bridging" or self.curriculum is None:
            return
        self._apply_bridging_rehearsal()
        # PPO does not optimize while episodes are being collected. Choose one
        # entropy coefficient from the completed rollout instead of oscillating
        # it after every vector-environment episode near a mastery threshold.
        self._update_bridging_entropy()
        if not self._bridging_validation_requested:
            return
        previous_stage = self.curriculum.stage
        self._bridging_validation_requested = False
        if not self._bridging_window_ready():
            count = len(self.curriculum.successes)
            rate = sum(self.curriculum.successes) / count if count else 0.0
            print(
                f"[deterministic-validation] active={previous_stage} "
                f"deferred=stochastic_window_changed "
                f"success={rate:.1%} required="
                f"{self.curriculum.success_thresholds[self.curriculum.stage_index]:.1%}",
                flush=True,
            )
            return
        self._bridging_last_validation_eligible = self.stage_eligible_episodes
        if self._run_bridging_validation():
            # No eligible training episodes changed during the isolated validation,
            # so the stochastic mastery window remains ready to advance.
            if self.curriculum.stage_index < len(self.curriculum.stages) - 1:
                self.curriculum.advance()
                self._advance_curriculum(previous_stage)
            else:
                checkpoint = self._save_final_bridging_certification()
                if checkpoint is not None:
                    print(
                        f"[checkpoint] final bridging curriculum certified: "
                        f"{checkpoint}.zip",
                        flush=True,
                    )
                print(
                    f"[curriculum] final stage certified after "
                    f"{self.stage_eligible_episodes} eligible episodes; "
                    "all stages passed deterministic validation",
                    flush=True,
                )
            return
        retry_episodes = (
            self.BRIDGING_VALIDATION_RETRY_EPISODES
            if self.curriculum.stage_index < len(self.curriculum.stages) - 1
            else self.BRIDGING_FINAL_REVALIDATION_EPISODES
        )
        self._bridging_exploration_boost_until = (
            self.stage_eligible_episodes + retry_episodes
        )
        self._update_bridging_entropy(force=True)


def make_environment_factory(
    *,
    task_id: str,
    max_episode_steps: int,
    host: str,
    port: int,
    instance_id: str,
    visualized: bool,
    target_tps: float,
    profile: bool = False,
    profile_java_phases: bool = False,
    strict_validation: bool = False,
) -> Callable[[], MinecraftEnv]:
    """Return a cloudpickle-safe constructor for one SB3 worker process."""

    def initialize() -> MinecraftEnv:
        worker_preset = get_task_preset(task_id)
        worker_task = worker_preset.create_task(max_episode_steps)
        worker_task.set_training_mode(True)
        worker_task.set_training_progress(0)
        return MinecraftEnv(
            worker_task,
            host=host,
            port=port,
            instance_id=instance_id,
            visualized=visualized,
            target_tps=target_tps,
            profile=profile,
            profile_java_phases=profile_java_phases,
            include_raw_info=False,
            strict_validation=strict_validation,
            info_mode="episode",
        )

    return initialize


def parse_args(default_task: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=task_names(),
        default=default_task,
        required=default_task is None,
        help="registered task id",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--base-port", "--port", dest="base_port", type=int, default=8765,
        help="TCP port for instance index 0; --port remains as a compatibility alias",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--vec-env",
        choices=("batched", "shared", "subproc", "lean-shared"),
        default="lean-shared",
        help=(
            "vector transport; lean-shared uses lightweight Windows workers (default); shared is the reference backend; "
            "batched owns all sockets in the trainer process with lower memory use"
        ),
    )
    parser.add_argument("--first-instance-index", type=int, default=0)
    parser.add_argument("--instance-prefix", default="rl")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--max-training-seconds", type=float, help="optional warmed training time limit")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-freq", type=int)
    parser.add_argument("--log-every-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="override the task preset (tasks without one use 3e-4)",
    )
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help=(
            "SB3 policy and learner device (CPU is the measured default for the current "
            "structured MLP workload; use cuda for an explicit comparison)"
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="PyTorch intra/inter-op CPU threads; 1 avoids contention with Minecraft",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--visualized",
        action="store_true",
        help="debug training with rendering, audio, and real-time pacing",
    )
    parser.add_argument("--tps", type=float, default=20.0)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print real rollout/worker/reset/PPO timing attribution",
    )
    parser.add_argument(
        "--profile-java-phases",
        action="store_true",
        help="include sampled client/server/serialization phase timings in observations",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="optional JSONL file for per-rollout profiling records",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="validate every action/raw/observation instead of only reset contracts",
    )
    parser.add_argument(
        "--profile-every-rollouts",
        type=int,
        default=1,
        help="profiling report interval when --profile is enabled",
    )
    args = parser.parse_args()
    if args.max_training_seconds is not None and args.max_training_seconds <= 0:
        parser.error("--max-training-seconds must be positive")

    preset = get_task_preset(args.task)
    if args.total_timesteps is None:
        args.total_timesteps = preset.default_timesteps
    if args.max_episode_steps is None:
        args.max_episode_steps = preset.default_max_episode_steps
    if args.checkpoint_freq is None:
        args.checkpoint_freq = preset.default_checkpoint_freq
    if args.output_dir is None:
        args.output_dir = Path("models") / preset.task_id

    for name in (
        "total_timesteps",
        "max_episode_steps",
        "checkpoint_freq",
        "log_every_episodes",
        "torch_threads",
        "profile_every_rollouts",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.first_instance_index < 0:
        parser.error("--first-instance-index must be non-negative")
    first_instance_id = f"{args.instance_prefix}-{args.first_instance_index}"
    last_instance_id = (
        f"{args.instance_prefix}-{args.first_instance_index + args.num_envs - 1}"
    )
    if any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", instance_id) is None
        for instance_id in (first_instance_id, last_instance_id)
    ):
        parser.error(
            "generated instance ids must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
    last_port = args.base_port + args.first_instance_index + args.num_envs - 1
    if args.base_port < 1 or last_port > 65535:
        parser.error("the requested instance port range must stay between 1 and 65535")
    if args.visualized and args.num_envs != 1:
        parser.error("--visualized is supported only with --num-envs 1")
    return args


def main(default_task: str | None = None) -> None:
    args = parse_args(default_task)
    resolved_device = resolve_training_device(args.device)
    torch_intraop, torch_interop = configure_torch_threads(args.torch_threads)
    print(
        f"Torch CPU threads: intra-op={torch_intraop}, inter-op={torch_interop}; "
        f"BLAS worker limit={args.torch_threads}"
    )
    device_reason = (
        "measured CPU choice for structured MLP policies"
        if args.device == "auto"
        else "explicit CLI selection"
    )
    print(
        f"Training device: {resolved_device} "
        f"(requested={args.device}; {device_reason})"
    )
    preset = get_task_preset(args.task)
    algorithm_class = get_algorithm_class(preset.algorithm)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    run_metadata = {
        "task": preset.task_id,
        "vector_backend": args.vec_env,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "device_selection": (
            "CPU selected for structured MLP policies based on local end-to-end "
            "CPU/CUDA benchmark"
            if args.device == "auto"
            else "explicit CLI selection"
        ),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    instance_connections = [
        (
            f"{args.instance_prefix}-{instance_index}",
            args.base_port + instance_index,
        )
        for instance_index in range(
            args.first_instance_index,
            args.first_instance_index + args.num_envs,
        )
    ]
    environment_factories = [
        make_environment_factory(
            task_id=preset.task_id,
            max_episode_steps=args.max_episode_steps,
            host=args.host,
            port=port,
            instance_id=instance_id,
            visualized=args.visualized,
            target_tps=args.tps,
            profile=args.profile,
            profile_java_phases=args.profile_java_phases,
            strict_validation=args.strict_validation,
        )
        for instance_id, port in instance_connections
    ]
    if args.vec_env == "lean-shared":
        vector_environment = LeanSharedMemoryVecEnv([
            {"task_id": preset.task_id, "max_episode_steps": args.max_episode_steps,
             "environment": {"host": args.host, "port": port, "instance_id": instance_id,
                 "visualized": args.visualized, "target_tps": args.tps,
                 "profile": args.profile, "profile_java_phases": args.profile_java_phases,
                 "include_raw_info": False, "strict_validation": args.strict_validation,
                 "info_mode": "episode"}}
            for instance_id, port in instance_connections
        ])
        print("Lightweight workers:", vector_environment.diagnostics(), flush=True)
    elif args.vec_env == "batched":
        vector_environment = BatchedMinecraftVecEnv(environment_factories)
    elif args.num_envs == 1:
        vector_environment = DummyVecEnv(environment_factories)
    else:
        vector_environment = (
            SharedMemoryVecEnv(environment_factories, start_method="spawn")
            if args.vec_env == "shared"
            else SubprocVecEnv(environment_factories, start_method="spawn")
        )
    env = VecMonitor(vector_environment, filename=str(output_dir / "monitor.csv"))

    if args.resume is not None:
        resume_path = args.resume.resolve()
        print(f"Loading {preset.algorithm.upper()} checkpoint: {resume_path}")
        model = algorithm_class.load(
            resume_path,
            env=env,
            device=resolved_device,
            custom_objects=checkpoint_custom_objects(
                env.observation_space, env.action_space
            ),
        )
        if preset.task_id == "bridging":
            apply_bridging_resume_safety(model)
            print(
                "Applied bridging resume safety: learning_rate=0.0001, "
                "n_epochs=4, clip_range=0.15, target_kl=0.02",
                flush=True,
            )
        reset_num_timesteps = False
    else:
        algorithm_kwargs = resolve_algorithm_kwargs(
            preset.algorithm_kwargs,
            num_envs=args.num_envs,
            n_steps_override=args.n_steps,
            batch_size_override=args.batch_size,
        )
        learning_rate = algorithm_kwargs.pop("learning_rate", 3e-4)
        if args.learning_rate is not None:
            learning_rate = args.learning_rate
        model = algorithm_class(
            preset.policy,
            env,
            learning_rate=learning_rate,
            verbose=1,
            seed=args.seed,
            device=resolved_device,
            tensorboard_log=None,
            **algorithm_kwargs,
        )
        if preset.task_id == "bed_defense_break_v2":
            initialize_bed_v2_action_biases(model)
            print(
                "Initialized bed-defense V2 exploration prior: "
                "camera zero-delta favored, attack=True favored"
            )
        reset_num_timesteps = True
    initial_curriculum_stage = int(getattr(model, "curriculum_stage_index", 0))
    env.env_method("set_training_progress", model.num_timesteps)
    env.env_method("set_curriculum_stage", initial_curriculum_stage)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_name_prefix = f"{preset.algorithm}_{preset.task_id}"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=checkpoint_name_prefix,
    )
    console_callback = EpisodeConsoleCallback(args.log_every_episodes)
    progress_callback = TrainingProgressCallback(
        preset.task_id,
        initial_curriculum_stage,
        stage_checkpoint_dir=checkpoint_dir,
        checkpoint_name_prefix=checkpoint_name_prefix,
    )
    callbacks: list[BaseCallback] = [
        checkpoint_callback,
        console_callback,
        progress_callback,
        SteadyThroughputCallback(output_dir / "throughput.json", max_seconds=args.max_training_seconds),
    ]
    if args.profile:
        callbacks.append(
            TrainingProfilerCallback(
                args.profile_every_rollouts,
                args.profile_output.resolve() if args.profile_output else None,
            )
        )
    latest_path = output_dir / "latest_model"
    final_path = output_dir / "final_model"

    print(
        f"Waiting for {args.num_envs} Minecraft instance(s): "
        + ", ".join(f"{instance_id}@{args.host}:{port}" for instance_id, port in instance_connections)
        + f"; task={preset.task_id}, "
        f"algorithm={preset.algorithm.upper()}, training={args.total_timesteps} steps, "
        f"output={output_dir}, visualized={args.visualized}, profile={args.profile}"
    )
    if hasattr(model, "n_steps"):
        print(
            f"PPO rollout: n_steps={model.n_steps} per environment, "
            f"total={model.n_steps * args.num_envs}, batch_size={model.batch_size}",
            flush=True,
        )
    training_completed = False
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        model.save(final_path)
        training_completed = True
        print(f"Training complete. Final model: {final_path}.zip")
    except KeyboardInterrupt:
        print("Training interrupted; saving the current model...", flush=True)
        model.save(latest_path)
        print(f"Interrupted model: {latest_path}.zip")
    except Exception:
        # Preserve as much progress as possible while keeping the original
        # exception and traceback visible to the caller.
        try:
            model.save(latest_path)
            print(
                f"Training failed; emergency model saved: {latest_path}.zip",
                flush=True,
            )
        except Exception as save_failure:
            print(
                f"Training failed and emergency save also failed: {save_failure!r}",
                flush=True,
            )
        raise
    finally:
        close_training_environment(env, force=not training_completed)


if __name__ == "__main__":
    main()
