"""Registry connecting task definitions to default training/evaluation presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from bed_defense_features import BedDefenseFlatFeaturesExtractor
from minecraft_env import MinecraftTask
from tasks import (
    AimBreakTntTask,
    AimToolBreakTask,
    BedDefenseBreakTask,
    BedDefenseBreakV2Task,
    BridgingTask,
    MultiToolBreakTask,
    PvpBenchmarkTask,
)


@dataclass(frozen=True)
class TaskPreset:
    task_id: str
    factory: Callable[[int], MinecraftTask]
    algorithm: str = "ppo"
    policy: str = "MlpPolicy"
    default_timesteps: int = 100_000
    default_max_episode_steps: int = 80
    default_checkpoint_freq: int = 10_000
    algorithm_kwargs: dict[str, Any] = field(default_factory=dict)

    def create_task(self, max_episode_steps: int | None = None) -> MinecraftTask:
        episode_steps = (
            self.default_max_episode_steps
            if max_episode_steps is None
            else max_episode_steps
        )
        return self.factory(episode_steps)


TASK_PRESETS: dict[str, TaskPreset] = {
    "aim_break_tnt": TaskPreset(
        task_id="aim_break_tnt",
        factory=AimBreakTntTask,
        algorithm="ppo",
        algorithm_kwargs={"n_steps": 512, "batch_size": 64},
    ),
    "aim_tool_break": TaskPreset(
        task_id="aim_tool_break",
        factory=AimToolBreakTask,
        algorithm="ppo",
        default_timesteps=300_000,
        default_max_episode_steps=100,
        default_checkpoint_freq=25_000,
        algorithm_kwargs={"n_steps": 1024, "batch_size": 128},
    ),
    "multi_tool_break": TaskPreset(
        task_id="multi_tool_break",
        factory=MultiToolBreakTask,
        algorithm="ppo",
        default_timesteps=750_000,
        default_max_episode_steps=300,
        default_checkpoint_freq=50_000,
        algorithm_kwargs={"n_steps": 2048, "batch_size": 256},
    ),
    "pvp_benchmark": TaskPreset(
        task_id="pvp_benchmark",
        factory=PvpBenchmarkTask,
        algorithm="ppo",
        default_timesteps=500_000,
        default_max_episode_steps=240,
        default_checkpoint_freq=50_000,
        algorithm_kwargs={
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 5,
            "policy_kwargs": {"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
        },
    ),
    "bed_defense_break": TaskPreset(
        task_id="bed_defense_break",
        factory=BedDefenseBreakTask,
        algorithm="ppo",
        policy="MultiInputPolicy",
        default_timesteps=2_000_000,
        default_max_episode_steps=600,
        default_checkpoint_freq=50_000,
        algorithm_kwargs={
            "n_steps": 2048,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "n_epochs": 5,
            "clip_range": 0.15,
            "target_kl": 0.02,
            "gamma": BedDefenseBreakTask.GAMMA,
            "ent_coef": 0.0,
            "policy_kwargs": {
                "features_extractor_class": BedDefenseFlatFeaturesExtractor,
                "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            },
        },
    ),
    "bed_defense_break_v2": TaskPreset(
        task_id="bed_defense_break_v2",
        factory=BedDefenseBreakV2Task,
        algorithm="ppo",
        policy="MultiInputPolicy",
        default_timesteps=2_000_000,
        default_max_episode_steps=320,
        default_checkpoint_freq=50_000,
        algorithm_kwargs={
            "n_steps": 2048,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "n_epochs": 5,
            "clip_range": 0.15,
            "target_kl": 0.02,
            "gamma": BedDefenseBreakV2Task.GAMMA,
            "ent_coef": 0.0,
            "policy_kwargs": {
                "features_extractor_class": BedDefenseFlatFeaturesExtractor,
                "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            },
        },
    ),
    "bridging": TaskPreset(
        task_id="bridging",
        factory=BridgingTask,
        algorithm="ppo",
        policy="MultiInputPolicy",
        default_timesteps=5_000_000,
        default_max_episode_steps=850,
        default_checkpoint_freq=50_000,
        algorithm_kwargs={
            "n_steps": 2048,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "n_epochs": 4,
            "clip_range": 0.15,
            "target_kl": 0.02,
            "gamma": BridgingTask.GAMMA,
            "ent_coef": 0.005,
            "policy_kwargs": {
                "features_extractor_class": BedDefenseFlatFeaturesExtractor,
                "net_arch": {"pi": [256, 128], "vf": [256, 128]},
            },
        },
    ),
}


def task_names() -> tuple[str, ...]:
    return tuple(sorted(TASK_PRESETS))


def get_task_preset(task_id: str) -> TaskPreset:
    try:
        return TASK_PRESETS[task_id]
    except KeyError as error:
        raise ValueError(
            f"unknown task {task_id!r}; available={', '.join(task_names())}"
        ) from error


__all__ = ["TaskPreset", "get_task_preset", "task_names"]
