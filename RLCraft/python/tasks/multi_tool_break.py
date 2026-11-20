"""Task: aim at and efficiently break all 3–5 randomized target blocks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from minecraft_env import MAX_TASK_BLOCKS, MinecraftTask
from tasks.aim_tool_break import AimToolBreakTask


class MultiToolBreakTask(MinecraftTask):
    """A fixed-size multi-target curriculum with count-normalized rewards."""

    CURRICULUM_STAGES = (
        ("one_guided", 1, True),
        ("two_guided", 2, True),
        ("three_guided", 3, True),
        ("multi_guided", 0, True),
        ("multi_free", 0, False),
    )

    YAW_DEGREES = AimToolBreakTask.YAW_DEGREES
    PITCH_DEGREES = AimToolBreakTask.PITCH_DEGREES
    TOOL_NAMES = AimToolBreakTask.TOOL_NAMES
    BLOCK_NAMES = AimToolBreakTask.BLOCK_NAMES

    TARGET_ATTRIBUTES = (
        "x", "y", "z", "distance", "yaw_delta", "pitch_delta", "block_id",
        "tool_speed_0", "tool_speed_1", "tool_speed_2",
    )

    def __init__(self, max_episode_steps: int = 300) -> None:
        target_fields = tuple(
            f"task_block_{index}.{attribute}"
            for index in range(MAX_TASK_BLOCKS)
            for attribute in self.TARGET_ATTRIBUTES
        )
        observation_low = [0.0] * 4
        for _ in range(MAX_TASK_BLOCKS):
            observation_low.extend([-1.0, -1.0] + [0.0] * 7)
        super().__init__(
            task_id="multi_tool_break",
            action_space=spaces.MultiDiscrete(
                [len(self.YAW_DEGREES), len(self.PITCH_DEGREES), 3, 2]
            ),
            observation_space=spaces.Box(
                low=np.asarray(observation_low, dtype=np.float32),
                high=np.ones(4 + 9 * MAX_TASK_BLOCKS, dtype=np.float32),
                dtype=np.float32,
            ),
            action_fields=("mouse_dx", "mouse_dy", "hotbar_slot", "attack"),
            observation_fields=(
                "task_blocks.count",
                *target_fields,
                "player.selected_slot",
                "player.block_break_progress",
                "raycast.type",
                "raycast.x",
                "raycast.y",
                "raycast.z",
            ),
            max_episode_steps=max_episode_steps,
        )
        self._initial_block_ids: list[int] = []
        self._target_count = 0
        self._training_mode = False
        self._training_steps = 0
        self._curriculum_stage_index = 0
        self._curriculum_stage = "full_free"
        self._focus_enabled = False
        self._focus_index = -1
        self._last_attack = False

    def set_training_mode(self, enabled: bool) -> None:
        self._training_mode = bool(enabled)

    def set_training_progress(self, timesteps: int) -> None:
        if timesteps < 0:
            raise ValueError("training timesteps cannot be negative")
        self._training_steps = int(timesteps)

    def set_curriculum_stage(self, stage_index: int) -> None:
        if not 0 <= stage_index < len(self.CURRICULUM_STAGES):
            raise ValueError(
                f"curriculum stage must be between 0 and "
                f"{len(self.CURRICULUM_STAGES) - 1}"
            )
        self._curriculum_stage_index = int(stage_index)

    def reset_payload(self, seed: int, options: Mapping[str, Any]) -> dict[str, Any]:
        target_count, focus_enabled, stage = self.curriculum_configuration()
        self._focus_enabled = focus_enabled
        self._curriculum_stage = stage
        return {"target_count": target_count}

    def curriculum_configuration(self) -> tuple[int, bool, str]:
        if not self._training_mode:
            return 0, False, "full_free"
        stage, target_count, focus_enabled = self.CURRICULUM_STAGES[
            self._curriculum_stage_index
        ]
        return target_count, focus_enabled, stage

    def encode_action(self, action: Any) -> dict[str, Any]:
        action_array = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action_array):
            raise ValueError(f"invalid multi-tool action: {action!r}")
        self._last_attack = bool(action_array[3])
        return {
            "mouse_dx": float(self.YAW_DEGREES[int(action_array[0])]) / 0.15,
            "mouse_dy": -float(self.PITCH_DEGREES[int(action_array[1])]) / 0.15,
            "hotbar_slot": int(action_array[2]),
            "attack": self._last_attack,
        }

    def observation(self, raw: Mapping[str, Any]) -> np.ndarray:
        selected_slot = int(raw["player.selected_slot"])
        values: list[float] = [
            np.clip(self.remaining_count(raw) / MAX_TASK_BLOCKS, 0.0, 1.0),
            np.clip(float(raw["player.block_break_progress"]), 0.0, 1.0),
            selected_slot / 2.0,
            float(self.aimed_index(raw) >= 0),
        ]
        for source_index in self.model_target_order(raw):
            present = self.target_present(raw, source_index)
            values.extend(
                [
                    np.clip(
                        float(raw[self.key(source_index, "yaw_delta")]) / 180.0,
                        -1.0,
                        1.0,
                    ),
                    np.clip(
                        float(raw[self.key(source_index, "pitch_delta")]) / 90.0,
                        -1.0,
                        1.0,
                    ),
                    np.clip(
                        float(raw[self.key(source_index, "distance")]) / 5.0,
                        0.0,
                        1.0,
                    ),
                    float(present),
                    float(present and self.is_aimed_at(raw, source_index)),
                    float(present and source_index == self._focus_index),
                    *[
                        np.clip(speed, 0.0, 1.0)
                        for speed in self.tool_speeds(raw, source_index)
                    ],
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def model_target_order(self, raw: Mapping[str, Any]) -> list[int]:
        """Put the guided focus first without changing its stable raw identity."""
        indices = list(range(MAX_TASK_BLOCKS))
        if not self._focus_enabled or not self.target_present(raw, self._focus_index):
            return indices
        return [self._focus_index] + [
            index for index in indices if index != self._focus_index
        ]

    def on_reset(self, initial: Mapping[str, Any]) -> None:
        if not bool(initial["world_reset_applied"]):
            raise RuntimeError("Java did not apply the multi_tool_break reset profile")
        count = int(initial["task_blocks.count"])
        if not 1 <= count <= MAX_TASK_BLOCKS:
            raise RuntimeError(f"multi_tool_break produced {count} targets")
        block_ids = [int(initial[self.key(index, "block_id")]) for index in range(count)]
        if any(block_id == 0 for block_id in block_ids):
            raise RuntimeError("multi_tool_break produced an air target")
        self._target_count = count
        self._initial_block_ids = block_ids
        self._focus_index = 0 if self._focus_enabled else -1
        self._last_attack = False

    def reward(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        previous_remaining = self.remaining_count(previous)
        current_remaining = self.remaining_count(current)
        newly_broken = max(0, previous_remaining - current_remaining)
        reward = -0.01 + newly_broken * (10.0 / self._target_count)

        previous_progress = float(previous["player.block_break_progress"])
        current_progress = float(current["player.block_break_progress"])
        if newly_broken == 0:
            # Signed deltas make abandoning a partly mined block repay the
            # shaping reward previously earned on it. The old one-sided delta
            # could be farmed by repeatedly acquiring and losing progress.
            reward += 2.0 * (current_progress - previous_progress)
            reward += self.aim_potential(current) - self.aim_potential(previous)
            reward += 0.25 * (
                self.aimed_potential(current) - self.aimed_potential(previous)
            )
            reward += 0.50 * (
                self.tool_potential(current) - self.tool_potential(previous)
            )
            if self._last_attack:
                target = self.guidance_index(current)
                valid_attack = target >= 0 and self.is_aimed_at(current, target)
                valid_attack = valid_attack and (
                    int(current["player.selected_slot"])
                    == self.correct_slot(current, target)
                )
                reward += 0.03 if valid_attack else -0.01

        if current_remaining == 0:
            reward += 5.0

        self._advance_focus(current)
        return reward

    def terminated(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        return self._target_count > 0 and self.remaining_count(current) == 0

    def info(
        self, raw: Mapping[str, Any], *, episode_end: bool = True
    ) -> dict[str, Any]:
        info = super().info(raw, episode_end=episode_end)
        if not episode_end:
            return info
        remaining = self.remaining_count(raw)
        info.update(
            {
                "target_count": self._target_count,
                "remaining_blocks": remaining,
                "broken_blocks": self._target_count - remaining,
                "completion_fraction": (
                    (self._target_count - remaining) / self._target_count
                    if self._target_count
                    else 0.0
                ),
                "aimed_index": self.aimed_index(raw),
                "focus_index": self._focus_index,
                "curriculum_stage": self._curriculum_stage,
                "curriculum_stage_index": self._curriculum_stage_index,
                "curriculum_steps": self._training_steps,
                "success": self._target_count > 0 and remaining == 0,
            }
        )
        return info

    def remaining_indices(self, raw: Mapping[str, Any]) -> list[int]:
        return [
            index for index in range(self._target_count)
            if self.target_present(raw, index)
        ]

    def remaining_count(self, raw: Mapping[str, Any]) -> int:
        return len(self.remaining_indices(raw))

    def target_present(self, raw: Mapping[str, Any], index: int) -> bool:
        if index < 0 or index >= self._target_count or index >= len(self._initial_block_ids):
            return False
        return int(raw[self.key(index, "block_id")]) == self._initial_block_ids[index]

    def aimed_index(self, raw: Mapping[str, Any]) -> int:
        for index in self.remaining_indices(raw):
            if self.is_aimed_at(raw, index):
                return index
        return -1

    def is_aimed_at(self, raw: Mapping[str, Any], index: int) -> bool:
        return raw["raycast.type"] == "BLOCK" and all(
            int(raw[f"raycast.{axis}"])
            == math.floor(float(raw[self.key(index, axis)]))
            for axis in ("x", "y", "z")
        )

    def minimum_angular_error(self, raw: Mapping[str, Any]) -> float:
        remaining = self.remaining_indices(raw)
        if not remaining:
            return 0.0
        return min(
            math.hypot(
                float(raw[self.key(index, "yaw_delta")]),
                float(raw[self.key(index, "pitch_delta")]),
            )
            for index in remaining
        )

    def guidance_angular_error(self, raw: Mapping[str, Any]) -> float:
        target = self.guidance_index(raw)
        if target < 0:
            return 0.0
        return math.hypot(
            float(raw[self.key(target, "yaw_delta")]),
            float(raw[self.key(target, "pitch_delta")]),
        )

    def guidance_index(self, raw: Mapping[str, Any]) -> int:
        if self._focus_enabled and self.target_present(raw, self._focus_index):
            return self._focus_index
        remaining = self.remaining_indices(raw)
        if not remaining:
            return -1
        return min(
            remaining,
            key=lambda index: math.hypot(
                float(raw[self.key(index, "yaw_delta")]),
                float(raw[self.key(index, "pitch_delta")]),
            ),
        )

    def aim_potential(self, raw: Mapping[str, Any]) -> float:
        return 1.0 - min(self.guidance_angular_error(raw) / 180.0, 1.0)

    def aimed_potential(self, raw: Mapping[str, Any]) -> float:
        target = self.guidance_index(raw)
        return float(target >= 0 and self.is_aimed_at(raw, target))

    def tool_potential(self, raw: Mapping[str, Any]) -> float:
        target = self.guidance_index(raw)
        if target < 0:
            return 0.0
        speeds = self.tool_speeds(raw, target)
        best_speed = max(speeds)
        if best_speed <= 0.0:
            return 0.0
        selected_slot = int(raw["player.selected_slot"])
        return float(np.clip(speeds[selected_slot] / best_speed, 0.0, 1.0))

    def _advance_focus(self, raw: Mapping[str, Any]) -> None:
        if not self._focus_enabled:
            self._focus_index = -1
            return
        if self.target_present(raw, self._focus_index):
            return
        remaining = self.remaining_indices(raw)
        self._focus_index = remaining[0] if remaining else -1

    def tool_speeds(self, raw: Mapping[str, Any], index: int) -> tuple[float, float, float]:
        return tuple(
            float(raw[self.key(index, f"tool_speed_{slot}")])
            for slot in range(3)
        )

    def correct_slot(self, raw: Mapping[str, Any], index: int) -> int:
        return int(np.argmax(self.tool_speeds(raw, index)))

    def describe_action(self, action: Any) -> str:
        action_array = np.asarray(action, dtype=np.int64)
        return (
            f"yaw={self.YAW_DEGREES[int(action_array[0])]:+g}deg "
            f"pitch={self.PITCH_DEGREES[int(action_array[1])]:+g}deg "
            f"tool={self.TOOL_NAMES[int(action_array[2])]} "
            f"attack={bool(action_array[3])}"
        )

    def describe_reset(self, raw: Mapping[str, Any]) -> str:
        names = [
            self.BLOCK_NAMES.get(int(raw[self.key(index, "block_id")]), "block")
            for index in range(self._target_count)
        ]
        return (
            f"stage={self._curriculum_stage} targets={self._target_count} "
            f"focus={self._focus_index} blocks={names}"
        )

    @staticmethod
    def key(index: int, attribute: str) -> str:
        return f"task_block_{index}.{attribute}"
