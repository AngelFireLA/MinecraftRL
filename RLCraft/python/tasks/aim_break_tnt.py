"""Task: rotate toward a randomized TNT block and break it with one click."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from minecraft_env import MinecraftTask


class AimBreakTntTask(MinecraftTask):
    """A small, shaped-reward task for the first learned Minecraft skill."""

    YAW_DEGREES = np.asarray(
        [-18.0, -6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0, 18.0],
        dtype=np.float32,
    )
    PITCH_DEGREES = np.asarray(
        [-12.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 12.0],
        dtype=np.float32,
    )

    def __init__(self, max_episode_steps: int = 80) -> None:
        super().__init__(
            task_id="aim_break_tnt",
            action_space=spaces.MultiDiscrete(
                [len(self.YAW_DEGREES), len(self.PITCH_DEGREES), 2]
            ),
            observation_space=spaces.Box(
                low=np.asarray([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            action_fields=("mouse_dx", "mouse_dy", "attack_once"),
            observation_fields=(
                "task_goal.x",
                "task_goal.y",
                "task_goal.z",
                "task_goal.distance",
                "task_goal.yaw_delta",
                "task_goal.pitch_delta",
                "task_goal.block_id",
                "raycast.type",
                "raycast.x",
                "raycast.y",
                "raycast.z",
            ),
            max_episode_steps=max_episode_steps,
        )
        self._target_block_id: int | None = None
        self._last_attack_once = False

    def encode_action(self, action: Any) -> dict[str, Any]:
        action_array = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action_array):
            raise ValueError(f"invalid aim/break action: {action!r}")
        yaw_degrees = float(self.YAW_DEGREES[int(action_array[0])])
        pitch_degrees = float(self.PITCH_DEGREES[int(action_array[1])])
        self._last_attack_once = bool(action_array[2])
        # Entity.setAngles applies +0.15 * mouse_dx to yaw and
        # -0.15 * mouse_dy to pitch.
        return {
            "mouse_dx": yaw_degrees / 0.15,
            "mouse_dy": -pitch_degrees / 0.15,
            "attack_once": self._last_attack_once,
        }

    def observation(self, raw: Mapping[str, Any]) -> np.ndarray:
        yaw_delta = float(raw["task_goal.yaw_delta"])
        pitch_delta = float(raw["task_goal.pitch_delta"])
        return np.asarray(
            [
                np.clip(yaw_delta / 45.0, -1.0, 1.0),
                np.clip(pitch_delta / 45.0, -1.0, 1.0),
                np.clip(float(raw["task_goal.distance"]) / 5.0, 0.0, 1.0),
                float(self.is_aimed(raw)),
                float(self.target_present(raw)),
            ],
            dtype=np.float32,
        )

    def on_reset(self, initial: Mapping[str, Any]) -> None:
        if not bool(initial["world_reset_applied"]):
            raise RuntimeError("Java did not apply the aim_break_tnt reset profile")
        block_id = int(initial["task_goal.block_id"])
        if block_id == 0:
            raise RuntimeError("aim_break_tnt reset produced air instead of TNT")
        self._target_block_id = block_id
        self._last_attack_once = False

    def reward(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        if self.success(current):
            return 10.0

        previous_error = self.angular_error(previous)
        current_error = self.angular_error(current)
        reward = -0.01 + 0.20 * (previous_error - current_error) / 45.0
        if self.is_aimed(current):
            reward += 0.05
        if self._last_attack_once and not self.is_aimed(previous):
            reward -= 0.05
        return reward

    def terminated(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        return self.success(current)

    def info(
        self, raw: Mapping[str, Any], *, episode_end: bool = True
    ) -> dict[str, Any]:
        info = super().info(raw, episode_end=episode_end)
        if not episode_end:
            return info
        info.update(
            {
                "aimed": self.is_aimed(raw),
                "success": self.success(raw),
                "angular_error_degrees": self.angular_error(raw),
                "target_block_id": int(raw["task_goal.block_id"]),
            }
        )
        return info

    def success(self, raw: Mapping[str, Any]) -> bool:
        return self._target_block_id is not None and not self.target_present(raw)

    def target_present(self, raw: Mapping[str, Any]) -> bool:
        block_id = int(raw["task_goal.block_id"])
        return self._target_block_id is None or block_id == self._target_block_id

    @staticmethod
    def angular_error(raw: Mapping[str, Any]) -> float:
        return math.hypot(
            float(raw["task_goal.yaw_delta"]),
            float(raw["task_goal.pitch_delta"]),
        )

    @staticmethod
    def is_aimed(raw: Mapping[str, Any]) -> bool:
        if raw["raycast.type"] != "BLOCK":
            return False
        return (
            int(raw["raycast.x"]) == math.floor(float(raw["task_goal.x"]))
            and int(raw["raycast.y"]) == math.floor(float(raw["task_goal.y"]))
            and int(raw["raycast.z"]) == math.floor(float(raw["task_goal.z"]))
        )

    def describe_action(self, action: Any) -> str:
        action_array = np.asarray(action, dtype=np.int64)
        return (
            f"yaw={self.YAW_DEGREES[int(action_array[0])]:+g}deg "
            f"pitch={self.PITCH_DEGREES[int(action_array[1])]:+g}deg "
            f"click={bool(action_array[2])}"
        )

    def describe_reset(self, raw: Mapping[str, Any]) -> str:
        return (
            f"target=({float(raw['task_goal.x']):.1f},"
            f"{float(raw['task_goal.y']):.1f},"
            f"{float(raw['task_goal.z']):.1f}) "
            f"initial_error={self.angular_error(raw):.2f}deg"
        )
