"""Task: aim at stone, oak planks, or sand and break it with the right tool."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from minecraft_env import MinecraftTask


class AimToolBreakTask(MinecraftTask):
    """Learn aiming, absolute hotbar selection, and sustained block breaking."""

    YAW_DEGREES = np.asarray(
        [-90.0, -30.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 30.0, 90.0],
        dtype=np.float32,
    )
    PITCH_DEGREES = np.asarray(
        [-45.0, -15.0, -5.0, -1.0, -0.25, 0.0, 0.25, 1.0, 5.0, 15.0, 45.0],
        dtype=np.float32,
    )

    STONE_BLOCK_ID = 1
    OAK_PLANKS_BLOCK_ID = 5
    SAND_BLOCK_ID = 12
    BLOCK_NAMES = {
        STONE_BLOCK_ID: "stone",
        OAK_PLANKS_BLOCK_ID: "oak_planks",
        SAND_BLOCK_ID: "sand",
    }
    TOOL_NAMES = ("diamond_pickaxe", "diamond_axe", "iron_shovel")

    def __init__(self, max_episode_steps: int = 100) -> None:
        super().__init__(
            task_id="aim_tool_break",
            action_space=spaces.MultiDiscrete(
                [len(self.YAW_DEGREES), len(self.PITCH_DEGREES), 3, 2]
            ),
            observation_space=spaces.Box(
                low=np.asarray([-1.0, -1.0] + [0.0] * 10, dtype=np.float32),
                high=np.ones(12, dtype=np.float32),
                dtype=np.float32,
            ),
            action_fields=("mouse_dx", "mouse_dy", "hotbar_slot", "attack"),
            observation_fields=(
                "task_goal.x",
                "task_goal.y",
                "task_goal.z",
                "task_goal.distance",
                "task_goal.yaw_delta",
                "task_goal.pitch_delta",
                "task_goal.block_id",
                "task_goal.tool_speed_0",
                "task_goal.tool_speed_1",
                "task_goal.tool_speed_2",
                "player.selected_slot",
                "player.block_break_progress",
                "raycast.type",
                "raycast.x",
                "raycast.y",
                "raycast.z",
            ),
            max_episode_steps=max_episode_steps,
        )
        self._target_block_id: int | None = None
        self._last_attack = False

    def encode_action(self, action: Any) -> dict[str, Any]:
        action_array = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action_array):
            raise ValueError(f"invalid aim/tool/break action: {action!r}")
        yaw_degrees = float(self.YAW_DEGREES[int(action_array[0])])
        pitch_degrees = float(self.PITCH_DEGREES[int(action_array[1])])
        self._last_attack = bool(action_array[3])
        return {
            "mouse_dx": yaw_degrees / 0.15,
            "mouse_dy": -pitch_degrees / 0.15,
            "hotbar_slot": int(action_array[2]),
            "attack": self._last_attack,
        }

    def observation(self, raw: Mapping[str, Any]) -> np.ndarray:
        selected_slot = int(raw["player.selected_slot"])
        tool_speeds = [
            np.clip(float(raw[f"task_goal.tool_speed_{slot}"]), 0.0, 1.0)
            for slot in range(3)
        ]
        slot_one_hot = [float(selected_slot == slot) for slot in range(3)]
        return np.asarray(
            [
                np.clip(float(raw["task_goal.yaw_delta"]) / 180.0, -1.0, 1.0),
                np.clip(float(raw["task_goal.pitch_delta"]) / 90.0, -1.0, 1.0),
                np.clip(float(raw["task_goal.distance"]) / 5.0, 0.0, 1.0),
                float(self.is_aimed(raw)),
                float(self.target_present(raw)),
                np.clip(float(raw["player.block_break_progress"]), 0.0, 1.0),
                *tool_speeds,
                *slot_one_hot,
            ],
            dtype=np.float32,
        )

    def on_reset(self, initial: Mapping[str, Any]) -> None:
        if not bool(initial["world_reset_applied"]):
            raise RuntimeError("Java did not apply the aim_tool_break reset profile")
        block_id = int(initial["task_goal.block_id"])
        if block_id == 0:
            raise RuntimeError("aim_tool_break reset produced air")
        self._target_block_id = block_id
        self._last_attack = False

    def reward(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        if self.success(current):
            return 10.0

        reward = -0.02
        reward += 0.20 * (
            self.angular_error(previous) - self.angular_error(current)
        ) / 180.0

        previous_progress = float(previous["player.block_break_progress"])
        current_progress = float(current["player.block_break_progress"])
        if current_progress >= previous_progress:
            reward += 0.50 * (current_progress - previous_progress)

        if self._last_attack:
            if not self.is_aimed(previous):
                reward -= 0.03
            elif not self.correct_tool(current):
                reward -= 0.02
        return reward

    def terminated(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        return self.success(current)

    def info(
        self, raw: Mapping[str, Any], *, episode_end: bool = True
    ) -> dict[str, Any]:
        info = super().info(raw, episode_end=episode_end)
        if not episode_end:
            return info
        block_id = int(raw["task_goal.block_id"])
        info.update(
            {
                "aimed": self.is_aimed(raw),
                "success": self.success(raw),
                "angular_error_degrees": self.angular_error(raw),
                "target_block_id": block_id,
                "target_block": self.BLOCK_NAMES.get(
                    self._target_block_id or block_id,
                    f"block_{self._target_block_id or block_id}",
                ),
                "selected_slot": int(raw["player.selected_slot"]),
                "correct_tool": self.correct_tool(raw),
                "tool_speeds": self.tool_speeds(raw),
                "break_progress": float(raw["player.block_break_progress"]),
            }
        )
        return info

    def success(self, raw: Mapping[str, Any]) -> bool:
        return self._target_block_id is not None and not self.target_present(raw)

    def target_present(self, raw: Mapping[str, Any]) -> bool:
        block_id = int(raw["task_goal.block_id"])
        return self._target_block_id is None or block_id == self._target_block_id

    def correct_tool(self, raw: Mapping[str, Any]) -> bool:
        return int(raw["player.selected_slot"]) == self.correct_slot(raw)

    def correct_slot(self, raw: Mapping[str, Any]) -> int:
        return int(np.argmax(self.tool_speeds(raw)))

    @staticmethod
    def tool_speeds(raw: Mapping[str, Any]) -> tuple[float, float, float]:
        return tuple(float(raw[f"task_goal.tool_speed_{slot}"]) for slot in range(3))

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
            f"tool={self.TOOL_NAMES[int(action_array[2])]} "
            f"attack={bool(action_array[3])}"
        )

    def describe_reset(self, raw: Mapping[str, Any]) -> str:
        block_id = int(raw["task_goal.block_id"])
        return (
            f"block={self.BLOCK_NAMES.get(block_id, f'block_{block_id}')} "
            f"target=({float(raw['task_goal.x']):.1f},"
            f"{float(raw['task_goal.y']):.1f},"
            f"{float(raw['task_goal.z']):.1f}) "
            f"initial_error={self.angular_error(raw):.2f}deg"
        )
