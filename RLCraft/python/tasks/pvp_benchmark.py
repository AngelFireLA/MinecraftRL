"""Deterministic structured-state combat workload used for throughput testing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from minecraft_env import MinecraftTask


class PvpBenchmarkTask(MinecraftTask):
    """One client-controlled player fighting one reset-locked scripted zombie."""

    YAW_DEGREES = np.asarray(
        [-30.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 30.0],
        dtype=np.float32,
    )
    PITCH_DEGREES = np.asarray(
        [-15.0, -5.0, -1.0, 0.0, 1.0, 5.0, 15.0], dtype=np.float32
    )

    def __init__(self, max_episode_steps: int = 240) -> None:
        super().__init__(
            task_id="pvp_benchmark",
            action_space=spaces.MultiDiscrete(
                [len(self.YAW_DEGREES), len(self.PITCH_DEGREES), 5, 2, 2, 2, 3]
            ),
            observation_space=spaces.Box(
                low=-1.0, high=1.0, shape=(28,), dtype=np.float32
            ),
            action_fields=(
                "mouse_dx",
                "mouse_dy",
                "forward",
                "back",
                "left",
                "right",
                "jump",
                "sprint",
                "attack",
                "attack_once",
                "hotbar_slot",
            ),
            observation_fields=(
                "player.x",
                "player.y",
                "player.z",
                "player.yaw",
                "player.pitch",
                "player.health",
                "player.on_ground",
                "player.motion_x",
                "player.motion_y",
                "player.motion_z",
                "player.sprinting",
                "player.collided_horizontally",
                "player.selected_slot",
                "task_entity.id",
                "task_entity.relative_x",
                "task_entity.relative_y",
                "task_entity.relative_z",
                "task_entity.distance",
                "task_entity.yaw_delta",
                "task_entity.pitch_delta",
                "task_entity.motion_x",
                "task_entity.motion_y",
                "task_entity.motion_z",
                "task_entity.health",
                "task_entity.alive",
                "raycast.type",
                "raycast.id",
            ),
            max_episode_steps=max_episode_steps,
        )
        self._origin = np.zeros(3, dtype=np.float32)
        self._last_attack = False

    def encode_action(self, action: Any) -> dict[str, Any]:
        values = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(values):
            raise ValueError(f"invalid PvP benchmark action: {action!r}")
        direction = int(values[2])
        self._last_attack = bool(values[5])
        return {
            "mouse_dx": float(self.YAW_DEGREES[int(values[0])]) / 0.15,
            "mouse_dy": -float(self.PITCH_DEGREES[int(values[1])]) / 0.15,
            "forward": direction == 1,
            "back": direction == 2,
            "left": direction == 3,
            "right": direction == 4,
            "sprint": bool(values[3]),
            "jump": bool(values[4]),
            "attack": self._last_attack,
            # Vanilla entity attacks are click-edge driven. Keep the held
            # control for parity while pulsing one click on each policy action.
            "attack_once": self._last_attack,
            "hotbar_slot": int(values[6]),
        }

    @staticmethod
    def _clip(value: Any, scale: float) -> float:
        return float(np.clip(float(value) / scale, -1.0, 1.0))

    def observation(self, raw: Mapping[str, Any]) -> np.ndarray:
        player_position = np.asarray(
            [raw["player.x"], raw["player.y"], raw["player.z"]], dtype=np.float32
        )
        relative_player = np.clip((player_position - self._origin) / 8.0, -1.0, 1.0)
        yaw_radians = math.radians(float(raw["player.yaw"]))
        aimed = raw["raycast.type"] == "ENTITY" and int(raw["raycast.id"]) == int(
            raw["task_entity.id"]
        )
        slot = int(raw["player.selected_slot"])
        return np.asarray(
            [
                *relative_player,
                math.sin(yaw_radians),
                math.cos(yaw_radians),
                self._clip(raw["player.pitch"], 90.0),
                self._clip(raw["player.health"], 20.0),
                float(bool(raw["player.on_ground"])),
                self._clip(raw["player.motion_x"], 1.0),
                self._clip(raw["player.motion_y"], 1.0),
                self._clip(raw["player.motion_z"], 1.0),
                float(bool(raw["player.sprinting"])),
                float(bool(raw["player.collided_horizontally"])),
                *[float(slot == index) for index in range(3)],
                self._clip(raw["task_entity.relative_x"], 8.0),
                self._clip(raw["task_entity.relative_y"], 4.0),
                self._clip(raw["task_entity.relative_z"], 8.0),
                self._clip(raw["task_entity.distance"], 12.0),
                self._clip(raw["task_entity.yaw_delta"], 180.0),
                self._clip(raw["task_entity.pitch_delta"], 90.0),
                self._clip(raw["task_entity.motion_x"], 1.0),
                self._clip(raw["task_entity.motion_y"], 1.0),
                self._clip(raw["task_entity.motion_z"], 1.0),
                self._clip(raw["task_entity.health"], 20.0),
                float(bool(raw["task_entity.alive"])),
                float(aimed),
            ],
            dtype=np.float32,
        )

    def on_reset(self, initial: Mapping[str, Any]) -> None:
        if not bool(initial["world_reset_applied"]):
            raise RuntimeError("Java did not apply the pvp_benchmark reset profile")
        if int(initial["task_entity.id"]) < 0 or not bool(
            initial["task_entity.alive"]
        ):
            raise RuntimeError("pvp_benchmark reset did not create a living opponent")
        self._origin = np.asarray(
            [initial["player.x"], initial["player.y"], initial["player.z"]],
            dtype=np.float32,
        )
        self._last_attack = False

    def reward(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        opponent_damage = max(
            0.0,
            float(previous["task_entity.health"])
            - float(current["task_entity.health"]),
        )
        player_damage = max(
            0.0, float(previous["player.health"]) - float(current["player.health"])
        )
        angular_improvement = self._angular_error(previous) - self._angular_error(current)
        return float(
            -0.005
            + opponent_damage * 0.5
            - player_damage * 0.25
            + angular_improvement / 720.0
        )

    def terminated(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        return not bool(current["task_entity.alive"]) or float(
            current["player.health"]
        ) <= 0.0

    def info(
        self, raw: Mapping[str, Any], *, episode_end: bool = True
    ) -> dict[str, Any]:
        info = super().info(raw, episode_end=episode_end)
        if episode_end:
            won = not bool(raw["task_entity.alive"])
            info.update(
                {
                    "success": won,
                    "opponent_alive": bool(raw["task_entity.alive"]),
                    "opponent_health": float(raw["task_entity.health"]),
                    "player_health": float(raw["player.health"]),
                    "distance": float(raw["task_entity.distance"]),
                }
            )
        return info

    @staticmethod
    def _angular_error(raw: Mapping[str, Any]) -> float:
        return math.hypot(
            float(raw["task_entity.yaw_delta"]),
            float(raw["task_entity.pitch_delta"]),
        )
