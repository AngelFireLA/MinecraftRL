"""Curriculum task for speed-bridging across a straight void gap."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from minecraft_env import MinecraftTask
from seed_utils import splitmix64
from tasks.aim_tool_break import AimToolBreakTask


class BridgingTask(MinecraftTask):
    """Learn to place blocks and traverse a one-block-wide straight bridge."""

    CURRICULUM_STAGES = (
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
    )
    FIXED_GAPS = (1, 1, 5, 5, 8, 11, 15, 22, 30, None)
    MAX_GAP = 64
    GRID_LONGITUDINAL = MAX_GAP + 2
    GRID_VERTICAL = 1
    GRID_CELLS = GRID_LONGITUDINAL
    STATE_SIZE = 32

    # Relative camera controls retain fine corrections needed for edge work and
    # larger turns needed by randomized reset poses.
    YAW_DEGREES = AimToolBreakTask.YAW_DEGREES
    PITCH_DEGREES = AimToolBreakTask.PITCH_DEGREES
    FACE_NAMES = ("DOWN", "UP", "NORTH", "SOUTH", "WEST", "EAST")

    STEP_PENALTY = 0.005
    SUCCESS_BASE_REWARD = 10.0
    SUCCESS_TIME_REWARD = 10.0
    FALL_PENALTY = 10.0
    TIMEOUT_PENALTY = 8.0
    SAFE_PROGRESS_TOTAL_REWARD = 4.0
    BRIDGE_EXTENSION_TOTAL_REWARD = 1.5
    INVALID_PLACE_PENALTY = 0.05
    LATERAL_BLOCK_PENALTY = 0.10
    ILLEGAL_HEIGHT_PENALTY = 10.0
    SAFE_GROUND_TICKS = 3
    GAMMA = 0.99
    MIN_REHEARSAL_PERCENT = 20
    INITIAL_REHEARSAL_PERCENT = 50
    MAX_REHEARSAL_PERCENT = 90

    def __init__(self, max_episode_steps: int = 850) -> None:
        state_low = np.asarray(
            [-66.0, -66.0, -32.0, -90.0, -1.0, -1.0, 0.0, 0.0]
            + [-66.0] * 4
            + [0.0] * 2
            + [0.0] * 3
            + [0.0] * 3
            + [0.0]
            + [0.0] * 6
            + [0.0, 0.0]
            + [-6.0] * 2
            + [0.0],
            dtype=np.float32,
        )
        state_high = np.asarray(
            [2.0, 66.0, 32.0, 90.0, 1.0, 1.0, 1.0, 10_000.0]
            + [66.0] * 4
            + [1.0] * 2
            + [1.0] * 3
            + [1.0] * 3
            + [1.0]
            + [1.0] * 6
            + [6.0, 1.0]
            + [6.0] * 2
            + [10_000.0],
            dtype=np.float32,
        )
        if state_low.size != self.STATE_SIZE or state_high.size != self.STATE_SIZE:
            raise RuntimeError("bridging state bounds have the wrong size")
        super().__init__(
            task_id="bridging",
            action_space=spaces.MultiDiscrete(
                [
                    2,  # place/use
                    3,  # nothing/left/right
                    3,  # nothing/back/forward
                    2,  # sprint
                    2,  # sneak
                    len(self.YAW_DEGREES),
                    len(self.PITCH_DEGREES),
                ]
            ),
            observation_space=spaces.Dict(
                {
                    "state": spaces.Box(state_low, state_high, dtype=np.float32),
                    "grid": spaces.Box(
                        low=0,
                        high=1,
                        shape=(1, self.GRID_VERTICAL, 1, self.GRID_LONGITUDINAL),
                        dtype=np.uint8,
                    ),
                }
            ),
            action_fields=(
                "use_once",
                "left",
                "right",
                "back",
                "forward",
                "sprint",
                "sneak",
                "mouse_dx",
                "mouse_dy",
            ),
            observation_fields=(
                "player.x",
                "player.y",
                "player.z",
                "player.yaw",
                "player.pitch",
                "player.motion_x",
                "player.motion_z",
                "player.on_ground",
                "player.sneaking",
                "player.sprinting",
                "held.count",
                "raycast.type",
                "raycast.side",
                "raycast.distance",
                "task_goal.x",
                "task_goal.y",
                "task_goal.z",
                "bridging.ground_fraction",
                "bridging.distance_forward_edge",
                "bridging.distance_left_edge",
                "bridging.distance_right_edge",
                "bridging.distance_rear_edge",
                "bridging.can_place",
                "bridging.block_grid",
                "bridging.above_level_count",
            ),
            max_episode_steps=max_episode_steps,
        )
        self._training_mode = False
        self._curriculum_stage_index = 0
        self._curriculum_validation_mode = False
        final_stage = len(self.CURRICULUM_STAGES) - 1
        self._active_curriculum_stage = final_stage
        self._active_stage = final_stage
        self._stage_name = self.CURRICULUM_STAGES[-1]
        self._curriculum_rehearsal = False
        self._curriculum_rehearsal_kind = "none"
        self._rehearsal_percent = self.INITIAL_REHEARSAL_PERCENT
        self._rehearsal_focus_stage = -1
        self._gap = self.MAX_GAP
        self._episode_limit = max_episode_steps
        self._last_use = False
        self._last_lateral = 0
        self._last_longitudinal = 0
        self._last_sprint = False
        self._last_sneak = False
        self._ticks_since_ground = 0
        self._grounded_ticks = 0
        self._ticks_since_place = 0
        self._best_safe_longitudinal = -(self.MAX_GAP + 1.0)
        self._furthest_bridge_index = -1
        self._success = False
        self._fell = False
        self._timed_out = False
        self._illegal_height = False
        self._episode_steps = 0
        self._blocks_placed = 0
        self._invalid_place_attempts = 0
        self._lateral_blocks_placed = 0
        self._safe_progress_reward = 0.0
        self._extension_reward = 0.0
        self._illegal_height_penalty = 0.0

    def set_training_mode(self, enabled: bool) -> None:
        self._training_mode = bool(enabled)

    def set_curriculum_stage(self, stage_index: int) -> None:
        if not 0 <= stage_index < len(self.CURRICULUM_STAGES):
            raise ValueError(
                "bridging curriculum stage must be between 0 and "
                f"{len(self.CURRICULUM_STAGES) - 1}"
            )
        self._curriculum_stage_index = int(stage_index)

    def set_curriculum_validation_mode(self, enabled: bool) -> None:
        """Disable rehearsal while validating the active curriculum stage."""

        self._curriculum_validation_mode = bool(enabled)

    def set_curriculum_rehearsal(
        self, percent: int, focus_stage: int = -1
    ) -> None:
        percent = int(percent)
        focus_stage = int(focus_stage)
        if not 0 <= percent <= self.MAX_REHEARSAL_PERCENT:
            raise ValueError(
                f"bridging rehearsal percent must be between 0 and "
                f"{self.MAX_REHEARSAL_PERCENT}"
            )
        if focus_stage >= self._curriculum_stage_index or focus_stage < -1:
            raise ValueError("bridging rehearsal focus must be a prior stage")
        self._rehearsal_percent = percent
        self._rehearsal_focus_stage = focus_stage

    def curriculum_configuration(self) -> tuple[int, str]:
        stage = (
            self._curriculum_stage_index
            if self._training_mode
            else len(self.CURRICULUM_STAGES) - 1
        )
        return stage, self.CURRICULUM_STAGES[stage]

    def reset_payload(self, seed: int, options: Mapping[str, Any]) -> dict[str, Any]:
        active_stage, _ = self.curriculum_configuration()
        stage = active_stage
        rehearsal_kind = "none"
        if (
            self._training_mode
            and not self._curriculum_validation_mode
            and active_stage > 0
        ):
            mixed = splitmix64(seed)
            bucket = int(mixed % 100)
            if bucket < self._rehearsal_percent:
                if self._rehearsal_focus_stage >= 0:
                    stage = self._rehearsal_focus_stage
                    rehearsal_kind = "retention_focus"
                elif bucket < (self._rehearsal_percent + 1) // 2:
                    stage = active_stage - 1
                    rehearsal_kind = "previous"
                else:
                    stage = int((mixed >> 8) % active_stage)
                    rehearsal_kind = "earlier"
        name = self.CURRICULUM_STAGES[stage]
        self._active_curriculum_stage = active_stage
        self._active_stage = stage
        self._stage_name = name
        self._curriculum_rehearsal = stage != active_stage
        self._curriculum_rehearsal_kind = rehearsal_kind
        fixed_gap = self.FIXED_GAPS[stage]
        self._gap = (
            int(fixed_gap)
            if fixed_gap is not None
            else 16 + int((int(seed) & ((1 << 63) - 1)) % 49)
        )
        # Java derives the identical variable gap from seed. The field selects
        # pose difficulty; horizons remain task-local and scale with actual gap.
        self._episode_limit = min(self.max_episode_steps, 40 + 12 * self._gap)
        return {"curriculum_stage": stage}

    def encode_action(self, action: Any) -> dict[str, Any]:
        values = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(values):
            raise ValueError(f"invalid bridging action: {action!r}")
        self._last_use = bool(values[0])
        self._last_lateral = int(values[1])
        self._last_longitudinal = int(values[2])
        self._last_sprint = bool(values[3])
        self._last_sneak = bool(values[4])
        yaw = float(self.YAW_DEGREES[int(values[5])])
        pitch = float(self.PITCH_DEGREES[int(values[6])])
        return {
            "use_once": self._last_use,
            "left": self._last_lateral == 1,
            "right": self._last_lateral == 2,
            "back": self._last_longitudinal == 1,
            "forward": self._last_longitudinal == 2,
            "sprint": self._last_sprint,
            "sneak": self._last_sneak,
            "mouse_dx": yaw / 0.15,
            "mouse_dy": -pitch / 0.15,
        }

    @staticmethod
    def _wrap_yaw(yaw: float) -> float:
        return (yaw + 180.0) % 360.0 - 180.0

    def _relative_coordinates(self, raw: Mapping[str, Any]) -> tuple[float, float, float]:
        # Arenas are aligned toward south (+Z). Horizontal coordinates use target center;
        # vertical zero is the target block's top surface (the player's feet goal).
        return (
            float(raw["player.z"]) - float(raw["task_goal.z"]),
            float(raw["player.x"]) - float(raw["task_goal.x"]),
            float(raw["player.y"]) - (float(raw["task_goal.y"]) + 0.5),
        )

    def _grid(self, raw: Mapping[str, Any]) -> np.ndarray:
        values = np.asarray(raw["bridging.block_grid"], dtype=np.uint8)
        if values.shape != (self.GRID_CELLS,):
            raise RuntimeError(
                f"bridging grid has shape {values.shape}, expected ({self.GRID_CELLS},)"
            )
        return values.reshape(self.GRID_VERTICAL, self.GRID_LONGITUDINAL)

    def observation(self, raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
        longitudinal, lateral, vertical = self._relative_coordinates(raw)
        # Physical yaw zero points from the start toward the target and is also
        # model-relative yaw zero.
        yaw = self._wrap_yaw(float(raw["player.yaw"]))
        looking = raw["raycast.type"] == "BLOCK"
        side = str(raw["raycast.side"]) if looking else ""
        state = np.asarray(
            [
                longitudinal,
                lateral,
                vertical,
                float(raw["player.pitch"]),
                math.sin(math.radians(yaw)),
                math.cos(math.radians(yaw)),
                float(np.clip(raw["bridging.ground_fraction"], 0.0, 1.0)),
                float(self._ticks_since_ground),
                float(raw["bridging.distance_forward_edge"]),
                float(raw["bridging.distance_left_edge"]),
                float(raw["bridging.distance_right_edge"]),
                float(raw["bridging.distance_rear_edge"]),
                float(bool(raw["player.sneaking"])),
                float(bool(raw["player.sprinting"])),
                *[float(self._last_lateral == value) for value in range(3)],
                *[float(self._last_longitudinal == value) for value in range(3)],
                float(looking),
                *[float(side == face) for face in self.FACE_NAMES],
                float(raw["raycast.distance"]) if looking else 0.0,
                float(bool(raw["bridging.can_place"])),
                float(raw["player.motion_z"]),
                float(raw["player.motion_x"]),
                float(self._ticks_since_place),
            ],
            dtype=np.float32,
        )
        if state.shape != (self.STATE_SIZE,):
            raise RuntimeError(
                f"bridging state produced {state.size} values, expected {self.STATE_SIZE}"
            )
        grid = self._grid(raw)[None, :, None, :]
        return {"state": state, "grid": grid}

    def on_reset(self, initial: Mapping[str, Any]) -> None:
        if not bool(initial["world_reset_applied"]):
            raise RuntimeError("Java did not apply the bridging reset profile")
        grid = self._grid(initial)
        if int(initial["held.count"]) != 64:
            raise RuntimeError("bridging reset must provide exactly 64 blocks")
        if int(initial["bridging.above_level_count"]) != 0:
            raise RuntimeError("bridging reset left blocks above the bridge level")
        if self._active_stage in (0, 2) and (
            initial["raycast.type"] != "BLOCK"
            or not bool(initial["bridging.can_place"])
        ):
            raise RuntimeError(
                "bridging aimed reset is not looking at a placeable block face: "
                f"raycast={initial['raycast.type']!r}, "
                f"side={initial['raycast.side']!r}, "
                f"distance={initial['raycast.distance']!r}, "
                f"can_place={initial['bridging.can_place']!r}, "
                f"yaw={initial['player.yaw']!r}, pitch={initial['player.pitch']!r}, "
                f"z={initial['player.z']!r}"
            )
        target_index = self.GRID_LONGITUDINAL - 1
        start_index = target_index - self._gap - 1
        if not grid[0, target_index] or not grid[0, start_index]:
            raise RuntimeError("bridging reset is missing its start or target bedrock")
        self._last_use = False
        self._last_lateral = 0
        self._last_longitudinal = 0
        self._last_sprint = False
        self._last_sneak = False
        self._ticks_since_ground = 0
        self._grounded_ticks = 1
        self._ticks_since_place = 0
        self._best_safe_longitudinal = float(
            self._relative_coordinates(initial)[0]
        )
        occupied_before_target = np.flatnonzero(grid[0, :target_index])
        self._furthest_bridge_index = int(occupied_before_target.max())
        self._success = False
        self._fell = False
        self._timed_out = False
        self._illegal_height = False
        self._episode_steps = 0
        self._blocks_placed = 0
        self._invalid_place_attempts = 0
        self._lateral_blocks_placed = 0
        self._safe_progress_reward = 0.0
        self._extension_reward = 0.0
        self._illegal_height_penalty = 0.0

    def _standing_on_target(self, raw: Mapping[str, Any]) -> bool:
        longitudinal, lateral, vertical = self._relative_coordinates(raw)
        return (
            bool(raw["player.on_ground"])
            and float(raw["bridging.ground_fraction"]) > 0.0
            and abs(longitudinal) <= 0.5
            and abs(lateral) <= 0.5
            and abs(vertical) <= 0.08
        )

    def reward(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        self._episode_steps += 1
        reward = -self.STEP_PENALTY
        on_ground = bool(current["player.on_ground"]) and float(
            current["bridging.ground_fraction"]
        ) > 0.0
        if on_ground:
            self._ticks_since_ground = 0
            self._grounded_ticks += 1
        else:
            self._ticks_since_ground += 1
            self._grounded_ticks = 0

        previous_count = int(previous["held.count"])
        current_count = int(current["held.count"])
        placed = max(previous_count - current_count, 0)
        if placed:
            self._blocks_placed += placed
            self._ticks_since_place = 0
        else:
            self._ticks_since_place += 1

        previous_grid = self._grid(previous)
        current_grid = self._grid(current)
        above_placed = max(
            int(current["bridging.above_level_count"])
            - int(previous["bridging.above_level_count"]),
            0,
        )
        self._illegal_height = above_placed > 0
        new_line_blocks = np.argwhere(
            (previous_grid == 0) & (current_grid == 1)
        )
        target_index = self.GRID_LONGITUDINAL - 1
        new_longitudinal_indices = (
            new_line_blocks[:, 1]
            if new_line_blocks.size
            else np.empty(0, dtype=np.int64)
        )
        valid_extensions = new_longitudinal_indices[
            new_longitudinal_indices < target_index
        ]
        if valid_extensions.size:
            new_furthest = max(self._furthest_bridge_index, int(valid_extensions.max()))
            extension = max(new_furthest - self._furthest_bridge_index, 0)
            if extension:
                gained = self.BRIDGE_EXTENSION_TOTAL_REWARD * extension / self._gap
                reward += gained
                self._extension_reward += gained
                self._furthest_bridge_index = new_furthest

        lateral_placed = max(
            placed - int(new_line_blocks.shape[0]) - above_placed, 0
        )
        if lateral_placed:
            self._lateral_blocks_placed += lateral_placed
            reward -= self.LATERAL_BLOCK_PENALTY * lateral_placed
        if self._last_use and placed == 0:
            self._invalid_place_attempts += 1
            reward -= self.INVALID_PLACE_PENALTY

        # Progress is distance closed toward target zero; walking beyond it can
        # never farm additional shaping reward.
        longitudinal = min(self._relative_coordinates(current)[0], 0.0)
        if self._grounded_ticks >= self.SAFE_GROUND_TICKS:
            progress = max(longitudinal - self._best_safe_longitudinal, 0.0)
            if progress:
                gained = self.SAFE_PROGRESS_TOTAL_REWARD * progress / (self._gap + 1)
                reward += gained
                self._safe_progress_reward += gained
                self._best_safe_longitudinal = longitudinal

        self._success = not self._illegal_height and self._standing_on_target(current)
        initial_feet_y = float(current["task_goal.y"]) + 0.5
        self._fell = float(current["player.y"]) < initial_feet_y
        self._timed_out = self._episode_steps >= self._episode_limit
        if self._illegal_height:
            reward -= self.ILLEGAL_HEIGHT_PENALTY
            self._illegal_height_penalty += self.ILLEGAL_HEIGHT_PENALTY
        elif self._success:
            time_fraction = max(0.0, 1.0 - self._episode_steps / self._episode_limit)
            reward += self.SUCCESS_BASE_REWARD + self.SUCCESS_TIME_REWARD * time_fraction
        elif self._fell:
            reward -= self.FALL_PENALTY
        elif self._timed_out:
            reward -= self.TIMEOUT_PENALTY
        return float(reward)

    def terminated(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        return self._success

    def truncated(self, step_count: int, current: Mapping[str, Any]) -> bool:
        return (
            self._illegal_height
            or self._fell
            or self._timed_out
            or step_count >= self.max_episode_steps
        )

    def info(self, raw: Mapping[str, Any], *, episode_end: bool = True) -> dict[str, Any]:
        if not episode_end:
            return {}
        return {
            "task_id": self.task_id,
            "success": self._success,
            "fell": self._fell,
            "timed_out": self._timed_out,
            "illegal_height": self._illegal_height,
            "curriculum_stage": self._stage_name,
            "curriculum_stage_index": self._active_stage,
            "curriculum_active_stage": self.CURRICULUM_STAGES[
                self._active_curriculum_stage
            ],
            "curriculum_active_stage_index": self._active_curriculum_stage,
            "curriculum_eligible": not self._curriculum_rehearsal,
            "curriculum_rehearsal": self._curriculum_rehearsal,
            "curriculum_rehearsal_kind": self._curriculum_rehearsal_kind,
            "curriculum_rehearsal_percent": self._rehearsal_percent,
            "curriculum_rehearsal_focus_stage": self._rehearsal_focus_stage,
            "gap": self._gap,
            "episode_tick_limit": self._episode_limit,
            "blocks_placed": self._blocks_placed,
            "invalid_place_attempts": self._invalid_place_attempts,
            "lateral_blocks_placed": self._lateral_blocks_placed,
            "safe_progress_reward_total": self._safe_progress_reward,
            "extension_reward_total": self._extension_reward,
            "illegal_height_penalty_total": self._illegal_height_penalty,
        }

    def describe_action(self, action: Any) -> str:
        values = np.asarray(action, dtype=np.int64)
        return (
            f"place={bool(values[0])} lateral={('none', 'left', 'right')[values[1]]} "
            f"longitudinal={('none', 'back', 'forward')[values[2]]} "
            f"sprint={bool(values[3])} sneak={bool(values[4])} "
            f"yaw={self.YAW_DEGREES[values[5]]:+g}deg "
            f"pitch={self.PITCH_DEGREES[values[6]]:+g}deg"
        )

    def describe_reset(self, raw: Mapping[str, Any]) -> str:
        return f"task=bridging stage={self._stage_name} gap={self._gap}"


__all__ = ["BridgingTask"]
