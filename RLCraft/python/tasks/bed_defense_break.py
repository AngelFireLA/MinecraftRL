"""Task: expose and destroy a two-block BedWars-style defended bed."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from minecraft_env import MinecraftTask
from tasks.aim_tool_break import AimToolBreakTask


class BedDefenseBreakTask(MinecraftTask):
    """One-policy curriculum for stationary BedWars defense breaking."""

    GRID_X = 7
    GRID_Y = 5
    GRID_Z = 7
    GRID_CELLS = GRID_X * GRID_Y * GRID_Z
    CATEGORY_UNKNOWN = 0
    CATEGORY_AIR = 1
    CATEGORY_BED = 2
    CATEGORY_WOOL = 3
    CATEGORY_WOOD = 4
    CATEGORY_END_STONE = 5
    CATEGORY_OBSIDIAN = 6
    CATEGORY_OTHER = 7
    CATEGORY_NAMES = {
        CATEGORY_UNKNOWN: "unknown",
        CATEGORY_AIR: "air",
        CATEGORY_BED: "bed",
        CATEGORY_WOOL: "wool",
        CATEGORY_WOOD: "wood",
        CATEGORY_END_STONE: "end_stone",
        CATEGORY_OBSIDIAN: "obsidian",
        CATEGORY_OTHER: "other",
    }
    BREAK_COST = {
        CATEGORY_BED: 1.0,
        CATEGORY_WOOL: 1.0,
        CATEGORY_WOOD: 2.0,
        CATEGORY_END_STONE: 3.0,
        CATEGORY_OBSIDIAN: 10.0,
    }
    CURRICULUM_STAGES = (
        ("exposed_near", 0, 1.00),
        ("exposed_random", 1, 1.00),
        ("single_mixed_aimed", 2, 1.00),
        ("single_mixed", 3, 1.00),
        ("tunnel_guided", 4, 1.00),
        ("layer_one", 5, 1.00),
        ("layer_two", 6, 1.00),
        ("varied_pose", 7, 1.00),
        ("obsidian_guided", 8, 1.00),
        ("layered_hint_75", 9, 0.75),
        ("layered_hint_50", 10, 0.50),
        ("layered_hint_25", 11, 0.25),
        ("layered_free", 12, 0.00),
    )
    STAGE_STEP_LIMITS = (120, 180, 240, 300, 400, 450, 600, 600, 600, 600, 600, 600, 600)
    YAW_DEGREES = AimToolBreakTask.YAW_DEGREES
    PITCH_DEGREES = AimToolBreakTask.PITCH_DEGREES
    TOOL_NAMES = ("shears", "diamond_axe", "diamond_pickaxe")
    BED_ATTRIBUTES = ("x", "y", "z", "block_id")
    GRID_CHANNEL_NAMES = (
        "known",
        "solid",
        "wool",
        "wood",
        "end_stone",
        "obsidian",
        "bed_goal",
        "guidance",
    )
    STATE_SIZE = 50
    MATERIAL_FEATURE_CATEGORIES = (
        CATEGORY_BED,
        CATEGORY_WOOL,
        CATEGORY_WOOD,
        CATEGORY_END_STONE,
        CATEGORY_OBSIDIAN,
        CATEGORY_OTHER,
    )
    EYE_HEIGHT = 1.62
    REACH_DISTANCE = 4.5
    AIM_REACH_DISTANCE = 4.48
    MIN_AIM_INSET = 0.01
    BED_HEIGHT = 0.5625
    # Keep guided rays away from hitbox edges.  The old 0.02--0.06 margins
    # made a tiny systematic pitch bias enough to leave the block face during
    # long obsidian breaks.
    BED_X_AIM_INSET = 0.12
    BED_Z_AIM_INSET = 0.12
    BED_VERTICAL_AIM_INSET = 0.10
    BLOCK_AIM_INSET = 0.12
    GAMMA = 0.99
    ANGULAR_ERROR_COST = 0.05
    AIM_STABILITY_DEGREES = 4.0
    AIMED_CAMERA_MOTION_COST = 0.01
    FOCUS_LOSS_COST = 0.25
    TOOL_CHANGE_COST = 0.03
    ROUTE_SWITCH_MARGIN = 0.5
    TOOL_SELECTION_REWARD = 0.50
    PROGRESS_ACHIEVEMENT_REWARD = 4.0

    def __init__(self, max_episode_steps: int = 600) -> None:
        bed_fields = tuple(
            f"task_block_{index}.{attribute}"
            for index in range(2)
            for attribute in self.BED_ATTRIBUTES
        )
        super().__init__(
            task_id="bed_defense_break",
            action_space=spaces.MultiDiscrete(
                [len(self.YAW_DEGREES), len(self.PITCH_DEGREES), 3, 2]
            ),
            observation_space=spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(self.STATE_SIZE,),
                        dtype=np.float32,
                    ),
                    "grid": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(
                            len(self.GRID_CHANNEL_NAMES),
                            self.GRID_Y,
                            self.GRID_Z,
                            self.GRID_X,
                        ),
                        # Every channel is binary. uint8 reduces the
                        # SubprocVecEnv payload by 4x; the extractor converts
                        # it to float before applying learned weights.
                        dtype=np.uint8,
                    ),
                }
            ),
            action_fields=("mouse_dx", "mouse_dy", "hotbar_slot", "attack"),
            observation_fields=(
                "task_blocks.count",
                *bed_fields,
                "player.x",
                "player.y",
                "player.z",
                "player.yaw",
                "player.pitch",
                "player.on_ground",
                "player.motion_y",
                "player.selected_slot",
                "player.block_break_progress",
                "raycast.type",
                "raycast.id",
                "raycast.x",
                "raycast.y",
                "raycast.z",
                "bed_defense.origin_x",
                "bed_defense.origin_y",
                "bed_defense.origin_z",
                "bed_defense.block_grid",
                "bed_defense.visible_mask",
            ),
            max_episode_steps=max_episode_steps,
        )
        self._training_mode = False
        self._training_steps = 0
        self._curriculum_stage_index = 0
        self._curriculum_stage = "full_free"
        self._guidance_enabled = False
        self._guidance_override: bool | None = None
        self._last_attack = False
        self._last_yaw_degrees = 0.0
        self._last_pitch_degrees = 0.0
        self._initial_defense_blocks = 0
        self._initial_route_cost = 1.0
        self._locked_bed_position: tuple[int, int, int] | None = None
        self._tunnel_depth = 0
        self._curriculum_variant = ""
        self._oracle_cache: dict[
            int,
            tuple[
                Mapping[str, Any],
                int,
                float,
                tuple[float, float, float] | None,
            ],
        ] = {}
        self._grid_cache: dict[
            int,
            tuple[Mapping[str, Any], np.ndarray, np.ndarray],
        ] = {}
        self._oracle_search_target = -1
        self._oracle_search_phase = 0
        self._progress_maxima: dict[tuple[int, int, int], float] = {}
        self._initial_target_material = "unknown"
        self._episode_steps = 0
        self._focus_steps = 0
        self._attack_steps = 0
        self._valid_attack_steps = 0
        self._angular_error_sum = 0.0

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

    def set_guidance_override(self, enabled: bool | None) -> None:
        """Force evaluation guidance on/off without changing reset geometry."""
        if enabled is not None and not isinstance(enabled, bool):
            raise TypeError("guidance override must be bool or None")
        self._guidance_override = enabled

    def reset_payload(self, seed: int, options: Mapping[str, Any]) -> dict[str, Any]:
        java_stage, hint_probability, stage = self.curriculum_configuration()
        hint_rng = random.Random(seed ^ 0x4D43524C5F48494E)
        self._guidance_enabled = hint_rng.random() < hint_probability
        if self._guidance_override is not None:
            self._guidance_enabled = self._guidance_override
        self._curriculum_stage = stage
        return {"curriculum_stage": java_stage}

    def curriculum_configuration(self) -> tuple[int, float, str]:
        if not self._training_mode:
            return 12, 0.0, "full_free"
        stage, java_stage, hint_probability = self.CURRICULUM_STAGES[
            self._curriculum_stage_index
        ]
        return java_stage, hint_probability, stage

    def encode_action(self, action: Any) -> dict[str, Any]:
        action_array = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action_array):
            raise ValueError(f"invalid bed-defense action: {action!r}")
        self._last_yaw_degrees = float(
            self.YAW_DEGREES[int(action_array[0])]
        )
        self._last_pitch_degrees = float(
            self.PITCH_DEGREES[int(action_array[1])]
        )
        self._last_attack = bool(action_array[3])
        return {
            "mouse_dx": self._last_yaw_degrees / 0.15,
            "mouse_dy": -self._last_pitch_degrees / 0.15,
            "hotbar_slot": int(action_array[2]),
            "attack": self._last_attack,
        }

    def observation(self, raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
        grid, visible = self._grid_and_visibility(raw)
        bed_positions = self.bed_positions(raw)
        state: list[float] = []
        for position in bed_positions:
            state.extend(self._target_features(raw, position, bed_target=True))

        selected_slot = int(raw["player.selected_slot"])
        yaw_radians = math.radians(float(raw["player.yaw"]))
        guide_index = self.oracle_target_index(raw)
        raycast_position = self.raycast_position(raw)
        state.extend(
            [
                math.sin(yaw_radians),
                math.cos(yaw_radians),
                np.clip(float(raw["player.pitch"]) / 90.0, -1.0, 1.0),
                np.clip(float(raw["player.motion_y"]), -1.0, 1.0),
                float(bool(raw["player.on_ground"])),
                *[float(selected_slot == slot) for slot in range(3)],
                np.clip(float(raw["player.block_break_progress"]), 0.0, 1.0),
                self._progress_maxima.get(raycast_position, 0.0),
                float(self._last_attack),
            ]
        )

        raycast_category = self.raycast_category(raw)
        state.extend(
            [
                float(raycast_position is not None),
                *self._grid_relative_features(raw, raycast_position),
                *self._category_features(raycast_category),
                float(
                    self._guidance_enabled
                    and guide_index >= 0
                    and self.is_aimed_at_index(raw, guide_index)
                ),
            ]
        )

        if self._guidance_enabled and guide_index >= 0:
            guide_position = self.index_to_position(raw, guide_index)
            guide_aim_point = self.oracle_target_aim_point(raw)
            guide_features = [
                1.0,
                *self._target_features(
                    raw,
                    guide_position,
                    bed_target=False,
                    aim_point=guide_aim_point,
                )[:6],
                *self._category_features(self.category_at_index(raw, guide_index)),
                float(self.is_aimed_at_index(raw, guide_index)),
            ]
        else:
            guide_features = [0.0] * 14
        state.extend(guide_features)

        if len(state) != self.STATE_SIZE:
            raise RuntimeError(
                f"bed-defense state contract produced {len(state)} values, "
                f"expected {self.STATE_SIZE}"
            )

        grid_channels = np.zeros(
            (
                len(self.GRID_CHANNEL_NAMES),
                self.GRID_Y,
                self.GRID_Z,
                self.GRID_X,
            ),
            dtype=np.uint8,
        )
        shaped_grid = grid.reshape(self.GRID_Y, self.GRID_Z, self.GRID_X)
        shaped_visible = visible.reshape(self.GRID_Y, self.GRID_Z, self.GRID_X)
        grid_channels[0] = shaped_visible
        grid_channels[1] = shaped_visible & (shaped_grid != self.CATEGORY_AIR)
        for channel, category in enumerate(
            (
                self.CATEGORY_WOOL,
                self.CATEGORY_WOOD,
                self.CATEGORY_END_STONE,
                self.CATEGORY_OBSIDIAN,
            ),
            start=2,
        ):
            grid_channels[channel] = shaped_visible & (shaped_grid == category)
        for position in bed_positions:
            self._set_grid_channel(grid_channels[6], raw, position)
        if self._guidance_enabled and guide_index >= 0:
            y, z, x = self._index_coordinates(guide_index)
            grid_channels[7, y, z, x] = 1.0

        return {
            "state": np.asarray(state, dtype=np.float32),
            "grid": grid_channels,
        }

    def on_reset(self, initial: Mapping[str, Any]) -> None:
        if not bool(initial["world_reset_applied"]):
            raise RuntimeError("Java did not apply the bed_defense_break reset profile")
        if int(initial["task_blocks.count"]) != 2:
            raise RuntimeError("bed_defense_break requires exactly two bed halves")
        self._oracle_cache.clear()
        self._grid_cache.clear()
        grid, visible = self._grid_and_visibility(initial)
        if len(grid) != self.GRID_CELLS or len(visible) != self.GRID_CELLS:
            raise RuntimeError("bed defense grid has the wrong dimensions")
        if self.bed_present_count(initial) != 2:
            raise RuntimeError("bed_defense_break reset did not produce both bed halves")
        self._locked_bed_position = None
        self._initial_defense_blocks = self.defense_block_count(initial)
        self._tunnel_depth = (
            self._initial_defense_blocks // 2
            if self._curriculum_stage == "tunnel_guided"
            else 0
        )
        self._curriculum_variant = (
            f"depth_{self._tunnel_depth}" if self._tunnel_depth else ""
        )
        self._initial_route_cost = max(self.oracle_route_cost(initial), 1.0)
        if not math.isfinite(self._initial_route_cost):
            raise RuntimeError("bed defense reset has no reachable break route")
        initial_target = self.oracle_target_index(initial)
        self._initial_target_material = (
            self.CATEGORY_NAMES[self.category_at_index(initial, initial_target)]
            if initial_target >= 0
            else "unknown"
        )
        self._last_attack = False
        self._last_yaw_degrees = 0.0
        self._last_pitch_degrees = 0.0
        self._oracle_search_target = -1
        self._oracle_search_phase = 0
        self._progress_maxima.clear()
        self._episode_steps = 0
        self._focus_steps = 0
        self._attack_steps = 0
        self._valid_attack_steps = 0
        self._angular_error_sum = 0.0

    def reward(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        previous_beds = self.bed_present_count(previous)
        current_beds = self.bed_present_count(current)
        reward = -0.01

        previous_target = self.oracle_target_index(previous)
        target = self.oracle_target_index(current)
        angular_error = self.oracle_angular_error(current)
        if math.isfinite(angular_error):
            # This is a cost, never a bonus: looking farther from the selected
            # bed sightline can only make the return worse and cannot be farmed
            # by oscillating around the target.
            reward -= self.ANGULAR_ERROR_COST * min(angular_error / 90.0, 1.0)

        previous_focused = previous_target >= 0 and self.is_aimed_at_index(
            previous, previous_target
        )
        focused = target >= 0 and self.is_aimed_at_index(current, target)
        tool_correct = target >= 0 and self.target_tool_correct(current, target)
        previous_target_broken = self._route_target_was_broken(
            previous, previous_target, current
        )
        if not previous_target_broken:
            if int(current["player.selected_slot"]) != int(
                previous["player.selected_slot"]
            ):
                # Tool changes reset vanilla block-breaking continuity. Keep
                # the immediate correct-tool shaping, but make purposeless
                # cycling strictly costly and therefore non-farmable.
                reward -= self.TOOL_CHANGE_COST
            previous_angular_error = self.oracle_angular_error(previous)
            if (
                previous_focused
                and math.isfinite(previous_angular_error)
                and previous_angular_error <= self.AIM_STABILITY_DEGREES
            ):
                # This is strictly a cost: once the correct route is acquired,
                # needless camera motion cannot be rewarded or recovered by an
                # opposite action.  Necessary corrections outside the stable
                # aim region remain unpenalized.
                reward -= self.AIMED_CAMERA_MOTION_COST * (
                    abs(self._last_yaw_degrees)
                    + abs(self._last_pitch_degrees)
                )
            if previous_focused and not focused:
                # Losing a still-valid target is also a one-way cost.  There is
                # deliberately no matching reacquisition bonus to farm.
                reward -= self.FOCUS_LOSS_COST
            # A correct selection must be learnable from one action. Reversing
            # the selection or changing to a route requiring another tool
            # repays the same shaping. Only an irreversible block break can
            # bank it.
            reward += self.TOOL_SELECTION_REWARD * (
                self.tool_potential(current, target)
                - self.tool_potential(previous, previous_target)
            )
            # Breaking requires consecutive correct ticks. Signed progress
            # makes abandoning a partly damaged block repay its shaping even
            # when the oracle switches between the two bed sightlines. An
            # actual break is irreversible and is rewarded below.
            reward += self.PROGRESS_ACHIEVEMENT_REWARD * (
                self.progress_potential(current, target)
                - self.progress_potential(previous, previous_target)
            )

        if focused and tool_correct:
            position = self.index_to_position(current, target)
            progress = float(
                np.clip(current["player.block_break_progress"], 0.0, 1.0)
            )
            previous_maximum = self._progress_maxima.get(position, 0.0)
            if progress > previous_maximum:
                self._progress_maxima[position] = progress

        self._episode_steps += 1
        self._focus_steps += int(focused)
        self._attack_steps += int(self._last_attack)
        self._valid_attack_steps += int(
            self._last_attack and focused and tool_correct
        )
        if math.isfinite(angular_error):
            self._angular_error_sum += angular_error

        previous_cost = self.oracle_route_cost(previous)
        current_cost = self.oracle_route_cost(current)
        if math.isfinite(previous_cost) and math.isfinite(current_cost):
            # Route reductions are irreversible within an episode. Reward only
            # the reduction so switching between the two bed sightlines cannot
            # create a positive oscillation.
            reward += 3.0 * max(previous_cost - current_cost, 0.0) / (
                self._initial_route_cost
            )
        if previous_beds == 2 and current_beds < 2:
            reward += 20.0
        return reward

    def terminated(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        return self.bed_present_count(current) < 2

    def truncated(self, step_count: int, current: Mapping[str, Any]) -> bool:
        if not self._training_mode:
            return step_count >= self.max_episode_steps
        stage_limit = self.STAGE_STEP_LIMITS[self._curriculum_stage_index]
        return step_count >= min(self.max_episode_steps, stage_limit)

    def info(
        self, raw: Mapping[str, Any], *, episode_end: bool = True
    ) -> dict[str, Any]:
        info = super().info(raw, episode_end=episode_end)
        if not episode_end:
            return info
        current_defense = self.defense_block_count(raw)
        broken = max(0, self._initial_defense_blocks - current_defense)
        route_cost = self.oracle_route_cost(raw)
        success = self.bed_present_count(raw) < 2
        if success:
            completion = 1.0
        elif math.isfinite(route_cost):
            completion = float(
                np.clip(1.0 - route_cost / self._initial_route_cost, 0.0, 0.99)
            )
        else:
            completion = 0.0
        dropout_stage = 9 <= self._curriculum_stage_index <= 11
        tunnel_mastery_episode = (
            self._curriculum_stage != "tunnel_guided"
            or self._tunnel_depth == 3
        )
        info.update(
            {
                "success": success,
                "bed_halves_remaining": self.bed_present_count(raw),
                "initial_defense_blocks": self._initial_defense_blocks,
                "remaining_defense_blocks": current_defense,
                "broken_blocks": broken,
                "completion_fraction": completion,
                "oracle_route_cost": route_cost,
                "focus_fraction": self._fraction(self._focus_steps),
                "attack_fraction": self._fraction(self._attack_steps),
                "valid_attack_fraction": self._fraction(
                    self._valid_attack_steps
                ),
                "max_break_progress": max(
                    self._progress_maxima.values(), default=0.0
                ),
                "mean_angular_error_degrees": self._fraction(
                    self._angular_error_sum
                ),
                "initial_target_material": self._initial_target_material,
                "guidance_enabled": self._guidance_enabled,
                "curriculum_variant": self._curriculum_variant,
                "tunnel_depth": self._tunnel_depth,
                "curriculum_eligible": (
                    (not dropout_stage or not self._guidance_enabled)
                    and tunnel_mastery_episode
                ),
                "curriculum_stage": self._curriculum_stage,
                "curriculum_stage_index": self._curriculum_stage_index,
                "curriculum_steps": self._training_steps,
                "episode_limit": (
                    min(
                        self.max_episode_steps,
                        self.STAGE_STEP_LIMITS[self._curriculum_stage_index],
                    )
                    if self._training_mode
                    else self.max_episode_steps
                ),
            }
        )
        return info

    def bed_present_count(self, raw: Mapping[str, Any]) -> int:
        grid, _ = self._grid_and_visibility(raw)
        return sum(
            int(grid[self.position_to_index(raw, position)] == self.CATEGORY_BED)
            for position in self.bed_positions(raw)
        )

    def defense_block_count(self, raw: Mapping[str, Any]) -> int:
        grid, _ = self._grid_and_visibility(raw)
        return int(np.count_nonzero((grid >= self.CATEGORY_WOOL) & (grid <= self.CATEGORY_OBSIDIAN)))

    def oracle_target_index(self, raw: Mapping[str, Any]) -> int:
        return self._oracle_analysis(raw)[0]

    def oracle_route_cost(self, raw: Mapping[str, Any]) -> float:
        return self._oracle_analysis(raw)[1]

    def oracle_target_aim_point(
        self, raw: Mapping[str, Any]
    ) -> tuple[float, float, float] | None:
        return self._oracle_analysis(raw)[2]

    def _oracle_analysis(
        self, raw: Mapping[str, Any]
    ) -> tuple[int, float, tuple[float, float, float] | None]:
        cache_key = id(raw)
        cached = self._oracle_cache.get(cache_key)
        if cached is not None and cached[0] is raw:
            return cached[1], cached[2], cached[3]
        if self.bed_present_count(raw) < 2:
            target = -1
            route_cost = 0.0
            aim_point = None
        else:
            target, route_cost, aim_point = self._cheapest_sightline(raw)
        if len(self._oracle_cache) >= 4:
            # Keep the newest previous/current observations hot instead of
            # periodically discarding both and recomputing an unchanged route.
            self._oracle_cache.pop(next(iter(self._oracle_cache)))
        self._oracle_cache[cache_key] = (raw, target, route_cost, aim_point)
        return target, route_cost, aim_point

    def _cheapest_sightline(
        self, raw: Mapping[str, Any]
    ) -> tuple[int, float, tuple[float, float, float] | None]:
        """Choose a stable route, switching only for a materially better one."""
        grid, _ = self._grid_and_visibility(raw)
        routes: list[
            tuple[
                float,
                int,
                float,
                int,
                tuple[float, float, float],
                tuple[int, int, int],
            ]
        ] = []
        for position in self.bed_positions(raw):
            bed_index = self.position_to_index(raw, position)
            if int(grid[bed_index]) != self.CATEGORY_BED:
                continue
            path: list[int] = []
            valid = True
            reached_bed = False
            for index in self._sightline_indices(raw, position):
                category = int(grid[index])
                if category == self.CATEGORY_AIR:
                    continue
                if category not in self.BREAK_COST:
                    valid = False
                    break
                path.append(index)
                if category == self.CATEGORY_BED:
                    reached_bed = True
                    break
            if not valid or not reached_bed or not path:
                continue
            target = path[0]
            if not self._within_reach(raw, target):
                continue
            cost = sum(self.BREAK_COST[int(grid[index])] for index in path)
            # Trace routes toward the bed, but guide actions safely inside the
            # current blocker. Aiming through a blocker at the bed itself put
            # the cursor near hitbox edges on deep defenses.
            route_aim_point = self.target_aim_point(raw, target)
            yaw_delta, pitch_delta = self._angle_deltas_to_point(
                raw, route_aim_point
            )
            routes.append(
                (
                    cost,
                    len(path),
                    math.hypot(yaw_delta, pitch_delta),
                    target,
                    route_aim_point,
                    position,
                )
            )
        if not routes:
            return -1, math.inf, None
        best = min(routes)
        locked = next(
            (
                route
                for route in routes
                if route[5] == self._locked_bed_position
            ),
            None,
        )
        chosen = best
        if locked is not None:
            chosen = locked
            if best[5] != locked[5]:
                locked_target = locked[3]
                best_target = best[3]
                locked_progress_active = (
                    float(raw["player.block_break_progress"]) > 0.0
                    and self.raycast_position(raw)
                    == self.index_to_position(raw, locked_target)
                )
                best_reaches_bed = (
                    self.category_at_index(raw, best_target) == self.CATEGORY_BED
                )
                locked_reaches_bed = (
                    self.category_at_index(raw, locked_target) == self.CATEGORY_BED
                )
                if not locked_progress_active and (
                    (best_reaches_bed and not locked_reaches_bed)
                    or best[0] + self.ROUTE_SWITCH_MARGIN < locked[0]
                ):
                    chosen = best
        cost, _, _, target, aim_point, bed_position = chosen
        self._locked_bed_position = bed_position
        return target, cost, aim_point

    def _sightline_indices(
        self, raw: Mapping[str, Any], destination: tuple[int, int, int]
    ) -> tuple[int, ...]:
        """Traverse every voxel touched from the eye to the hittable bed aim point."""
        start = (
            float(raw["player.x"]),
            float(raw["player.y"]) + self.EYE_HEIGHT,
            float(raw["player.z"]),
        )
        end = self.target_aim_point(raw, self.position_to_index(raw, destination))
        delta = tuple(end[axis] - start[axis] for axis in range(3))
        current = [math.floor(coordinate) for coordinate in start]
        end_cell = list(destination)
        step = [1 if value > 0.0 else -1 if value < 0.0 else 0 for value in delta]
        t_delta = [math.inf if value == 0.0 else abs(1.0 / value) for value in delta]
        t_max: list[float] = []
        for axis, value in enumerate(delta):
            if value > 0.0:
                t_max.append((current[axis] + 1.0 - start[axis]) / value)
            elif value < 0.0:
                t_max.append((start[axis] - current[axis]) / -value)
            else:
                t_max.append(math.inf)

        indices: list[int] = []

        def append_current() -> None:
            try:
                index = self.position_to_index(raw, tuple(current))
            except ValueError:
                return
            if not indices or indices[-1] != index:
                indices.append(index)

        append_current()
        for _ in range(self.GRID_X + self.GRID_Y + self.GRID_Z + 3):
            if current == end_cell:
                break
            next_t = min(t_max)
            if next_t > 1.0 + 1e-12:
                break
            for axis in range(3):
                if abs(t_max[axis] - next_t) <= 1e-12:
                    current[axis] += step[axis]
                    t_max[axis] += t_delta[axis]
            append_current()
        return tuple(indices)

    def _within_reach(self, raw: Mapping[str, Any], index: int) -> bool:
        """Measure reach to the nearest hittable point, not the farther block center."""
        x, y, z = self.index_to_position(raw, index)
        eye = (
            float(raw["player.x"]),
            float(raw["player.y"]) + self.EYE_HEIGHT,
            float(raw["player.z"]),
        )
        top = y + (
            self.BED_HEIGHT
            if self.category_at_index(raw, index) == self.CATEGORY_BED
            else 1.0
        )
        nearest = (
            min(max(eye[0], x), x + 1.0),
            min(max(eye[1], y), top),
            min(max(eye[2], z), z + 1.0),
        )
        return math.dist(eye, nearest) <= self.REACH_DISTANCE + 1e-9

    def oracle_angular_error(self, raw: Mapping[str, Any]) -> float:
        aim_point = self.oracle_target_aim_point(raw)
        if aim_point is None:
            return math.inf
        yaw_delta, pitch_delta = self._angle_deltas_to_point(raw, aim_point)
        return math.hypot(yaw_delta, pitch_delta)

    def target_tool_correct(self, raw: Mapping[str, Any], target: int) -> bool:
        preferred_slot = self.correct_slot(self.category_at_index(raw, target))
        return (
            preferred_slot is None
            or int(raw["player.selected_slot"]) == preferred_slot
        )

    def tool_potential(self, raw: Mapping[str, Any], target: int) -> float:
        if target < 0:
            return 0.0
        preferred_slot = self.correct_slot(self.category_at_index(raw, target))
        if preferred_slot is None:
            return 0.0
        return float(int(raw["player.selected_slot"]) == preferred_slot)

    def progress_potential(self, raw: Mapping[str, Any], target: int) -> float:
        if target < 0 or not self.is_aimed_at_index(raw, target):
            return 0.0
        if not self.target_tool_correct(raw, target):
            return 0.0
        return float(np.clip(raw["player.block_break_progress"], 0.0, 1.0))

    def _route_target_was_broken(
        self,
        previous: Mapping[str, Any],
        previous_target: int,
        current: Mapping[str, Any],
    ) -> bool:
        if previous_target < 0:
            return False
        previous_position = self.index_to_position(previous, previous_target)
        try:
            current_index = self.position_to_index(current, previous_position)
        except ValueError:
            return False
        return self.category_at_index(current, current_index) == self.CATEGORY_AIR

    def angular_error(self, raw: Mapping[str, Any], index: int) -> float:
        yaw_delta, pitch_delta = self.target_angle_deltas(raw, index)
        return math.hypot(yaw_delta, pitch_delta)

    def target_angle_deltas(
        self, raw: Mapping[str, Any], index: int
    ) -> tuple[float, float]:
        return self._angle_deltas_to_point(raw, self.target_aim_point(raw, index))

    def _angle_deltas_to_point(
        self, raw: Mapping[str, Any], point: tuple[float, float, float]
    ) -> tuple[float, float]:
        target_x, target_y, target_z = point
        dx = target_x - float(raw["player.x"])
        dy = target_y - (float(raw["player.y"]) + self.EYE_HEIGHT)
        dz = target_z - float(raw["player.z"])
        horizontal = math.hypot(dx, dz)
        desired_yaw = math.degrees(math.atan2(-dx, dz))
        desired_pitch = -math.degrees(math.atan2(dy, horizontal))
        yaw_delta = (desired_yaw - float(raw["player.yaw"]) + 180.0) % 360.0 - 180.0
        pitch_delta = desired_pitch - float(raw["player.pitch"])
        return yaw_delta, pitch_delta

    def target_aim_point(
        self, raw: Mapping[str, Any], index: int
    ) -> tuple[float, float, float]:
        x, y, z = self.index_to_position(raw, index)
        if self.category_at_index(raw, index) != self.CATEGORY_BED:
            eye = (
                float(raw["player.x"]),
                float(raw["player.y"]) + self.EYE_HEIGHT,
                float(raw["player.z"]),
            )
            return self._reachable_interior_aim_point(
                eye,
                (float(x), float(y), float(z)),
                (float(x + 1), float(y + 1), float(z + 1)),
                (self.BLOCK_AIM_INSET,) * 3,
            )

        return self._bed_aim_point(raw, (x, y, z))

    def _bed_aim_point(
        self, raw: Mapping[str, Any], position: tuple[int, int, int]
    ) -> tuple[float, float, float]:
        x, y, z = position

        eye = (
            float(raw["player.x"]),
            float(raw["player.y"]) + self.EYE_HEIGHT,
            float(raw["player.z"]),
        )
        x_inset = self.BED_X_AIM_INSET
        z_inset = self.BED_Z_AIM_INSET
        vertical_inset = self.BED_VERTICAL_AIM_INSET
        return self._reachable_interior_aim_point(
            eye,
            (float(x), float(y), float(z)),
            (float(x + 1), float(y) + self.BED_HEIGHT, float(z + 1)),
            (x_inset, vertical_inset, z_inset),
        )

    def _reachable_interior_aim_point(
        self,
        eye: tuple[float, float, float],
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        safe_insets: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Use the safest hitbox point that remains inside survival reach."""

        def inset_point(insets: tuple[float, float, float]) -> tuple[float, ...]:
            return tuple(
                min(max(eye[axis], lower[axis] + insets[axis]),
                    upper[axis] - insets[axis])
                for axis in range(3)
            )

        safe = inset_point(safe_insets)
        if math.dist(eye, safe) <= self.AIM_REACH_DISTANCE:
            return safe

        nearest = inset_point((self.MIN_AIM_INSET,) * 3)
        if math.dist(eye, nearest) >= self.AIM_REACH_DISTANCE:
            return nearest

        low = 0.0
        high = 1.0
        for _ in range(32):
            amount = (low + high) * 0.5
            candidate = tuple(
                nearest[axis] + amount * (safe[axis] - nearest[axis])
                for axis in range(3)
            )
            if math.dist(eye, candidate) <= self.AIM_REACH_DISTANCE:
                low = amount
            else:
                high = amount
        return tuple(
            nearest[axis] + low * (safe[axis] - nearest[axis])
            for axis in range(3)
        )

    def is_aimed_at_index(self, raw: Mapping[str, Any], index: int) -> bool:
        if raw["raycast.type"] != "BLOCK":
            return False
        return (
            int(raw["raycast.x"]),
            int(raw["raycast.y"]),
            int(raw["raycast.z"]),
        ) == self.index_to_position(raw, index)

    def raycast_category(self, raw: Mapping[str, Any]) -> int:
        if raw["raycast.type"] != "BLOCK":
            return self.CATEGORY_UNKNOWN
        position = (
            int(raw["raycast.x"]),
            int(raw["raycast.y"]),
            int(raw["raycast.z"]),
        )
        try:
            return self.category_at_index(raw, self.position_to_index(raw, position))
        except ValueError:
            return self.CATEGORY_OTHER

    def raycast_position(
        self, raw: Mapping[str, Any]
    ) -> tuple[int, int, int] | None:
        if raw["raycast.type"] != "BLOCK":
            return None
        return (
            int(raw["raycast.x"]),
            int(raw["raycast.y"]),
            int(raw["raycast.z"]),
        )

    def _target_features(
        self,
        raw: Mapping[str, Any],
        position: tuple[int, int, int],
        *,
        bed_target: bool,
        aim_point: tuple[float, float, float] | None = None,
    ) -> list[float]:
        player = (
            float(raw["player.x"]),
            float(raw["player.y"]),
            float(raw["player.z"]),
        )
        center = tuple(float(coordinate) + 0.5 for coordinate in position)
        if aim_point is None:
            aim_point = (
                self._bed_aim_point(raw, position)
                if bed_target
                else self.target_aim_point(
                    raw, self.position_to_index(raw, position)
                )
            )
        yaw_delta, pitch_delta = self._angle_deltas_to_point(raw, aim_point)
        eye = (player[0], player[1] + self.EYE_HEIGHT, player[2])
        try:
            category = self.category_at_index(
                raw, self.position_to_index(raw, position)
            )
            present = float(
                category == self.CATEGORY_BED
                if bed_target
                else category != self.CATEGORY_AIR
            )
        except ValueError:
            present = 0.0
        return [
            *[
                float(np.clip((center[axis] - player[axis]) / 5.0, -1.0, 1.0))
                for axis in range(3)
            ],
            float(np.clip(yaw_delta / 180.0, -1.0, 1.0)),
            float(np.clip(pitch_delta / 90.0, -1.0, 1.0)),
            float(np.clip(math.dist(eye, aim_point) / 5.0, 0.0, 1.0)),
            present,
        ]

    def _grid_relative_features(
        self,
        raw: Mapping[str, Any],
        position: tuple[int, int, int] | None,
    ) -> tuple[float, float, float]:
        if position is None:
            return 0.0, 0.0, 0.0
        foot = self.bed_positions(raw)[0]
        return tuple(
            float(np.clip((position[axis] - foot[axis]) / 3.0, -1.0, 1.0))
            for axis in range(3)
        )

    def _category_features(self, category: int) -> tuple[float, ...]:
        return tuple(
            float(category == candidate)
            for candidate in self.MATERIAL_FEATURE_CATEGORIES
        )

    @staticmethod
    def _index_coordinates(index: int) -> tuple[int, int, int]:
        x = index % BedDefenseBreakTask.GRID_X
        yz = index // BedDefenseBreakTask.GRID_X
        z = yz % BedDefenseBreakTask.GRID_Z
        y = yz // BedDefenseBreakTask.GRID_Z
        return y, z, x

    def _set_grid_channel(
        self,
        channel: np.ndarray,
        raw: Mapping[str, Any],
        position: tuple[int, int, int],
    ) -> None:
        try:
            index = self.position_to_index(raw, position)
        except ValueError:
            return
        y, z, x = self._index_coordinates(index)
        channel[y, z, x] = 1.0

    def _fraction(self, numerator: float) -> float:
        return numerator / self._episode_steps if self._episode_steps else 0.0

    def category_at_index(self, raw: Mapping[str, Any], index: int) -> int:
        grid, _ = self._grid_and_visibility(raw)
        return int(grid[index])

    @classmethod
    def correct_slot(cls, category: int) -> int | None:
        if category == cls.CATEGORY_WOOL:
            return 0
        if category == cls.CATEGORY_WOOD:
            return 1
        if category in (cls.CATEGORY_END_STONE, cls.CATEGORY_OBSIDIAN):
            return 2
        return None

    def bed_positions(self, raw: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            tuple(math.floor(float(raw[self.key(index, axis)])) for axis in ("x", "y", "z"))
            for index in range(2)
        )

    def position_to_index(
        self, raw: Mapping[str, Any], position: tuple[int, int, int]
    ) -> int:
        origin = self.grid_origin(raw)
        x = position[0] - origin[0]
        y = position[1] - origin[1]
        z = position[2] - origin[2]
        if not (0 <= x < self.GRID_X and 0 <= y < self.GRID_Y and 0 <= z < self.GRID_Z):
            raise ValueError(f"position {position} lies outside bed defense grid")
        return (y * self.GRID_Z + z) * self.GRID_X + x

    def index_to_position(self, raw: Mapping[str, Any], index: int) -> tuple[int, int, int]:
        if not 0 <= index < self.GRID_CELLS:
            raise ValueError("bed defense grid index out of range")
        x = index % self.GRID_X
        yz = index // self.GRID_X
        z = yz % self.GRID_Z
        y = yz // self.GRID_Z
        origin = self.grid_origin(raw)
        return origin[0] + x, origin[1] + y, origin[2] + z

    @staticmethod
    def grid_origin(raw: Mapping[str, Any]) -> tuple[int, int, int]:
        return (
            int(raw["bed_defense.origin_x"]),
            int(raw["bed_defense.origin_y"]),
            int(raw["bed_defense.origin_z"]),
        )

    def _grid_and_visibility(
        self, raw: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = id(raw)
        cached = self._grid_cache.get(cache_key)
        if cached is not None and cached[0] is raw:
            return cached[1], cached[2]

        grid = np.asarray(raw["bed_defense.block_grid"], dtype=np.int8)
        visible = np.asarray(raw["bed_defense.visible_mask"], dtype=np.int8).astype(bool)
        if grid.shape != (self.GRID_CELLS,) or visible.shape != (self.GRID_CELLS,):
            raise ValueError("bed defense grid must contain exactly 245 cells")
        if np.any((grid < self.CATEGORY_AIR) | (grid > self.CATEGORY_OTHER)):
            raise ValueError("bed defense grid contains an unknown category")
        # Raw observations are immutable after protocol validation. Cache the
        # two conversions because reward/observation/info query the same raw
        # object dozens of times during one environment step.
        grid.setflags(write=False)
        visible.setflags(write=False)
        if len(self._grid_cache) >= 4:
            self._grid_cache.pop(next(iter(self._grid_cache)))
        self._grid_cache[cache_key] = (raw, grid, visible)
        return grid, visible

    def oracle_action(self, raw: Mapping[str, Any]) -> np.ndarray:
        target = self.oracle_target_index(raw)
        if target < 0:
            return np.asarray([5, 5, 2, 0], dtype=np.int64)
        if target != self._oracle_search_target:
            self._oracle_search_target = target
            self._oracle_search_phase = 0
        aimed = self.is_aimed_at_index(raw, target)
        aim_point = self.oracle_target_aim_point(raw)
        if aim_point is None:
            return np.asarray([5, 5, 2, 0], dtype=np.int64)
        yaw_delta, pitch_delta = self._angle_deltas_to_point(raw, aim_point)
        yaw_index = int(np.argmin(np.abs(self.YAW_DEGREES - (0.0 if aimed else yaw_delta))))
        pitch_index = int(
            np.argmin(np.abs(self.PITCH_DEGREES - (0.0 if aimed else pitch_delta)))
        )
        if not aimed and yaw_index == 5 and pitch_index == 5:
            search_actions = ((6, 5), (4, 5), (5, 6), (5, 4))
            yaw_index, pitch_index = search_actions[
                self._oracle_search_phase % len(search_actions)
            ]
            self._oracle_search_phase += 1
        preferred_slot = self.correct_slot(self.category_at_index(raw, target))
        return np.asarray(
            [
                yaw_index,
                pitch_index,
                (
                    int(raw["player.selected_slot"])
                    if preferred_slot is None
                    else preferred_slot
                ),
                int(aimed),
            ],
            dtype=np.int64,
        )

    def describe_action(self, action: Any) -> str:
        action_array = np.asarray(action, dtype=np.int64)
        return (
            f"yaw={self.YAW_DEGREES[int(action_array[0])]:+g}deg "
            f"pitch={self.PITCH_DEGREES[int(action_array[1])]:+g}deg "
            f"tool={self.TOOL_NAMES[int(action_array[2])]} "
            f"attack={bool(action_array[3])}"
        )

    def describe_observation(self, raw: Mapping[str, Any]) -> str:
        """Compact oracle/raycast diagnostics for verbose visual evaluation."""
        target = self.oracle_target_index(raw)
        target_position = (
            self.index_to_position(raw, target) if target >= 0 else None
        )
        target_category = (
            self.CATEGORY_NAMES[self.category_at_index(raw, target)]
            if target >= 0
            else "none"
        )
        raycast_position = self.raycast_position(raw)
        raycast_category = self.CATEGORY_NAMES[self.raycast_category(raw)]
        angle = self.oracle_angular_error(raw)
        return (
            f"target={target_position}:{target_category} "
            f"raycast={raycast_position}:{raycast_category} "
            f"tool={self.TOOL_NAMES[int(raw['player.selected_slot'])]} "
            f"progress={float(raw['player.block_break_progress']):.3f} "
            f"angle={angle:.2f}deg"
        )

    def describe_reset(self, raw: Mapping[str, Any]) -> str:
        grid, _ = self._grid_and_visibility(raw)
        materials = {
            self.CATEGORY_NAMES[category]: int(np.count_nonzero(grid == category))
            for category in self.BREAK_COST
            if category != self.CATEGORY_BED and np.any(grid == category)
        }
        return (
            f"stage={self._curriculum_stage} guidance={self._guidance_enabled} "
            f"variant={self._curriculum_variant or 'default'} "
            f"player=({float(raw['player.x']):.2f},"
            f"{float(raw['player.y']):.2f},"
            f"{float(raw['player.z']):.2f}) "
            f"bed_halves={self.bed_positions(raw)} materials={materials}"
        )

    @staticmethod
    def key(index: int, attribute: str) -> str:
        return f"task_block_{index}.{attribute}"
