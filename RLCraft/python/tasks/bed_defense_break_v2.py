"""V2 task: efficiently expose, aim at, and break one BedWars bed half."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from minecraft_env import MinecraftTask
from tasks.aim_tool_break import AimToolBreakTask


class BedDefenseBreakV2Task(MinecraftTask):
    """Seven-stage BedWars curriculum with no oracle or route guidance."""

    GRID_LONGITUDINAL = 8
    GRID_VERTICAL = 4
    GRID_LATERAL = 7
    GRID_CELLS = GRID_LONGITUDINAL * GRID_VERTICAL * GRID_LATERAL

    # Raw Java categories. Unknown exists only in the model-facing one-hot grid.
    CATEGORY_AIR = 1
    CATEGORY_BED = 2
    CATEGORY_WOOL = 3
    CATEGORY_WOOD = 4
    CATEGORY_END_STONE = 5
    CATEGORY_OBSIDIAN = 6
    CATEGORY_OTHER = 7
    CATEGORY_NAMES = (
        "air",
        "bed",
        "wool",
        "wood",
        "end_stone",
        "obsidian",
        "other_solid",
        "unknown",
    )
    LOOK_CATEGORY_NAMES = (
        "nothing",
        "bed",
        "wool",
        "wood",
        "end_stone",
        "obsidian",
        "other_solid",
    )
    DEFENSE_CATEGORIES = frozenset(
        (CATEGORY_WOOL, CATEGORY_WOOD, CATEGORY_END_STONE, CATEGORY_OBSIDIAN)
    )

    CURRICULUM_STAGES = (
        ("exposed_fixed_aimed", 0),
        ("exposed_random", 1),
        ("one_layer_fixed_aimed", 2),
        ("one_layer_random", 3),
        ("two_layers_random", 4),
        ("obsidian_inner_random", 5),
        ("full_random", 6),
    )
    # Correct-tool worst cases are tiny except obsidian (~188 attack ticks with
    # a diamond pickaxe). The horizons leave time to acquire each target but no
    # longer allow hundreds of irrelevant actions after an obvious failure.
    STAGE_STEP_LIMITS = (30, 60, 60, 100, 140, 320, 320)

    # Keep the useful coarse-to-fine relative controls from aim_tool_break, but
    # omit its emergency full-screen jumps. Those extremes made a randomly
    # initialized policy throw away the aimed reset before its attack action
    # could produce any progress signal.
    YAW_DEGREES = tuple(
        value for value in AimToolBreakTask.YAW_DEGREES if abs(value) < 90.0
    )
    PITCH_DEGREES = tuple(
        value for value in AimToolBreakTask.PITCH_DEGREES if abs(value) < 45.0
    )
    # Model order follows the V2 specification. Java hotbar order remains
    # shears, axe, pickaxe for compatibility with the existing reset code.
    TOOL_NAMES = ("diamond_pickaxe", "diamond_axe", "shears")
    TOOL_HOTBAR_SLOTS = (2, 1, 0)
    STATE_SIZE = 24
    BED_HEIGHT = 0.5625
    EYE_HEIGHT = 1.62

    STEP_PENALTY = 0.01
    BED_BREAK_REWARD = 20.0
    # Keep wrong-tool attacks worse than idling without making early material
    # exploration catastrophically expensive. Correct-tool progress remains
    # the positive signal that teaches the material/tool mapping.
    WRONG_TOOL_ATTACK_PENALTY = 0.04
    PROGRESS_SCALE = 5.0
    # Progress is capped per logical shell, so an extra block cannot earn a
    # second progress budget. Keep only a small route-efficiency cost so a
    # mistaken first block does not make recovering and reaching the bed look
    # worse than giving up.
    EXTRA_BLOCK_PENALTY = 0.5
    IDLE_CAMERA_JITTER_DEGREES = 1.0
    PARTIAL_DEFENSE_REHEARSAL_LIMIT = 80
    GAMMA = 0.99

    def __init__(self, max_episode_steps: int = 320) -> None:
        bed_fields = tuple(
            f"task_block_{index}.{axis}"
            for index in range(2)
            for axis in ("x", "y", "z")
        )
        state_low = np.asarray(
            [0.0] * 3
            + [-8.0, -8.0, -8.0, -4.0, 0.0, -180.0, -1.0, -1.0]
            + [0.0] * 7
            + [0.0, -16.0, -16.0, -16.0, -180.0, -180.0],
            dtype=np.float32,
        )
        state_high = np.asarray(
            [1.0] * 3
            + [8.0, 8.0, 8.0, 4.0, 1.0, 180.0, 1.0, 1.0]
            + [1.0] * 7
            + [1.0, 16.0, 16.0, 16.0, 180.0, 180.0],
            dtype=np.float32,
        )
        super().__init__(
            task_id="bed_defense_break_v2",
            action_space=spaces.MultiDiscrete(
                [len(self.YAW_DEGREES), len(self.PITCH_DEGREES), 3, 2]
            ),
            observation_space=spaces.Dict(
                {
                    "state": spaces.Box(
                        low=state_low, high=state_high, dtype=np.float32
                    ),
                    "grid": spaces.Box(
                        low=0,
                        high=1,
                        shape=(
                            len(self.CATEGORY_NAMES),
                            self.GRID_VERTICAL,
                            self.GRID_LATERAL,
                            self.GRID_LONGITUDINAL,
                        ),
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
                "player.motion_y",
                "player.on_ground",
                "player.selected_slot",
                "player.block_break_progress",
                "raycast.type",
                "raycast.category",
                "raycast.x",
                "raycast.y",
                "raycast.z",
                "raycast.safe_yaw_correction",
                "raycast.safe_pitch_correction",
                "bed_defense_v2.block_grid",
                "bed_defense_v2.visible_mask",
            ),
            max_episode_steps=max_episode_steps,
        )
        self._training_mode = False
        self._training_steps = 0
        self._curriculum_stage_index = 0
        self._active_stage_index = len(self.CURRICULUM_STAGES) - 1
        self._curriculum_stage = self.CURRICULUM_STAGES[-1][0]
        self._last_attack = False
        self._last_action_tool = 0
        self._observed_selected_slot = self.TOOL_HOTBAR_SLOTS[0]
        self._initial_layers = 0
        self._initial_layer_materials = ""
        self._initial_selected_tool = ""
        self._partial_defense_rehearsal = False
        self._non_bed_block_budget = 0
        self._non_bed_blocks_broken = 0
        self._wrong_tool_attack_steps = 0
        self._correct_tool_attack_steps = 0
        self._attack_command_steps = 0
        self._attack_steps = 0
        self._air_attack_steps = 0
        self._bed_attack_steps = 0
        self._defense_attack_steps = 0
        self._other_attack_steps = 0
        self._bed_focus_steps = 0
        self._breakable_focus_steps = 0
        self._camera_noop_steps = 0
        self._absolute_yaw_delta_total = 0.0
        self._absolute_pitch_delta_total = 0.0
        self._tool_switch_steps = 0
        self._tool_selection_steps = [0, 0, 0]
        self._episode_steps = 0
        self._max_break_progress = 0.0
        self._max_observed_break_progress = 0.0
        self._last_yaw_delta = 0.0
        self._last_pitch_delta = 0.0
        self._idle_jitter_rng = np.random.default_rng(0)
        # One non-repeatable progress budget is shared by each defense shell and
        # one by the bed. A brief correct hit therefore keeps its learning
        # signal, while switching targets, letting damage reset, or scraping
        # every block in one shell can never farm the same progress twice.
        self._progress_high_water: dict[int, float] = {}
        self._progress_reward_total = 0.0
        self._wrong_tool_penalty_total = 0.0
        self._excess_block_penalty_total = 0.0
        self._grid_cache: dict[int, tuple[Mapping[str, Any], np.ndarray, np.ndarray]] = {}

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

    def curriculum_configuration(self) -> tuple[int, str]:
        if not self._training_mode:
            return 6, self.CURRICULUM_STAGES[-1][0]
        name, java_stage = self.CURRICULUM_STAGES[self._curriculum_stage_index]
        return java_stage, name

    def reset_payload(self, seed: int, options: Mapping[str, Any]) -> dict[str, Any]:
        java_stage, name = self.curriculum_configuration()
        self._curriculum_stage = name
        self._active_stage_index = java_stage
        self._idle_jitter_rng = np.random.default_rng(
            int(seed) & ((1 << 64) - 1)
        )
        return {"curriculum_stage": java_stage}

    def encode_action(self, action: Any) -> dict[str, Any]:
        action_array = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action_array):
            raise ValueError(f"invalid bed-defense V2 action: {action!r}")
        self._last_action_tool = int(action_array[2])
        self._last_attack = bool(action_array[3])
        self._last_yaw_delta = float(self.YAW_DEGREES[int(action_array[0])])
        self._last_pitch_delta = float(self.PITCH_DEGREES[int(action_array[1])])
        requested_slot = self.TOOL_HOTBAR_SLOTS[self._last_action_tool]
        # Beds accept every tool, but changing slots resets vanilla block
        # damage. During exposed-bed curriculum stages the tool choice carries
        # no task information, so preserve the randomized current tool while
        # the trainer keeps this unused action head neutral. Material stages
        # release the requested absolute tool normally.
        effective_slot = (
            self._observed_selected_slot
            if self._training_mode and self._active_stage_index <= 1
            else requested_slot
        )
        # A deterministic policy can otherwise enter an absorbing state where
        # it selects the current tool, releases attack, and repeats a zero
        # camera action forever. Only perturb a genuinely effect-free action:
        # attacking, switching tools, or requesting camera motion is preserved
        # exactly. The per-reset RNG keeps seeded runs reproducible.
        if (
            self._last_yaw_delta == 0.0
            and self._last_pitch_delta == 0.0
            and not self._last_attack
            and effective_slot == self._observed_selected_slot
        ):
            self._last_yaw_delta = float(
                self._idle_jitter_rng.uniform(
                    -self.IDLE_CAMERA_JITTER_DEGREES,
                    self.IDLE_CAMERA_JITTER_DEGREES,
                )
            )
            self._last_pitch_delta = float(
                self._idle_jitter_rng.uniform(
                    -self.IDLE_CAMERA_JITTER_DEGREES,
                    self.IDLE_CAMERA_JITTER_DEGREES,
                )
            )
        return {
            "mouse_dx": self._last_yaw_delta / 0.15,
            "mouse_dy": -self._last_pitch_delta / 0.15,
            "hotbar_slot": effective_slot,
            "attack": self._last_attack,
        }

    def observation(self, raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
        selected_slot = int(raw["player.selected_slot"])
        self._observed_selected_slot = selected_slot
        player_longitudinal, player_lateral, player_vertical = (
            self._player_relative_coordinates(raw)
        )
        yaw_delta, pitch_delta = self._angles_to_bed_center(raw)
        looked_category = self._look_category(raw)
        looked_coordinates = self._looked_relative_coordinates(raw)
        state = np.asarray(
            [
                *[float(selected_slot == slot) for slot in self.TOOL_HOTBAR_SLOTS],
                player_longitudinal,
                player_lateral,
                player_vertical,
                float(np.clip(raw["player.motion_y"], -4.0, 4.0)),
                float(bool(raw["player.on_ground"])),
                float(np.clip(pitch_delta, -180.0, 180.0)),
                math.sin(math.radians(yaw_delta)),
                math.cos(math.radians(yaw_delta)),
                *[
                    float(looked_category == category)
                    for category in range(len(self.LOOK_CATEGORY_NAMES))
                ],
                float(np.clip(raw["player.block_break_progress"], 0.0, 1.0))
                if looked_category != 0
                else 0.0,
                *looked_coordinates,
                float(np.clip(raw["raycast.safe_yaw_correction"], -180.0, 180.0))
                if looked_category != 0
                else 0.0,
                float(np.clip(raw["raycast.safe_pitch_correction"], -180.0, 180.0))
                if looked_category != 0
                else 0.0,
            ],
            dtype=np.float32,
        )
        if state.shape != (self.STATE_SIZE,):
            raise RuntimeError(
                f"bed-defense V2 state produced {state.size} values, "
                f"expected {self.STATE_SIZE}"
            )

        categories, visible = self._grid_and_visibility(raw)
        channels = np.zeros(
            (
                len(self.CATEGORY_NAMES),
                self.GRID_VERTICAL,
                self.GRID_LATERAL,
                self.GRID_LONGITUDINAL,
            ),
            dtype=np.uint8,
        )
        for index, category in enumerate(categories):
            longitudinal, vertical, lateral = self._index_coordinates(index)
            channel = int(category) - 1 if visible[index] or category == self.CATEGORY_BED else 7
            channels[channel, vertical, lateral, longitudinal] = 1
        return {"state": state, "grid": channels}

    def on_reset(self, initial: Mapping[str, Any]) -> None:
        if not bool(initial["world_reset_applied"]):
            raise RuntimeError("Java did not apply the bed_defense_break_v2 reset profile")
        if int(initial["task_blocks.count"]) != 2:
            raise RuntimeError("bed_defense_break_v2 requires exactly two bed halves")
        self._grid_cache.clear()
        categories, visible = self._grid_and_visibility(initial)
        if categories.shape != (self.GRID_CELLS,) or visible.shape != (self.GRID_CELLS,):
            raise RuntimeError("bed-defense V2 grid has the wrong dimensions")
        if self.bed_present_count(initial) != 2:
            raise RuntimeError("bed-defense V2 reset did not produce both bed halves")
        self._initial_layers = self._defense_layer_count(categories)
        self._initial_layer_materials = self._layer_material_names(categories)
        bed_indices = (
            self._grid_index(3, 0, 3),
            self._grid_index(4, 0, 3),
        )
        self._partial_defense_rehearsal = (
            self._active_stage_index >= 3
            and self._initial_layers > 0
            and any(bool(visible[index]) for index in bed_indices)
        )
        self._non_bed_block_budget = (
            0 if self._partial_defense_rehearsal else self._initial_layers
        )
        selected_slot = int(initial["player.selected_slot"])
        self._observed_selected_slot = selected_slot
        self._initial_selected_tool = self._tool_name_for_slot(selected_slot)
        self._non_bed_blocks_broken = 0
        self._wrong_tool_attack_steps = 0
        self._correct_tool_attack_steps = 0
        self._attack_command_steps = 0
        self._attack_steps = 0
        self._air_attack_steps = 0
        self._bed_attack_steps = 0
        self._defense_attack_steps = 0
        self._other_attack_steps = 0
        self._bed_focus_steps = 0
        self._breakable_focus_steps = 0
        self._camera_noop_steps = 0
        self._absolute_yaw_delta_total = 0.0
        self._absolute_pitch_delta_total = 0.0
        self._tool_switch_steps = 0
        self._tool_selection_steps = [0, 0, 0]
        self._episode_steps = 0
        self._max_break_progress = 0.0
        self._max_observed_break_progress = 0.0
        self._progress_high_water.clear()
        if self._partial_defense_rehearsal:
            # The bed is already reachable. Consume defense-shell shaping so
            # mining around the opening cannot compete with breaking the bed.
            for index, category in enumerate(categories):
                progress_group = self._progress_group(index, int(category))
                if progress_group is not None and progress_group > 0:
                    self._progress_high_water[progress_group] = 1.0
        initial_target = self._raycast_index(initial)
        if initial_target is not None:
            initial_category = int(categories[initial_target])
            progress_group = self._progress_group(initial_target, initial_category)
            if progress_group is not None:
                self._progress_high_water[progress_group] = max(
                    self._progress_high_water.get(progress_group, 0.0),
                    float(
                        np.clip(
                            initial["player.block_break_progress"], 0.0, 1.0
                        )
                    ),
                )
        self._progress_reward_total = 0.0
        self._wrong_tool_penalty_total = 0.0
        self._excess_block_penalty_total = 0.0
        self._last_attack = False
        self._last_yaw_delta = 0.0
        self._last_pitch_delta = 0.0

    def reward(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        reward = -self.STEP_PENALTY
        previous_grid, _ = self._grid_and_visibility(previous)
        current_grid, _ = self._grid_and_visibility(current)
        broken = np.flatnonzero(
            (previous_grid != self.CATEGORY_AIR)
            & (current_grid == self.CATEGORY_AIR)
        )
        newly_broken_non_bed = sum(
            int(previous_grid[index] != self.CATEGORY_BED) for index in broken
        )
        previous_excess = max(
            self._non_bed_blocks_broken - self._non_bed_block_budget, 0
        )
        self._non_bed_blocks_broken += newly_broken_non_bed
        current_excess = max(
            self._non_bed_blocks_broken - self._non_bed_block_budget, 0
        )
        excess_penalty = self.EXTRA_BLOCK_PENALTY * (
            current_excess - previous_excess
        )
        reward -= excess_penalty
        self._excess_block_penalty_total += excess_penalty

        previous_target = self._raycast_index(previous)
        current_target = self._raycast_index(current)
        selected_slot = int(current["player.selected_slot"])
        current_look_category = self._look_category(current)
        if current_look_category == self.CATEGORY_BED - 1:
            self._bed_focus_steps += 1
        if current_look_category in range(
            self.CATEGORY_BED - 1, self.CATEGORY_OBSIDIAN
        ):
            self._breakable_focus_steps += 1
        if self._last_yaw_delta == 0.0 and self._last_pitch_delta == 0.0:
            self._camera_noop_steps += 1
        self._absolute_yaw_delta_total += abs(self._last_yaw_delta)
        self._absolute_pitch_delta_total += abs(self._last_pitch_delta)
        if int(previous["player.selected_slot"]) != selected_slot:
            self._tool_switch_steps += 1
        if selected_slot in self.TOOL_HOTBAR_SLOTS:
            tool_index = self.TOOL_HOTBAR_SLOTS.index(selected_slot)
            self._tool_selection_steps[tool_index] += 1

        attacked_category = self.CATEGORY_AIR
        if self._last_attack:
            self._attack_command_steps += 1
            if previous_target is not None and previous_target in broken:
                attacked_category = int(previous_grid[previous_target])
            elif current["raycast.type"] == "BLOCK":
                # Use the raw category so bedrock/other blocks outside the
                # model grid are still penalized as required.
                attacked_category = int(current["raycast.category"])
            if attacked_category >= self.CATEGORY_BED:
                self._attack_steps += 1
            if attacked_category == self.CATEGORY_AIR:
                self._air_attack_steps += 1
            elif attacked_category == self.CATEGORY_BED:
                self._bed_attack_steps += 1
            elif attacked_category in self.DEFENSE_CATEGORIES:
                self._defense_attack_steps += 1
            else:
                self._other_attack_steps += 1
            if self._correct_tool(attacked_category, selected_slot):
                self._correct_tool_attack_steps += 1
            else:
                # Air/misses and other solid blocks have no correct tool in the
                # V2 contract, so attacking either is always penalized.
                self._wrong_tool_attack_steps += 1
                reward -= self.WRONG_TOOL_ATTACK_PENALTY
                self._wrong_tool_penalty_total += self.WRONG_TOOL_ATTACK_PENALTY

        current_progress = float(
            np.clip(current["player.block_break_progress"], 0.0, 1.0)
        )
        self._max_observed_break_progress = max(
            self._max_observed_break_progress, current_progress
        )
        progress_delta = 0.0
        previous_category = (
            int(previous_grid[previous_target])
            if previous_target is not None
            else self.CATEGORY_AIR
        )

        if previous_target is not None and previous_target in broken:
            progress_group = self._progress_group(
                previous_target, previous_category
            )
            threshold = (
                self._progress_high_water.get(progress_group, 0.0)
                if progress_group is not None
                else 0.0
            )
            if self._last_attack and self._correct_tool(
                previous_category, selected_slot
            ) and progress_group is not None:
                # Vanilla reports zero after a completed break. Treat the
                # completed logical layer as 1.0 without ever rewarding the
                # same range twice.
                gained = max(1.0 - threshold, 0.0)
                progress_delta += gained
                self._max_break_progress = 1.0
            if progress_group is not None:
                self._progress_high_water[progress_group] = 1.0

        if current_target is not None:
            current_category = int(current_grid[current_target])
            progress_group = self._progress_group(current_target, current_category)
            threshold = (
                self._progress_high_water.get(progress_group, 0.0)
                if progress_group is not None
                else 0.0
            )
            current_tool_correct = self._correct_tool(
                current_category, selected_slot
            )
            if progress_group is not None and current_tool_correct:
                gained = max(current_progress - threshold, 0.0)
                progress_delta += gained
                threshold = max(threshold, current_progress)
                self._progress_high_water[progress_group] = threshold
                self._max_break_progress = max(
                    self._max_break_progress, current_progress
                )
            elif progress_group is not None:
                # Remember progress achieved with a wrong tool so selecting the
                # right tool, or another block in this layer, cannot claim it
                # retroactively.
                self._progress_high_water[progress_group] = max(
                    threshold, current_progress
                )

        progress_reward = self.PROGRESS_SCALE * progress_delta
        reward += progress_reward
        self._progress_reward_total += progress_reward

        previous_beds = int(np.count_nonzero(previous_grid == self.CATEGORY_BED))
        current_beds = int(np.count_nonzero(current_grid == self.CATEGORY_BED))
        if current_beds < previous_beds:
            # Vanilla may remove the paired bed block automatically. Destroying
            # either half is one success event and therefore one terminal bonus.
            reward += self.BED_BREAK_REWARD

        self._episode_steps += 1
        return float(reward)

    def terminated(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        return self.bed_present_count(current) < 2

    def truncated(self, step_count: int, current: Mapping[str, Any]) -> bool:
        if self._partial_defense_rehearsal:
            limit = self.PARTIAL_DEFENSE_REHEARSAL_LIMIT
        else:
            limit = (
                self.STAGE_STEP_LIMITS[self._active_stage_index]
                if self._training_mode
                else self.STAGE_STEP_LIMITS[-1]
            )
        return step_count >= min(self.max_episode_steps, limit)

    def info(self, raw: Mapping[str, Any], *, episode_end: bool = True) -> dict[str, Any]:
        info = super().info(raw, episode_end=episode_end)
        if not episode_end:
            return info
        success = self.bed_present_count(raw) < 2
        excess = max(
            self._non_bed_blocks_broken - self._non_bed_block_budget, 0
        )
        no_obsidian_rehearsal = (
            self._training_mode
            and self._active_stage_index == 5
            and "obsidian" not in self._initial_layer_materials.split("+")
        )
        rehearsal_episode = (
            no_obsidian_rehearsal or self._partial_defense_rehearsal
        )
        rehearsal_kind = (
            "partial_defense"
            if self._partial_defense_rehearsal
            else "no_obsidian"
            if no_obsidian_rehearsal
            else "none"
        )
        info.update(
            {
                "success": success,
                "bed_halves_remaining": self.bed_present_count(raw),
                "initial_layers": self._initial_layers,
                "layer_materials": self._initial_layer_materials,
                "initial_selected_tool": self._initial_selected_tool,
                "tool_control_mode": (
                    "hold_current"
                    if self._training_mode and self._active_stage_index <= 1
                    else "policy"
                ),
                "broken_blocks": self._non_bed_blocks_broken,
                "excess_blocks_broken": excess,
                "completion_fraction": 1.0 if success else 0.0,
                # attack_fraction is the actual binary action. Keep block hits
                # separate so a policy attacking air can no longer look idle.
                "attack_fraction": self._fraction(self._attack_command_steps),
                "block_attack_fraction": self._fraction(self._attack_steps),
                "correct_tool_attack_fraction": self._fraction(
                    self._correct_tool_attack_steps
                ),
                "wrong_tool_attack_fraction": self._fraction(
                    self._wrong_tool_attack_steps
                ),
                "valid_attack_fraction": self._fraction(
                    self._correct_tool_attack_steps
                ),
                "air_attack_fraction": self._fraction(self._air_attack_steps),
                "bed_attack_fraction": self._fraction(self._bed_attack_steps),
                "defense_attack_fraction": self._fraction(
                    self._defense_attack_steps
                ),
                "other_attack_fraction": self._fraction(self._other_attack_steps),
                "bed_focus_fraction": self._fraction(self._bed_focus_steps),
                "breakable_focus_fraction": self._fraction(
                    self._breakable_focus_steps
                ),
                "camera_noop_fraction": self._fraction(self._camera_noop_steps),
                "mean_abs_yaw_delta_degrees": self._mean_per_step(
                    self._absolute_yaw_delta_total
                ),
                "mean_abs_pitch_delta_degrees": self._mean_per_step(
                    self._absolute_pitch_delta_total
                ),
                "tool_switch_fraction": self._fraction(self._tool_switch_steps),
                "pickaxe_selection_fraction": self._fraction(
                    self._tool_selection_steps[0]
                ),
                "axe_selection_fraction": self._fraction(
                    self._tool_selection_steps[1]
                ),
                "shears_selection_fraction": self._fraction(
                    self._tool_selection_steps[2]
                ),
                "max_break_progress": self._max_break_progress,
                "max_observed_break_progress": self._max_observed_break_progress,
                "progress_reward_total": self._progress_reward_total,
                "wrong_tool_penalty_total": self._wrong_tool_penalty_total,
                "excess_block_penalty_total": self._excess_block_penalty_total,
                "curriculum_eligible": not rehearsal_episode,
                "curriculum_rehearsal": rehearsal_episode,
                "curriculum_rehearsal_kind": rehearsal_kind,
                "curriculum_stage": self._curriculum_stage,
                "curriculum_stage_index": self._active_stage_index,
                "curriculum_steps": self._training_steps,
                "episode_limit": (
                    min(
                        self.max_episode_steps,
                        self.PARTIAL_DEFENSE_REHEARSAL_LIMIT,
                    )
                    if self._partial_defense_rehearsal
                    else min(
                        self.max_episode_steps,
                        self.STAGE_STEP_LIMITS[self._active_stage_index],
                    )
                    if self._training_mode
                    else min(self.max_episode_steps, self.STAGE_STEP_LIMITS[-1])
                ),
            }
        )
        return info

    def bed_present_count(self, raw: Mapping[str, Any]) -> int:
        grid, _ = self._grid_and_visibility(raw)
        return int(grid[self._grid_index(3, 0, 3)] == self.CATEGORY_BED) + int(
            grid[self._grid_index(4, 0, 3)] == self.CATEGORY_BED
        )

    def describe_action(self, action: Any) -> str:
        values = np.asarray(action, dtype=np.int64)
        return (
            f"yaw={self.YAW_DEGREES[int(values[0])]:+g}deg "
            f"pitch={self.PITCH_DEGREES[int(values[1])]:+g}deg "
            f"tool={self.TOOL_NAMES[int(values[2])]} attack={bool(values[3])}"
        )

    def describe_reset(self, raw: Mapping[str, Any]) -> str:
        return (
            f"task={self.task_id} stage={self._curriculum_stage} "
            f"layers={self._initial_layers} materials={self._initial_layer_materials or 'none'} "
            f"tool={self._initial_selected_tool}"
        )

    def describe_observation(self, raw: Mapping[str, Any]) -> str:
        looked = self.LOOK_CATEGORY_NAMES[self._look_category(raw)]
        progress = float(raw["player.block_break_progress"])
        return (
            f"looked={looked} progress={progress:.1%} "
            f"safe=({float(raw['raycast.safe_yaw_correction']):+.2f},"
            f"{float(raw['raycast.safe_pitch_correction']):+.2f})deg"
        )

    def _grid_and_visibility(
        self, raw: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = id(raw)
        cached = self._grid_cache.get(cache_key)
        if cached is not None and cached[0] is raw:
            return cached[1], cached[2]
        grid = np.asarray(raw["bed_defense_v2.block_grid"], dtype=np.int8)
        visible = np.asarray(
            raw["bed_defense_v2.visible_mask"], dtype=np.int8
        ).astype(bool)
        if grid.shape != (self.GRID_CELLS,) or visible.shape != (self.GRID_CELLS,):
            raise ValueError("bed-defense V2 grid must contain exactly 224 cells")
        if np.any((grid < self.CATEGORY_AIR) | (grid > self.CATEGORY_OTHER)):
            raise ValueError("bed-defense V2 grid contains an invalid category")
        grid.setflags(write=False)
        visible.setflags(write=False)
        if len(self._grid_cache) >= 4:
            self._grid_cache.pop(next(iter(self._grid_cache)))
        self._grid_cache[cache_key] = (raw, grid, visible)
        return grid, visible

    def _bed_axes(
        self, raw: Mapping[str, Any]
    ) -> tuple[tuple[float, float, float], tuple[float, float], tuple[float, float]]:
        foot = tuple(float(raw[f"task_block_0.{axis}"]) for axis in ("x", "y", "z"))
        head = tuple(float(raw[f"task_block_1.{axis}"]) for axis in ("x", "y", "z"))
        longitudinal = (head[0] - foot[0], head[2] - foot[2])
        length = math.hypot(*longitudinal)
        if not math.isclose(length, 1.0, abs_tol=1e-6):
            raise ValueError("bed halves do not define a unit longitudinal axis")
        longitudinal = (longitudinal[0] / length, longitudinal[1] / length)
        lateral = (-longitudinal[1], longitudinal[0])
        center = (
            (foot[0] + head[0]) * 0.5,
            math.floor(foot[1]),
            (foot[2] + head[2]) * 0.5,
        )
        return center, longitudinal, lateral

    def _player_relative_coordinates(
        self, raw: Mapping[str, Any]
    ) -> tuple[float, float, float]:
        center, longitudinal, lateral = self._bed_axes(raw)
        dx = float(raw["player.x"]) - center[0]
        dz = float(raw["player.z"]) - center[2]
        return (
            dx * longitudinal[0] + dz * longitudinal[1],
            dx * lateral[0] + dz * lateral[1],
            float(raw["player.y"]) - center[1],
        )

    def _angles_to_bed_center(self, raw: Mapping[str, Any]) -> tuple[float, float]:
        center, _, _ = self._bed_axes(raw)
        dx = center[0] - float(raw["player.x"])
        dy = center[1] + self.BED_HEIGHT * 0.5 - (
            float(raw["player.y"]) + self.EYE_HEIGHT
        )
        dz = center[2] - float(raw["player.z"])
        desired_yaw = math.degrees(math.atan2(-dx, dz))
        desired_pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
        return (
            self._wrap_degrees(desired_yaw - float(raw["player.yaw"])),
            desired_pitch - float(raw["player.pitch"]),
        )

    def _look_category(self, raw: Mapping[str, Any]) -> int:
        if raw["raycast.type"] != "BLOCK":
            return 0
        raw_category = int(raw["raycast.category"])
        return raw_category - 1 if self.CATEGORY_BED <= raw_category <= self.CATEGORY_OTHER else 0

    def _looked_relative_coordinates(
        self, raw: Mapping[str, Any]
    ) -> tuple[float, float, float]:
        if raw["raycast.type"] != "BLOCK":
            return 0.0, 0.0, 0.0
        center, longitudinal, lateral = self._bed_axes(raw)
        block_x = int(raw["raycast.x"]) + 0.5
        block_z = int(raw["raycast.z"]) + 0.5
        dx = block_x - center[0]
        dz = block_z - center[2]
        return (
            dx * longitudinal[0] + dz * longitudinal[1],
            dx * lateral[0] + dz * lateral[1],
            float(int(raw["raycast.y"]) - center[1]),
        )

    def _raycast_index(self, raw: Mapping[str, Any]) -> int | None:
        if raw["raycast.type"] != "BLOCK":
            return None
        center, longitudinal, lateral = self._bed_axes(raw)
        block_x = int(raw["raycast.x"]) + 0.5
        block_z = int(raw["raycast.z"]) + 0.5
        dx = block_x - center[0]
        dz = block_z - center[2]
        long_coordinate = dx * longitudinal[0] + dz * longitudinal[1]
        lateral_coordinate = dx * lateral[0] + dz * lateral[1]
        longitudinal_index = int(round(long_coordinate + 3.5))
        lateral_index = int(round(lateral_coordinate + 3.0))
        vertical = int(raw["raycast.y"]) - int(center[1])
        if not (
            0 <= longitudinal_index < self.GRID_LONGITUDINAL
            and 0 <= lateral_index < self.GRID_LATERAL
            and 0 <= vertical < self.GRID_VERTICAL
        ):
            return None
        return self._grid_index(longitudinal_index, vertical, lateral_index)

    def _defense_layer_count(self, grid: np.ndarray) -> int:
        maximum = 0
        for index in np.flatnonzero(np.isin(grid, tuple(self.DEFENSE_CATEGORIES))):
            maximum = max(maximum, self._block_shell(int(index)))
        return maximum

    def _progress_group(self, index: int, category: int) -> int | None:
        if category == self.CATEGORY_BED:
            return 0
        if category in self.DEFENSE_CATEGORIES:
            return self._block_shell(index)
        return None

    def _block_shell(self, index: int) -> int:
        longitudinal, vertical, lateral = self._index_coordinates(index)
        return min(
            abs(longitudinal - 3) + abs(lateral - 3) + vertical,
            abs(longitudinal - 4) + abs(lateral - 3) + vertical,
        )

    def _layer_material_names(self, grid: np.ndarray) -> str:
        names: list[str] = []
        for shell in range(1, self._initial_layers + 1):
            materials: set[int] = set()
            for index, category in enumerate(grid):
                longitudinal, vertical, lateral = self._index_coordinates(index)
                cell_shell = min(
                    abs(longitudinal - 3) + abs(lateral - 3) + vertical,
                    abs(longitudinal - 4) + abs(lateral - 3) + vertical,
                )
                if cell_shell == shell and int(category) in self.DEFENSE_CATEGORIES:
                    materials.add(int(category))
            names.append(
                self.CATEGORY_NAMES[next(iter(materials)) - 1]
                if len(materials) == 1
                else "mixed"
            )
        return "+".join(names)

    @classmethod
    def _correct_tool(cls, category: int, selected_slot: int) -> bool:
        if category == cls.CATEGORY_BED:
            return selected_slot in cls.TOOL_HOTBAR_SLOTS
        if category == cls.CATEGORY_WOOL:
            return selected_slot == 0
        if category == cls.CATEGORY_WOOD:
            return selected_slot == 1
        if category in (cls.CATEGORY_END_STONE, cls.CATEGORY_OBSIDIAN):
            return selected_slot == 2
        return False

    @classmethod
    def _tool_name_for_slot(cls, slot: int) -> str:
        try:
            return cls.TOOL_NAMES[cls.TOOL_HOTBAR_SLOTS.index(slot)]
        except ValueError:
            return "unknown"

    def _fraction(self, value: int) -> float:
        return value / self._episode_steps if self._episode_steps else 0.0

    def _mean_per_step(self, value: float) -> float:
        return value / self._episode_steps if self._episode_steps else 0.0

    @classmethod
    def _grid_index(cls, longitudinal: int, vertical: int, lateral: int) -> int:
        return (
            vertical * cls.GRID_LATERAL + lateral
        ) * cls.GRID_LONGITUDINAL + longitudinal

    @classmethod
    def _index_coordinates(cls, index: int) -> tuple[int, int, int]:
        longitudinal = index % cls.GRID_LONGITUDINAL
        remaining = index // cls.GRID_LONGITUDINAL
        lateral = remaining % cls.GRID_LATERAL
        vertical = remaining // cls.GRID_LATERAL
        return longitudinal, vertical, lateral

    @staticmethod
    def _wrap_degrees(value: float) -> float:
        return (value + 180.0) % 360.0 - 180.0


__all__ = ["BedDefenseBreakV2Task"]
