"""Focused unit coverage for registered task contracts and presets."""

import unittest

import numpy as np

from task_registry import get_task_preset, task_names
from tasks import (
    AimBreakTntTask,
    AimToolBreakTask,
    BedDefenseBreakV2Task,
    BridgingTask,
    MultiToolBreakTask,
    PvpBenchmarkTask,
)


def tnt_state(*, block_id: int = 777, aimed: bool = False) -> dict[str, object]:
    return {
        "type": "env_reset_result",
        "sequence": 0,
        "world_reset_applied": True,
        "task_goal.x": 2.5,
        "task_goal.y": 65.5,
        "task_goal.z": 3.5,
        "task_goal.distance": 4.0,
        "task_goal.yaw_delta": 18.0,
        "task_goal.pitch_delta": -9.0,
        "task_goal.block_id": block_id,
        "raycast.type": "BLOCK" if aimed else "MISS",
        "raycast.x": 2 if aimed else 0,
        "raycast.y": 65 if aimed else 0,
        "raycast.z": 3 if aimed else 0,
    }


def tool_state(
    *,
    block_id: int = AimToolBreakTask.STONE_BLOCK_ID,
    selected_slot: int = 0,
    aimed: bool = False,
    progress: float = 0.0,
    tool_speeds: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    default_speeds = {
        AimToolBreakTask.STONE_BLOCK_ID: (0.18, 0.007, 0.007),
        AimToolBreakTask.OAK_PLANKS_BLOCK_ID: (0.017, 0.13, 0.017),
        AimToolBreakTask.SAND_BLOCK_ID: (0.067, 0.067, 0.40),
    }
    speeds = tool_speeds or default_speeds.get(block_id, (0.0, 0.0, 0.0))
    return {
        **tnt_state(block_id=block_id, aimed=aimed),
        "task_goal.tool_speed_0": speeds[0],
        "task_goal.tool_speed_1": speeds[1],
        "task_goal.tool_speed_2": speeds[2],
        "player.selected_slot": selected_slot,
        "player.block_break_progress": progress,
    }


def multi_state(
    *,
    remaining: tuple[bool, ...] = (True, True, True),
    progress: float = 0.0,
    selected_slot: int = 0,
) -> dict[str, object]:
    target_count = len(remaining)
    result: dict[str, object] = {
        "type": "env_reset_result",
        "sequence": 0,
        "world_reset_applied": True,
        "task_blocks.count": target_count,
        "player.selected_slot": selected_slot,
        "player.block_break_progress": progress,
        "raycast.type": "BLOCK",
        "raycast.x": -2,
        "raycast.y": 64,
        "raycast.z": 2,
    }
    ids = (1, 5, 12, 0, 0)
    speeds = (
        (0.18, 0.007, 0.007),
        (0.017, 0.13, 0.017),
        (0.067, 0.067, 0.40),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    for index in range(5):
        present = index < target_count and remaining[index]
        x = -2 + index if index < target_count else 0
        values = {
            "x": x + 0.5 if index < target_count else 0.0,
            "y": 64.5 if index < target_count else 0.0,
            "z": 2.5 if index < target_count else 0.0,
            "distance": 3.0 if index < target_count else 0.0,
            "yaw_delta": float(index * 10) if index < target_count else 0.0,
            "pitch_delta": 0.0,
            "block_id": ids[index] if present else 0,
            "tool_speed_0": speeds[index][0] if present else 0.0,
            "tool_speed_1": speeds[index][1] if present else 0.0,
            "tool_speed_2": speeds[index][2] if present else 0.0,
        }
        for attribute, value in values.items():
            result[f"task_block_{index}.{attribute}"] = value
    return result


def bed_v2_state(
    *,
    grid: np.ndarray | None = None,
    visible: np.ndarray | None = None,
    raycast_index: int | None = None,
    progress: float = 0.0,
    selected_slot: int = 2,
    bed_halves: int = 2,
) -> dict[str, object]:
    task = BedDefenseBreakV2Task
    categories = (
        np.full(task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8)
        if grid is None
        else np.asarray(grid, dtype=np.int8).copy()
    )
    categories[task._grid_index(3, 0, 3)] = (
        task.CATEGORY_BED if bed_halves >= 1 else task.CATEGORY_AIR
    )
    categories[task._grid_index(4, 0, 3)] = (
        task.CATEGORY_BED if bed_halves >= 2 else task.CATEGORY_AIR
    )
    if visible is None:
        visibility = np.ones(task.GRID_CELLS, dtype=np.int8)
        # A complete defense does not expose either bed. Tests that model a
        # deliberately opened sightline opt in by supplying an explicit mask.
        if np.any(np.isin(categories, tuple(task.DEFENSE_CATEGORIES))):
            visibility[task._grid_index(3, 0, 3)] = 0
            visibility[task._grid_index(4, 0, 3)] = 0
    else:
        visibility = np.asarray(visible, dtype=np.int8).copy()
    result: dict[str, object] = {
        "type": "env_reset_result",
        "sequence": 0,
        "world_reset_applied": True,
        "task_blocks.count": 2,
        "task_block_0.x": 0.5,
        "task_block_0.y": 64.5,
        "task_block_0.z": 0.5,
        "task_block_1.x": 1.5,
        "task_block_1.y": 64.5,
        "task_block_1.z": 0.5,
        "player.x": 1.0,
        "player.y": 64.0,
        "player.z": 3.5,
        "player.yaw": 180.0,
        "player.pitch": 20.0,
        "player.motion_y": 0.0,
        "player.on_ground": True,
        "player.selected_slot": selected_slot,
        "player.block_break_progress": progress,
        "raycast.type": "MISS",
        "raycast.category": 0,
        "raycast.x": 0,
        "raycast.y": 0,
        "raycast.z": 0,
        "raycast.safe_yaw_correction": 0.0,
        "raycast.safe_pitch_correction": 0.0,
        "bed_defense_v2.block_grid": categories.tolist(),
        "bed_defense_v2.visible_mask": visibility.tolist(),
    }
    if raycast_index is not None:
        longitudinal, vertical, lateral = task._index_coordinates(raycast_index)
        result.update(
            {
                "raycast.type": "BLOCK",
                "raycast.category": int(categories[raycast_index]),
                "raycast.x": longitudinal - 3,
                "raycast.y": 64 + vertical,
                "raycast.z": lateral - 3,
            }
        )
    return result


def bridging_state(
    *,
    gap: int = 1,
    player_longitudinal: float | None = None,
    held_count: int = 64,
    ground_fraction: float = 1.0,
    on_ground: bool = True,
    extra_line_blocks: tuple[int, ...] = (),
    above_level_count: int = 0,
) -> dict[str, object]:
    target_z = gap + 1.5
    player_z = 1.29 if player_longitudinal is None else target_z + player_longitudinal
    grid = np.zeros(
        (BridgingTask.GRID_VERTICAL, BridgingTask.GRID_LONGITUDINAL),
        dtype=np.uint8,
    )
    target_index = BridgingTask.GRID_LONGITUDINAL - 1
    start_index = target_index - gap - 1
    grid[0, start_index] = 1
    grid[0, target_index] = 1
    for index in extra_line_blocks:
        grid[0, index] = 1
    return {
        "type": "env_reset_result",
        "sequence": 0,
        "world_reset_applied": True,
        "player.x": 0.5,
        "player.y": 65.0,
        "player.z": player_z,
        "player.yaw": 180.0,
        "player.pitch": 82.0,
        "player.motion_x": 0.0,
        "player.motion_z": 0.0,
        "player.on_ground": on_ground,
        "player.sneaking": False,
        "player.sprinting": False,
        "held.count": held_count,
        "raycast.type": "BLOCK",
        "raycast.side": "SOUTH",
        "raycast.distance": 2.0,
        "task_goal.x": 0.5,
        "task_goal.y": 64.5,
        "task_goal.z": target_z,
        "bridging.ground_fraction": ground_fraction,
        "bridging.distance_forward_edge": 0.3,
        "bridging.distance_left_edge": 0.5,
        "bridging.distance_right_edge": 0.5,
        "bridging.distance_rear_edge": 0.7,
        "bridging.can_place": True,
        "bridging.block_grid": grid.reshape(-1).tolist(),
        "bridging.above_level_count": above_level_count,
    }


class AimBreakTntTaskTest(unittest.TestCase):
    def test_action_and_success(self) -> None:
        task = AimBreakTntTask()
        encoded = task.encode_action(np.asarray([8, 8, 1], dtype=np.int64))
        self.assertAlmostEqual(encoded["mouse_dx"], 120.0)
        self.assertAlmostEqual(encoded["mouse_dy"], -80.0)
        self.assertTrue(encoded["attack_once"])

        initial = tnt_state()
        task.on_reset(initial)
        aimed = tnt_state(aimed=True)
        self.assertGreater(task.reward(initial, aimed), 0.0)
        destroyed = tnt_state(block_id=0)
        self.assertTrue(task.terminated(aimed, destroyed))
        self.assertEqual(task.reward(aimed, destroyed), 10.0)


class BridgingTaskTest(unittest.TestCase):
    def configured_task(self) -> BridgingTask:
        task = BridgingTask()
        task.set_training_mode(True)
        task.set_curriculum_stage(0)
        self.assertEqual(task.reset_payload(123, {}), {"curriculum_stage": 0})
        task.on_reset(bridging_state())
        return task

    def test_action_and_structured_observation_contract(self) -> None:
        task = self.configured_task()
        action = np.asarray([1, 1, 2, 1, 1, 5, 5], dtype=np.int64)
        encoded = task.encode_action(action)
        self.assertTrue(encoded["use_once"])
        self.assertTrue(encoded["left"])
        self.assertTrue(encoded["forward"])
        self.assertTrue(encoded["sprint"])
        self.assertTrue(encoded["sneak"])
        self.assertNotIn("jump", encoded)
        self.assertEqual(encoded["mouse_dx"], 0.0)
        observation = task.observation(bridging_state())
        self.assertTrue(task.observation_space.contains(observation))
        self.assertEqual(observation["state"].shape, (BridgingTask.STATE_SIZE,))
        self.assertEqual(
            observation["grid"].shape,
            (1, BridgingTask.GRID_VERTICAL, 1, BridgingTask.GRID_LONGITUDINAL),
        )
        self.assertEqual(observation["state"].size + observation["grid"].size, 98)
        self.assertAlmostEqual(float(observation["state"][0]), -1.21, places=5)
        self.assertAlmostEqual(float(observation["state"][4]), 0.0, places=5)
        self.assertAlmostEqual(float(observation["state"][5]), -1.0, places=5)

    def test_extension_safe_progress_success_and_fall_rewards(self) -> None:
        task = self.configured_task()
        initial = bridging_state()
        task.encode_action(np.asarray([1, 0, 0, 0, 0, 5, 5]))
        extended = bridging_state(
            held_count=63,
            extra_line_blocks=(BridgingTask.GRID_LONGITUDINAL - 2,),
        )
        self.assertGreater(task.reward(initial, extended), 0.0)
        target = bridging_state(player_longitudinal=0.0, held_count=63)
        self.assertGreater(task.reward(extended, target), BridgingTask.SUCCESS_BASE_REWARD)
        self.assertTrue(task.terminated(extended, target))

        falling_task = self.configured_task()
        fallen = bridging_state(on_ground=False, ground_fraction=0.0)
        fallen["player.y"] = 64.9
        self.assertLess(falling_task.reward(initial, fallen), -9.0)
        self.assertTrue(falling_task.truncated(1, fallen))

    def test_above_level_placement_is_penalized_and_truncated(self) -> None:
        task = self.configured_task()
        initial = bridging_state()
        task.encode_action(np.asarray([1, 0, 0, 0, 0, 5, 5]))
        above = bridging_state(held_count=63, above_level_count=1)
        reward = task.reward(initial, above)
        self.assertLess(reward, -9.0)
        self.assertFalse(task.terminated(initial, above))
        self.assertTrue(task.truncated(1, above))
        self.assertTrue(task.info(above)["illegal_height"])

    def test_variable_stage_samples_every_gap_inclusively(self) -> None:
        task = BridgingTask()
        task.set_training_mode(True)
        task.set_curriculum_stage(9)
        task.set_curriculum_validation_mode(True)
        sampled = set()
        for seed in range(49):
            task.reset_payload(seed, {})
            sampled.add(task._gap)
        self.assertEqual(sampled, set(range(16, 65)))

    def test_intermediate_curriculum_gap_sequence(self) -> None:
        task = BridgingTask()
        task.set_training_mode(True)
        task.set_curriculum_validation_mode(True)
        for stage, expected_gap in enumerate((1, 1, 5, 5, 8, 11, 15, 22, 30)):
            task.set_curriculum_stage(stage)
            task.reset_payload(123, {})
            self.assertEqual(task._gap, expected_gap)

    def test_rehearsal_uses_mastered_stages_and_is_not_mastery_eligible(self) -> None:
        task = BridgingTask()
        task.set_training_mode(True)
        task.set_curriculum_stage(4)
        task.set_curriculum_rehearsal(20)
        rehearsal_stages = set()
        rehearsal_count = 0
        for seed in range(2_000):
            task.reset_payload(seed, {})
            info = task.info(bridging_state())
            if info["curriculum_rehearsal"]:
                rehearsal_count += 1
                rehearsal_stages.add(info["curriculum_stage_index"])
                self.assertFalse(info["curriculum_eligible"])
                self.assertLess(info["curriculum_stage_index"], 4)
            else:
                self.assertTrue(info["curriculum_eligible"])
                self.assertEqual(info["curriculum_stage_index"], 4)
        self.assertGreaterEqual(rehearsal_count, 340)
        self.assertLessEqual(rehearsal_count, 460)
        self.assertIn(3, rehearsal_stages)
        self.assertTrue(rehearsal_stages.issubset(set(range(4))))

    def test_retention_focus_routes_rehearsal_to_failed_stage(self) -> None:
        task = BridgingTask()
        task.set_training_mode(True)
        task.set_curriculum_stage(6)
        task.set_curriculum_rehearsal(90, focus_stage=1)
        focused = 0
        active = 0
        for seed in range(2_000):
            task.reset_payload(seed, {})
            info = task.info(bridging_state())
            if info["curriculum_rehearsal"]:
                focused += 1
                self.assertEqual(info["curriculum_stage_index"], 1)
                self.assertEqual(
                    info["curriculum_rehearsal_kind"], "retention_focus"
                )
            else:
                active += 1
                self.assertEqual(info["curriculum_stage_index"], 6)
        self.assertGreater(focused, 1_700)
        self.assertGreater(active, 100)

    def test_validation_mode_disables_rehearsal(self) -> None:
        task = BridgingTask()
        task.set_training_mode(True)
        task.set_curriculum_stage(4)
        task.set_curriculum_validation_mode(True)
        for seed in range(100):
            task.reset_payload(seed, {})
            info = task.info(bridging_state())
            self.assertFalse(info["curriculum_rehearsal"])
            self.assertTrue(info["curriculum_eligible"])
            self.assertEqual(info["curriculum_stage_index"], 4)


class AimToolBreakTaskTest(unittest.TestCase):
    def test_action_encodes_rotation_tool_and_held_attack(self) -> None:
        task = AimToolBreakTask()
        encoded = task.encode_action(np.asarray([10, 10, 2, 1], dtype=np.int64))
        self.assertAlmostEqual(encoded["mouse_dx"], 600.0)
        self.assertAlmostEqual(encoded["mouse_dy"], -300.0)
        self.assertEqual(encoded["hotbar_slot"], 2)
        self.assertTrue(encoded["attack"])

    def test_observation_contains_block_and_selected_tool_one_hot(self) -> None:
        task = AimToolBreakTask()
        initial = tool_state(
            block_id=AimToolBreakTask.OAK_PLANKS_BLOCK_ID,
            selected_slot=1,
            aimed=True,
            progress=0.25,
        )
        task.on_reset(initial)
        observation = task.observation(initial)
        self.assertTrue(task.observation_space.contains(observation))
        np.testing.assert_allclose(observation[6:9], [0.017, 0.13, 0.017])
        np.testing.assert_array_equal(observation[9:12], [0.0, 1.0, 0.0])
        self.assertTrue(task.correct_tool(initial))

    def test_each_block_maps_to_its_effective_tool_slot(self) -> None:
        task = AimToolBreakTask()
        expected_slots = {
            task.STONE_BLOCK_ID: 0,
            task.OAK_PLANKS_BLOCK_ID: 1,
            task.SAND_BLOCK_ID: 2,
        }
        for block_id, expected_slot in expected_slots.items():
            initial = tool_state(block_id=block_id, selected_slot=expected_slot)
            task.on_reset(initial)
            self.assertEqual(task.correct_slot(initial), expected_slot)
            self.assertTrue(task.correct_tool(initial))

    def test_progress_reward_and_stable_goal_success(self) -> None:
        task = AimToolBreakTask()
        initial = tool_state(aimed=True, progress=0.0)
        task.on_reset(initial)
        task.encode_action(np.asarray([5, 5, 0, 1], dtype=np.int64))
        progressed = tool_state(aimed=True, progress=0.4)
        self.assertGreater(task.reward(initial, progressed), 0.0)
        destroyed = tool_state(block_id=0)
        self.assertTrue(task.terminated(progressed, destroyed))


class TaskRegistryTest(unittest.TestCase):
    def test_registered_tasks_create_their_contracts(self) -> None:
        tnt = get_task_preset("aim_break_tnt").create_task(max_episode_steps=17)
        tool = get_task_preset("aim_tool_break").create_task()
        self.assertIsInstance(tnt, AimBreakTntTask)
        self.assertEqual(tnt.max_episode_steps, 17)
        self.assertIsInstance(tool, AimToolBreakTask)
        self.assertEqual(tool.max_episode_steps, 100)
        multi = get_task_preset("multi_tool_break").create_task()
        self.assertIsInstance(multi, MultiToolBreakTask)
        bed_v2 = get_task_preset("bed_defense_break_v2").create_task()
        self.assertIsInstance(bed_v2, BedDefenseBreakV2Task)
        self.assertEqual(bed_v2.max_episode_steps, 320)
        bridging = get_task_preset("bridging").create_task()
        self.assertIsInstance(bridging, BridgingTask)
        self.assertEqual(bridging.max_episode_steps, 850)
        pvp = get_task_preset("pvp_benchmark").create_task()
        self.assertIsInstance(pvp, PvpBenchmarkTask)
        self.assertEqual(pvp.max_episode_steps, 240)
        self.assertEqual(task_names(), (
            "aim_break_tnt", "aim_tool_break", "bed_defense_break",
            "bed_defense_break_v2", "bridging", "multi_tool_break",
            "pvp_benchmark"
        ))

    def test_unknown_task_reports_available_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "aim_break_tnt"):
            get_task_preset("missing")


class MultiToolBreakTaskTest(unittest.TestCase):
    def test_training_curriculum_keeps_one_fixed_contract(self) -> None:
        task = MultiToolBreakTask()
        task.set_training_mode(True)
        expected = (
            (0, 1, "one_guided"),
            (1, 2, "two_guided"),
            (2, 3, "three_guided"),
            (3, 0, "multi_guided"),
            (4, 0, "multi_free"),
        )
        shape = task.observation_space.shape
        for stage_index, target_count, stage in expected:
            task.set_curriculum_stage(stage_index)
            self.assertEqual(task.reset_payload(1, {})["target_count"], target_count)
            self.assertEqual(task.curriculum_configuration()[2], stage)
            self.assertEqual(task.observation_space.shape, shape)

        with self.assertRaisesRegex(ValueError, "curriculum stage"):
            task.set_curriculum_stage(5)

    def test_progress_shaping_cannot_be_farmed_by_abandoning_a_block(self) -> None:
        task = MultiToolBreakTask()
        initial = multi_state(progress=0.0)
        progressed = multi_state(progress=0.5)
        abandoned = multi_state(progress=0.0)
        task.on_reset(initial)
        earned = task.reward(initial, progressed)
        repaid = task.reward(progressed, abandoned)
        self.assertGreater(earned, 0.0)
        self.assertLess(repaid, 0.0)
        self.assertLess(earned + repaid, 0.0)

    def test_guided_focus_advances_after_its_block_breaks(self) -> None:
        task = MultiToolBreakTask()
        task.set_training_mode(True)
        task.set_curriculum_stage(2)
        task.reset_payload(1, {})
        initial = multi_state()
        task.on_reset(initial)
        self.assertEqual(task.info(initial)["focus_index"], 0)

        one_broken = multi_state(remaining=(False, True, True))
        task.reward(initial, one_broken)
        self.assertEqual(task.info(one_broken)["focus_index"], 1)
        self.assertEqual(task.model_target_order(one_broken), [1, 0, 2, 3, 4])
        observation = task.observation(one_broken)
        # Raw target 1 has yaw_delta=10 degrees and axe as its best tool. It is
        # serialized into model slot 0 after raw target 0 is destroyed.
        self.assertAlmostEqual(observation[4], 10.0 / 180.0)
        self.assertEqual(observation[4 + 5], 1.0)
        np.testing.assert_allclose(observation[4 + 6:4 + 9], [0.017, 0.13, 0.017])

    def test_tool_shaping_is_reversible_and_valid_attack_is_rewarded(self) -> None:
        task = MultiToolBreakTask()
        wrong_tool = multi_state(selected_slot=1)
        correct_tool = multi_state(selected_slot=0)
        task.on_reset(wrong_tool)

        selected_correctly = task.reward(wrong_tool, correct_tool)
        selected_wrong_again = task.reward(correct_tool, wrong_tool)
        self.assertGreater(selected_correctly, 0.0)
        self.assertLess(selected_wrong_again, 0.0)
        self.assertLess(selected_correctly + selected_wrong_again, 0.0)

        task.encode_action(np.asarray([5, 5, 0, 1], dtype=np.int64))
        self.assertGreater(task.reward(correct_tool, correct_tool), 0.0)

    def test_free_stage_preserves_raw_target_order(self) -> None:
        task = MultiToolBreakTask()
        initial = multi_state()
        task.on_reset(initial)
        self.assertEqual(task.model_target_order(initial), [0, 1, 2, 3, 4])

    def test_fixed_observation_and_weighted_multi_block_reward(self) -> None:
        task = MultiToolBreakTask()
        initial = multi_state()
        task.on_reset(initial)
        observation = task.observation(initial)
        self.assertTrue(task.observation_space.contains(observation))
        self.assertEqual(task.remaining_count(initial), 3)
        self.assertEqual(task.correct_slot(initial, 0), 0)
        self.assertEqual(task.correct_slot(initial, 1), 1)
        self.assertEqual(task.correct_slot(initial, 2), 2)

        one_broken = multi_state(remaining=(False, True, True))
        reward = task.reward(initial, one_broken)
        self.assertGreater(reward, 3.0)
        self.assertFalse(task.terminated(initial, one_broken))

        all_broken = multi_state(remaining=(False, False, False))
        self.assertTrue(task.terminated(one_broken, all_broken))
        self.assertGreater(task.reward(one_broken, all_broken), 10.0)


class BedDefenseBreakV2TaskTest(unittest.TestCase):
    def test_camera_actions_omit_only_the_extreme_jitter_bins(self) -> None:
        task = BedDefenseBreakV2Task()
        self.assertEqual(task.YAW_DEGREES, (-30, -10, -3, -1, 0, 1, 3, 10, 30))
        self.assertEqual(
            task.PITCH_DEGREES, (-15, -5, -1, -0.25, 0, 0.25, 1, 5, 15)
        )
        # Attack makes this a real action, so the zero camera request must not
        # receive automatic idle jitter.
        encoded = task.encode_action(np.asarray([4, 4, 0, 1], dtype=np.int64))
        self.assertEqual(encoded["mouse_dx"], 0.0)
        self.assertEqual(encoded["mouse_dy"], 0.0)

    def test_truly_idle_action_gets_seeded_small_camera_jitter(self) -> None:
        first = BedDefenseBreakV2Task()
        second = BedDefenseBreakV2Task()
        initial = bed_v2_state(selected_slot=2)
        for task in (first, second):
            task.reset_payload(12345, {})
            task.on_reset(initial)

        action = np.asarray([4, 4, 0, 0], dtype=np.int64)
        first_idle = first.encode_action(action)
        second_idle = second.encode_action(action)
        self.assertEqual(first_idle, second_idle)
        yaw_delta = first_idle["mouse_dx"] * 0.15
        pitch_delta = -first_idle["mouse_dy"] * 0.15
        self.assertGreaterEqual(yaw_delta, -1.0)
        self.assertLessEqual(yaw_delta, 1.0)
        self.assertGreaterEqual(pitch_delta, -1.0)
        self.assertLessEqual(pitch_delta, 1.0)
        self.assertNotEqual((yaw_delta, pitch_delta), (0.0, 0.0))

        attacking = first.encode_action(
            np.asarray([4, 4, 0, 1], dtype=np.int64)
        )
        switching = first.encode_action(
            np.asarray([4, 4, 2, 0], dtype=np.int64)
        )
        self.assertEqual((attacking["mouse_dx"], attacking["mouse_dy"]), (0.0, 0.0))
        self.assertEqual((switching["mouse_dx"], switching["mouse_dy"]), (0.0, 0.0))

    def test_action_tool_order_maps_to_existing_hotbar(self) -> None:
        task = BedDefenseBreakV2Task()
        pickaxe = task.encode_action(np.asarray([4, 4, 0, 1], dtype=np.int64))
        shears = task.encode_action(np.asarray([4, 4, 2, 0], dtype=np.int64))
        self.assertEqual(pickaxe["hotbar_slot"], 2)
        self.assertEqual(shears["hotbar_slot"], 0)

    def test_exposed_stages_hold_randomized_tool_but_material_stage_releases_it(self) -> None:
        task = BedDefenseBreakV2Task()
        task.set_training_mode(True)
        initial = bed_v2_state(selected_slot=0)

        task.set_curriculum_stage(0)
        task.reset_payload(1, {})
        task.on_reset(initial)
        requested_pickaxe = task.encode_action(
            np.asarray([4, 4, 0, 1], dtype=np.int64)
        )
        self.assertEqual(requested_pickaxe["hotbar_slot"], 0)
        self.assertEqual(task.info(initial)["tool_control_mode"], "hold_current")

        task.set_curriculum_stage(2)
        task.reset_payload(2, {})
        requested_pickaxe = task.encode_action(
            np.asarray([4, 4, 0, 1], dtype=np.int64)
        )
        self.assertEqual(requested_pickaxe["hotbar_slot"], 2)
        self.assertEqual(task.info(initial)["tool_control_mode"], "policy")

    def test_observation_has_exact_state_and_visibility_contract(self) -> None:
        task = BedDefenseBreakV2Task()
        wool = task._grid_index(3, 0, 2)
        grid = np.full(task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8)
        grid[wool] = task.CATEGORY_WOOL
        visible = np.ones(task.GRID_CELLS, dtype=np.int8)
        visible[wool] = 0
        raw = bed_v2_state(
            grid=grid,
            visible=visible,
            raycast_index=wool,
            progress=0.25,
            selected_slot=2,
        )
        task.on_reset(raw)
        observation = task.observation(raw)
        self.assertTrue(task.observation_space.contains(observation))
        np.testing.assert_array_equal(observation["state"][:3], [1.0, 0.0, 0.0])
        self.assertEqual(observation["state"].shape, (24,))
        self.assertEqual(observation["grid"].shape, (8, 4, 7, 8))
        self.assertEqual(observation["grid"][7, 0, 2, 3], 1)
        # Beds remain known even if visibility says otherwise.
        hidden_bed = bed_v2_state(visible=np.zeros(task.GRID_CELLS, dtype=np.int8))
        bed_observation = task.observation(hidden_bed)
        self.assertEqual(bed_observation["grid"][1, 0, 3, 3], 1)

    def test_wrong_tool_penalty_progress_and_extra_block_penalty(self) -> None:
        task = BedDefenseBreakV2Task()
        self.assertEqual(task.WRONG_TOOL_ATTACK_PENALTY, 0.04)
        self.assertEqual(task.EXTRA_BLOCK_PENALTY, 0.5)
        first = task._grid_index(3, 0, 2)
        second = task._grid_index(4, 0, 2)
        grid = np.full(task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8)
        grid[first] = task.CATEGORY_WOOL
        grid[second] = task.CATEGORY_WOOL
        initial = bed_v2_state(
            grid=grid, raycast_index=first, selected_slot=0, progress=0.0
        )
        task.on_reset(initial)

        task.encode_action(np.asarray([4, 4, 2, 1], dtype=np.int64))
        progressed = bed_v2_state(
            grid=grid, raycast_index=first, selected_slot=0, progress=0.4
        )
        self.assertGreater(task.reward(initial, progressed), 1.0)

        after_first_grid = grid.copy()
        after_first_grid[first] = task.CATEGORY_AIR
        after_first = bed_v2_state(grid=after_first_grid, selected_slot=0)
        first_break_reward = task.reward(progressed, after_first)
        self.assertGreater(first_break_reward, 0.0)

        before_second = bed_v2_state(
            grid=after_first_grid,
            raycast_index=second,
            selected_slot=0,
            progress=0.9,
        )
        task.reward(after_first, before_second)
        after_second_grid = after_first_grid.copy()
        after_second_grid[second] = task.CATEGORY_AIR
        after_second = bed_v2_state(grid=after_second_grid, selected_slot=0)
        second_break_reward = task.reward(before_second, after_second)
        self.assertAlmostEqual(
            second_break_reward,
            -task.STEP_PENALTY - task.EXTRA_BLOCK_PENALTY,
        )

        wood_grid = grid.copy()
        wood_grid[first] = task.CATEGORY_WOOD
        wrong = bed_v2_state(
            grid=wood_grid, raycast_index=first, selected_slot=0
        )
        task.on_reset(wrong)
        task.encode_action(np.asarray([4, 4, 2, 1], dtype=np.int64))
        self.assertLess(task.reward(wrong, wrong), -task.STEP_PENALTY)

    def test_stage_contract_and_limits(self) -> None:
        task = BedDefenseBreakV2Task()
        self.assertEqual(task.STAGE_STEP_LIMITS, (30, 60, 60, 100, 140, 320, 320))
        task.set_training_mode(True)
        for index, (_, java_stage) in enumerate(task.CURRICULUM_STAGES):
            task.set_curriculum_stage(index)
            self.assertEqual(task.reset_payload(42, {})["curriculum_stage"], java_stage)
            self.assertFalse(task.truncated(task.STAGE_STEP_LIMITS[index] - 1, {}))
            self.assertTrue(task.truncated(task.STAGE_STEP_LIMITS[index], {}))

    def test_stage_five_no_obsidian_rehearsal_is_not_mastery_eligible(self) -> None:
        task = BedDefenseBreakV2Task()
        task.set_training_mode(True)
        task.set_curriculum_stage(5)
        task.reset_payload(5, {})
        wool = task._grid_index(3, 0, 2)
        rehearsal_grid = np.full(
            task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8
        )
        rehearsal_grid[wool] = task.CATEGORY_WOOL
        rehearsal = bed_v2_state(grid=rehearsal_grid, raycast_index=wool)
        task.on_reset(rehearsal)
        rehearsal_info = task.info(rehearsal)
        self.assertTrue(rehearsal_info["curriculum_rehearsal"])
        self.assertFalse(rehearsal_info["curriculum_eligible"])

        obsidian_grid = rehearsal_grid.copy()
        obsidian_grid[wool] = task.CATEGORY_OBSIDIAN
        obsidian = bed_v2_state(grid=obsidian_grid, raycast_index=wool)
        task.on_reset(obsidian)
        obsidian_info = task.info(obsidian)
        self.assertFalse(obsidian_info["curriculum_rehearsal"])
        self.assertTrue(obsidian_info["curriculum_eligible"])

    def test_partial_defense_rehearsal_targets_bed_without_shell_reward(self) -> None:
        task = BedDefenseBreakV2Task()
        task.set_training_mode(True)
        task.set_curriculum_stage(4)
        task.reset_payload(1, {})
        wool = task._grid_index(3, 0, 2)
        bed = task._grid_index(3, 0, 3)
        grid = np.full(task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8)
        grid[wool] = task.CATEGORY_WOOL
        visible = np.zeros(task.GRID_CELLS, dtype=np.int8)
        visible[wool] = 1
        visible[bed] = 1
        initial = bed_v2_state(
            grid=grid,
            visible=visible,
            raycast_index=wool,
            selected_slot=0,
            progress=0.0,
        )
        task.on_reset(initial)

        diagnostics = task.info(initial)
        self.assertTrue(diagnostics["curriculum_rehearsal"])
        self.assertEqual(
            diagnostics["curriculum_rehearsal_kind"], "partial_defense"
        )
        self.assertFalse(diagnostics["curriculum_eligible"])
        self.assertEqual(diagnostics["episode_limit"], 80)
        self.assertFalse(task.truncated(79, initial))
        self.assertTrue(task.truncated(80, initial))

        task.encode_action(np.asarray([4, 4, 2, 1], dtype=np.int64))
        progressed = bed_v2_state(
            grid=grid,
            visible=visible,
            raycast_index=wool,
            selected_slot=0,
            progress=0.5,
        )
        self.assertAlmostEqual(task.reward(initial, progressed), -task.STEP_PENALTY)

        broken_grid = grid.copy()
        broken_grid[wool] = task.CATEGORY_AIR
        broken = bed_v2_state(grid=broken_grid, visible=visible, selected_slot=0)
        break_reward = task.reward(progressed, broken)
        self.assertAlmostEqual(
            break_reward,
            -task.STEP_PENALTY - task.EXTRA_BLOCK_PENALTY,
        )
        final_diagnostics = task.info(broken)
        self.assertEqual(final_diagnostics["progress_reward_total"], 0.0)
        self.assertEqual(final_diagnostics["excess_blocks_broken"], 1)

    def test_outside_grid_bedrock_attack_is_penalized(self) -> None:
        task = BedDefenseBreakV2Task()
        outside = bed_v2_state(selected_slot=2)
        outside.update(
            {
                "raycast.type": "BLOCK",
                "raycast.category": task.CATEGORY_OTHER,
                "raycast.x": 30,
                "raycast.y": 63,
                "raycast.z": 30,
            }
        )
        task.on_reset(outside)
        task.encode_action(np.asarray([4, 4, 0, 1], dtype=np.int64))
        reward = task.reward(outside, outside)
        self.assertAlmostEqual(
            reward,
            -task.STEP_PENALTY - task.WRONG_TOOL_ATTACK_PENALTY,
        )

    def test_air_attack_is_penalized_and_logged_as_an_actual_action(self) -> None:
        task = BedDefenseBreakV2Task()
        air = bed_v2_state(selected_slot=2)
        task.on_reset(air)
        task.encode_action(np.asarray([4, 4, 0, 1], dtype=np.int64))
        reward = task.reward(air, air)
        diagnostics = task.info(air)

        self.assertAlmostEqual(
            reward,
            -task.STEP_PENALTY - task.WRONG_TOOL_ATTACK_PENALTY,
        )
        self.assertEqual(diagnostics["attack_fraction"], 1.0)
        self.assertEqual(diagnostics["air_attack_fraction"], 1.0)
        self.assertEqual(diagnostics["block_attack_fraction"], 0.0)
        self.assertEqual(diagnostics["wrong_tool_attack_fraction"], 1.0)

    def test_v2_action_focus_camera_and_tool_diagnostics_are_separate(self) -> None:
        task = BedDefenseBreakV2Task()
        bed = task._grid_index(3, 0, 3)
        initial = bed_v2_state(
            raycast_index=bed, selected_slot=0, progress=0.0
        )
        current = bed_v2_state(
            raycast_index=bed, selected_slot=2, progress=0.2
        )
        task.on_reset(initial)
        task.encode_action(np.asarray([4, 4, 0, 1], dtype=np.int64))
        task.reward(initial, current)
        diagnostics = task.info(current)

        self.assertEqual(diagnostics["attack_fraction"], 1.0)
        self.assertEqual(diagnostics["block_attack_fraction"], 1.0)
        self.assertEqual(diagnostics["bed_attack_fraction"], 1.0)
        self.assertEqual(diagnostics["air_attack_fraction"], 0.0)
        self.assertEqual(diagnostics["bed_focus_fraction"], 1.0)
        self.assertEqual(diagnostics["breakable_focus_fraction"], 1.0)
        self.assertEqual(diagnostics["camera_noop_fraction"], 1.0)
        self.assertEqual(diagnostics["tool_switch_fraction"], 1.0)
        self.assertEqual(diagnostics["pickaxe_selection_fraction"], 1.0)
        self.assertAlmostEqual(diagnostics["max_observed_break_progress"], 0.2)

    def test_wrong_to_right_tool_switch_cannot_reclaim_progress(self) -> None:
        task = BedDefenseBreakV2Task()
        wool = task._grid_index(3, 0, 2)
        grid = np.full(task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8)
        grid[wool] = task.CATEGORY_WOOL
        initial = bed_v2_state(
            grid=grid, raycast_index=wool, selected_slot=1, progress=0.0
        )
        task.on_reset(initial)

        task.encode_action(np.asarray([4, 4, 1, 1], dtype=np.int64))
        wrong_progress = bed_v2_state(
            grid=grid, raycast_index=wool, selected_slot=1, progress=0.5
        )
        task.reward(initial, wrong_progress)

        task.encode_action(np.asarray([4, 4, 2, 1], dtype=np.int64))
        correct_same_progress = bed_v2_state(
            grid=grid, raycast_index=wool, selected_slot=0, progress=0.5
        )
        switch_reward = task.reward(wrong_progress, correct_same_progress)
        self.assertAlmostEqual(switch_reward, -task.STEP_PENALTY)
        self.assertEqual(task.info(correct_same_progress)["progress_reward_total"], 0.0)

        # Repeating the wrong/right oscillation still cannot create progress reward.
        for _ in range(5):
            task.encode_action(np.asarray([4, 4, 1, 1], dtype=np.int64))
            task.reward(correct_same_progress, wrong_progress)
            task.encode_action(np.asarray([4, 4, 2, 1], dtype=np.int64))
            task.reward(wrong_progress, correct_same_progress)
        self.assertEqual(task.info(correct_same_progress)["progress_reward_total"], 0.0)

    def test_tool_selection_alone_has_no_reward(self) -> None:
        task = BedDefenseBreakV2Task()
        wool = task._grid_index(3, 0, 2)
        grid = np.full(task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8)
        grid[wool] = task.CATEGORY_WOOL
        initial = bed_v2_state(grid=grid, raycast_index=wool, selected_slot=1)
        changed = bed_v2_state(grid=grid, raycast_index=wool, selected_slot=0)
        task.on_reset(initial)
        task.encode_action(np.asarray([4, 4, 2, 0], dtype=np.int64))

        self.assertAlmostEqual(task.reward(initial, changed), -task.STEP_PENALTY)

    def test_correct_progress_survives_aim_loss_but_cannot_be_reearned(self) -> None:
        task = BedDefenseBreakV2Task()
        first = task._grid_index(3, 0, 2)
        second = task._grid_index(4, 0, 2)
        grid = np.full(task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8)
        grid[first] = task.CATEGORY_WOOL
        grid[second] = task.CATEGORY_WOOL
        initial = bed_v2_state(
            grid=grid, raycast_index=first, selected_slot=0, progress=0.0
        )
        task.on_reset(initial)
        task.encode_action(np.asarray([4, 4, 2, 1], dtype=np.int64))
        progressed = bed_v2_state(
            grid=grid, raycast_index=first, selected_slot=0, progress=0.5
        )
        earned = task.reward(initial, progressed)
        switched = bed_v2_state(
            grid=grid, raycast_index=second, selected_slot=0, progress=0.0
        )
        aim_lost = task.reward(progressed, switched)
        self.assertGreater(earned, 0.0)
        self.assertAlmostEqual(aim_lost, -task.STEP_PENALTY)
        self.assertAlmostEqual(
            task.info(switched)["progress_reward_total"],
            task.PROGRESS_SCALE * 0.5,
        )

        # Matching the same high-water mark on another block in the same layer
        # cannot farm another reward.
        second_same_progress = bed_v2_state(
            grid=grid, raycast_index=second, selected_slot=0, progress=0.5
        )
        repeated = task.reward(switched, second_same_progress)
        self.assertAlmostEqual(repeated, -task.STEP_PENALTY)

        # Only genuinely extending the layer's best progress earns more.
        second_new_progress = bed_v2_state(
            grid=grid, raycast_index=second, selected_slot=0, progress=0.75
        )
        extended = task.reward(second_same_progress, second_new_progress)
        self.assertAlmostEqual(
            extended,
            task.PROGRESS_SCALE * 0.25 - task.STEP_PENALTY,
        )

    def test_each_logical_layer_has_one_independent_progress_budget(self) -> None:
        task = BedDefenseBreakV2Task()
        inner = task._grid_index(3, 0, 2)
        outer = task._grid_index(3, 0, 1)
        grid = np.full(task.GRID_CELLS, task.CATEGORY_AIR, dtype=np.int8)
        grid[inner] = task.CATEGORY_WOOL
        grid[outer] = task.CATEGORY_WOOL
        initial = bed_v2_state(
            grid=grid, raycast_index=outer, selected_slot=0, progress=0.0
        )
        task.on_reset(initial)
        task.encode_action(np.asarray([4, 4, 2, 1], dtype=np.int64))

        outer_progress = bed_v2_state(
            grid=grid, raycast_index=outer, selected_slot=0, progress=0.5
        )
        outer_reward = task.reward(initial, outer_progress)
        inner_start = bed_v2_state(
            grid=grid, raycast_index=inner, selected_slot=0, progress=0.0
        )
        task.reward(outer_progress, inner_start)
        inner_progress = bed_v2_state(
            grid=grid, raycast_index=inner, selected_slot=0, progress=0.5
        )
        inner_reward = task.reward(inner_start, inner_progress)

        self.assertGreater(outer_reward, 2.0)
        self.assertGreater(inner_reward, 2.0)
        self.assertAlmostEqual(
            task.info(inner_progress)["progress_reward_total"],
            task.PROGRESS_SCALE,
        )

    def test_paired_bed_removal_is_one_terminal_bonus(self) -> None:
        task = BedDefenseBreakV2Task()
        bed_index = task._grid_index(3, 0, 3)
        initial = bed_v2_state(
            raycast_index=bed_index, selected_slot=1, progress=0.9
        )
        task.on_reset(initial)
        task.encode_action(np.asarray([4, 4, 1, 1], dtype=np.int64))
        destroyed = bed_v2_state(bed_halves=0, selected_slot=1)
        reward = task.reward(initial, destroyed)
        self.assertTrue(task.terminated(initial, destroyed))
        self.assertGreater(reward, task.BED_BREAK_REWARD)
        self.assertLess(reward, task.BED_BREAK_REWARD + 1.0)


if __name__ == "__main__":
    unittest.main()
