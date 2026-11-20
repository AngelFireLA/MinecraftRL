import unittest

import numpy as np

from task_registry import get_task_preset


def raw_state(**overrides):
    raw = {
        "world_reset_applied": True,
        "player.x": 10.0,
        "player.y": 64.0,
        "player.z": -4.0,
        "player.yaw": 0.0,
        "player.pitch": 0.0,
        "player.health": 20.0,
        "player.on_ground": True,
        "player.motion_x": 0.0,
        "player.motion_y": 0.0,
        "player.motion_z": 0.0,
        "player.sprinting": False,
        "player.collided_horizontally": False,
        "player.selected_slot": 0,
        "task_entity.id": 42,
        "task_entity.relative_x": 3.0,
        "task_entity.relative_y": 0.0,
        "task_entity.relative_z": 0.0,
        "task_entity.distance": 3.0,
        "task_entity.yaw_delta": -90.0,
        "task_entity.pitch_delta": 0.0,
        "task_entity.motion_x": 0.0,
        "task_entity.motion_y": 0.0,
        "task_entity.motion_z": 0.0,
        "task_entity.health": 20.0,
        "task_entity.alive": True,
        "raycast.type": "MISS",
        "raycast.id": -1,
    }
    raw.update(overrides)
    return raw


class PvpBenchmarkTaskTest(unittest.TestCase):
    def test_registered_observation_and_action_contracts(self) -> None:
        task = get_task_preset("pvp_benchmark").create_task()
        initial = raw_state()
        task.on_reset(initial)
        observation = task.observation(initial)
        self.assertEqual(observation.shape, (28,))
        self.assertTrue(task.observation_space.contains(observation))

        neutral = np.asarray([4, 3, 0, 0, 0, 0, 2], dtype=np.int64)
        encoded = task.encode_action(neutral)
        self.assertEqual(encoded["mouse_dx"], 0.0)
        self.assertEqual(encoded["mouse_dy"], 0.0)
        self.assertEqual(encoded["hotbar_slot"], 2)
        self.assertFalse(encoded["attack"])
        self.assertFalse(encoded["attack_once"])

        attack = task.encode_action(
            np.asarray([4, 3, 0, 0, 1, 1, 0], dtype=np.int64)
        )
        self.assertTrue(attack["attack"])
        self.assertTrue(attack["attack_once"])

    def test_damage_reward_and_terminal_outcomes(self) -> None:
        task = get_task_preset("pvp_benchmark").create_task()
        previous = raw_state()
        task.on_reset(previous)
        damaged = raw_state(**{"task_entity.health": 16.0})
        self.assertGreater(task.reward(previous, damaged), 1.0)
        self.assertFalse(task.terminated(previous, damaged))

        defeated = raw_state(
            **{"task_entity.health": 0.0, "task_entity.alive": False}
        )
        self.assertTrue(task.terminated(damaged, defeated))
        self.assertTrue(task.info(defeated)["success"])


if __name__ == "__main__":
    unittest.main()
