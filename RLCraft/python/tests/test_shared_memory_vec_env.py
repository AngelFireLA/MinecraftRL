import unittest
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from shared_memory_vec_env import SharedMemoryVecEnv


class CountingEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0, 1, shape=(2, 2), dtype=np.uint8),
                "state": spaces.Box(-100, 100, shape=(2,), dtype=np.float32),
            }
        )
        self.value = 0
        self.marker = "initial"

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "state": np.asarray([self.value, self.value + 0.5], dtype=np.float32),
            "grid": np.full((2, 2), self.value % 2, dtype=np.uint8),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.value = int((options or {}).get("value", 0))
        return self._observation(), {
            "instance_id": self.instance_id,
            "seed": seed,
            "value": self.value,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_value = int(action)
        if action_value == 2:
            raise RuntimeError("requested worker failure")
        self.value += action_value + 1
        done = self.value >= 2
        return self._observation(), float(self.value), done, False, {
            "final_value": self.value,
            "training_profile": {
                "instance_id": self.instance_id,
                "env_step_ms": 1.0 + self.value,
                "bridge_wait_ms": 0.5,
            },
        }

    def set_marker(self, value: str) -> str:
        self.marker = value
        return self.marker


def make_counting_env(instance_id: str):
    def initialize() -> CountingEnv:
        return CountingEnv(instance_id)

    return initialize


class SharedMemoryVecEnvTest(unittest.TestCase):
    def test_dict_observations_terminal_state_and_profile_use_shared_buffers(self) -> None:
        env = SharedMemoryVecEnv(
            [make_counting_env("test-0"), make_counting_env("test-1")]
        )
        try:
            initial = env.reset()
            self.assertEqual(initial["state"].shape, (2, 2))
            np.testing.assert_array_equal(initial["grid"], 0)

            first, rewards, dones, infos = env.step(np.asarray([0, 0]))
            np.testing.assert_array_equal(rewards, [1.0, 1.0])
            np.testing.assert_array_equal(dones, [False, False])
            np.testing.assert_array_equal(first["state"][:, 0], [1.0, 1.0])
            self.assertEqual(infos[0]["training_profile"]["instance_id"], "test-0")
            self.assertEqual(infos[0]["training_profile"]["bridge_wait_ms"], 0.5)

            second, rewards, dones, infos = env.step(np.asarray([0, 0]))
            np.testing.assert_array_equal(rewards, [2.0, 2.0])
            np.testing.assert_array_equal(dones, [True, True])
            np.testing.assert_array_equal(second["state"][:, 0], [0.0, 0.0])
            np.testing.assert_array_equal(
                infos[0]["terminal_observation"]["state"], [2.0, 2.5]
            )
            self.assertEqual(env.reset_infos[0]["value"], 0)
        finally:
            env.close()

    def test_seed_options_attributes_and_methods_match_sb3_contract(self) -> None:
        env = SharedMemoryVecEnv(
            [make_counting_env("test-0"), make_counting_env("test-1")]
        )
        try:
            env.seed(123)
            env.set_options([{"value": 3}, {"value": 4}])
            observation = env.reset()
            np.testing.assert_array_equal(observation["state"][:, 0], [3.0, 4.0])
            self.assertEqual([info["seed"] for info in env.reset_infos], [123, 124])
            self.assertEqual(env.get_attr("instance_id"), ["test-0", "test-1"])
            self.assertTrue(env.has_attr("marker"))
            env.set_attr("marker", "set")
            self.assertEqual(env.get_attr("marker"), ["set", "set"])
            self.assertEqual(env.env_method("set_marker", "called"), ["called", "called"])
        finally:
            env.close()

    def test_worker_failure_is_reported_and_force_closable(self) -> None:
        env = SharedMemoryVecEnv([make_counting_env("test-0")])
        try:
            env.reset()
            env.step_async(np.asarray([2]))
            with self.assertRaisesRegex(RuntimeError, "requested worker failure"):
                env.step_wait()
        finally:
            env.waiting = False
            env.close()


if __name__ == "__main__":
    unittest.main()
