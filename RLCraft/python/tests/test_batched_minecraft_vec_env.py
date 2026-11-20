import time
import unittest
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from batched_minecraft_vec_env import BatchedMinecraftVecEnv


class SplitCountingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self, instance_id: str, delay: float = 0.0, dispatch_delay: float = 0.0
    ) -> None:
        self.instance_id = instance_id
        self.delay = delay
        self.dispatch_delay = dispatch_delay
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-100, 100, shape=(1,), dtype=np.float32)
        self.value = 0
        self.pending: int | None = None
        self.dispatched_at = 0.0
        self.finished_at = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.value = int((options or {}).get("value", 0))
        return np.asarray([self.value], dtype=np.float32), {
            "instance_id": self.instance_id,
            "seed": seed,
        }

    def begin_step(self, action: np.ndarray) -> None:
        if self.pending is not None:
            raise RuntimeError("already pending")
        if self.dispatch_delay:
            time.sleep(self.dispatch_delay)
        self.pending = int(action)
        self.dispatched_at = time.perf_counter()

    def finish_step(
        self,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.pending is None:
            raise RuntimeError("not pending")
        if self.pending == 2:
            raise RuntimeError("requested failure")
        time.sleep(self.delay)
        self.value += self.pending + 1
        self.pending = None
        self.finished_at = time.perf_counter()
        return (
            np.asarray([self.value], dtype=np.float32),
            float(self.value),
            self.value >= 2,
            False,
            {"value": self.value},
        )

    def step(self, action: np.ndarray):
        self.begin_step(action)
        return self.finish_step()


def make_env(instance_id: str, delay: float = 0.0, dispatch_delay: float = 0.0):
    return lambda: SplitCountingEnv(instance_id, delay, dispatch_delay)


class BatchedMinecraftVecEnvTest(unittest.TestCase):
    def test_dispatches_all_actions_before_waiting_and_preserves_order(self) -> None:
        env = BatchedMinecraftVecEnv(
            [make_env("fast", 0.005), make_env("slow", 0.05)]
        )
        try:
            env.reset()
            started = time.perf_counter()
            env.step_async(np.asarray([0, 0]))
            self.assertTrue(all(item.dispatched_at >= started for item in env.envs))
            observations, rewards, dones, _ = env.step_wait()
            np.testing.assert_array_equal(observations[:, 0], [1.0, 1.0])
            np.testing.assert_array_equal(rewards, [1.0, 1.0])
            np.testing.assert_array_equal(dones, [False, False])
            self.assertLess(env.envs[0].finished_at, env.envs[1].finished_at)
        finally:
            env.close()

    def test_action_dispatch_runs_concurrently(self) -> None:
        env = BatchedMinecraftVecEnv(
            [make_env("dispatch-a", dispatch_delay=0.05),
             make_env("dispatch-b", dispatch_delay=0.05)]
        )
        try:
            env.reset()
            started = time.perf_counter()
            env.step_async(np.asarray([0, 0]))
            dispatch_seconds = time.perf_counter() - started
            env.step_wait()
            self.assertLess(dispatch_seconds, 0.09)
        finally:
            env.close()

    def test_mixed_done_auto_reset_and_terminal_observation(self) -> None:
        env = BatchedMinecraftVecEnv([make_env("a"), make_env("b")])
        try:
            env.set_options([{"value": 1}, {"value": 0}])
            initial = env.reset()
            np.testing.assert_array_equal(initial[:, 0], [1.0, 0.0])
            observations, _, dones, infos = env.step(np.asarray([0, 0]))
            np.testing.assert_array_equal(dones, [True, False])
            np.testing.assert_array_equal(observations[:, 0], [0.0, 1.0])
            np.testing.assert_array_equal(infos[0]["terminal_observation"], [2.0])
            self.assertFalse(infos[0]["TimeLimit.truncated"])
        finally:
            env.close()

    def test_failure_names_instance_and_phase(self) -> None:
        env = BatchedMinecraftVecEnv([make_env("pvp-0")])
        try:
            env.reset()
            env.step_async(np.asarray([2]))
            with self.assertRaisesRegex(
                RuntimeError, "pvp-0.*receive.*requested failure"
            ):
                env.step_wait()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
