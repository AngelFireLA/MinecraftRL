import unittest
import numpy as np
from lean_shared_memory_vec_env import LeanSharedMemoryVecEnv


class LeanWorkerTest(unittest.TestCase):
    def test_contract_and_import_isolation(self):
        for dictionary in (False, True):
            specs = [
                {
                    "factory": "tests.lean_worker_fixture:CountingEnv",
                    "kwargs": {"instance_id": f"test-{i}", "dictionary": dictionary},
                }
                for i in range(2)
            ]
            env = LeanSharedMemoryVecEnv(specs)
            try:
                self.assertTrue(
                    all(
                        not d["torch_loaded"] and not d["sb3_loaded"]
                        for d in env.diagnostics()
                    )
                )
                env.seed(41)
                env.reset()
                self.assertEqual([x["seed"] for x in env.reset_infos], [41, 42])
                self.assertFalse(env.has_attr("missing"))
                self.assertEqual(env.get_attr("instance_id"), ["test-0", "test-1"])
                env.step(np.array([0, 0]))
                obs, rewards, dones, infos = env.step(np.array([0, 0]))
                self.assertTrue(dones.all())
                self.assertTrue(infos[0]["TimeLimit.truncated"])
                terminal = infos[0]["terminal_observation"]
                np.testing.assert_array_equal(
                    terminal["state"] if dictionary else terminal, [2, 3]
                )
                np.testing.assert_array_equal(
                    obs["state"][0] if dictionary else obs[0], [0, 1]
                )
            finally:
                env.close()
                env.close()
            self.assertTrue(all(not p.is_alive() for p in env.processes))

    def test_worker_error_and_invalid_constructor(self):
        env = LeanSharedMemoryVecEnv(
            [{"factory": "tests.lean_worker_fixture:CountingEnv"}]
        )
        try:
            env.reset()
            with self.assertRaisesRegex(RuntimeError, "deliberate worker failure"):
                env.step(np.array([2]))
        finally:
            env.close()
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            LeanSharedMemoryVecEnv(
                [{"task_id": "invalid", "max_episode_steps": 10, "environment": {}}]
            )
