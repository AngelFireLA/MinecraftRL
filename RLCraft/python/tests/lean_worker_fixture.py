import gymnasium as gym
import numpy as np


class CountingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, instance_id="test-0", dictionary=True):
        self.instance_id = instance_id
        self.dictionary = dictionary
        box = gym.spaces.Box(-100, 100, (2,), np.float32)
        self.observation_space = gym.spaces.Dict({"state": box}) if dictionary else box
        self.action_space = gym.spaces.Discrete(3)
        self.value = 0

    def observation(self):
        a = np.array([self.value, self.value + 1], dtype=np.float32)
        return {"state": a} if self.dictionary else a

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.value = (options or {}).get("value", 0)
        return self.observation(), {"seed": seed, "value": self.value}

    def step(self, action):
        if action == 2:
            raise RuntimeError("deliberate worker failure")
        self.value += 1
        return self.observation(), 1.0, False, self.value >= 2, {}
