from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tests.lean_worker_fixture import CountingEnv
from phase_profiling import PhaseTimers
from gymnasium import spaces


def test_profiled_checkpoint_final_save_and_resume():
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: CountingEnv(dictionary=False)])
    env.action_space = env.envs[0].action_space = spaces.Discrete(2)
    model = PPO(
        "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, device="cpu", verbose=0
    )
    timers = PhaseTimers(model, env)
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint"
        model.learn(16)
        model.save(path)
        timers.close()
        assert "train" not in model.__dict__
        assert "collect_rollouts" not in model.__dict__
        model.save(path)
        restored = PPO.load(path, env=env, device="cpu")
        restored.learn(8)
    env.close()
