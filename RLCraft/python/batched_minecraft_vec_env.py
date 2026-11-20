"""Single-process, concurrently-driven VecEnv for MinecraftRL.

Unlike :class:`SubprocVecEnv` and :class:`SharedMemoryVecEnv`, this backend owns
all Gym environments and their TCP listeners in the trainer process.  Minecraft
actions are dispatched to every client before a small thread pool waits for the
independent socket replies.  Threads only cover blocking localhost I/O and task
post-processing; no Python interpreter or multiprocessing pipe is created per
Minecraft instance.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
import time
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn


class BatchedMinecraftVecEnv(DummyVecEnv):
    """Drive several persistent Minecraft environments concurrently in-process."""

    def __init__(self, env_fns: list[Any]) -> None:
        if not env_fns:
            raise ValueError("BatchedMinecraftVecEnv requires at least one environment")
        super().__init__(env_fns)
        self.waiting = False
        self.closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_envs,
            thread_name_prefix="minecraft-env",
        )
        self._step_futures: list[Future[Any]] = []
        self.last_dispatch_ms = 0.0
        self.last_wait_ms = 0.0

    @staticmethod
    def _supports_split_step(env: Any) -> bool:
        return callable(getattr(env, "begin_step", None)) and callable(
            getattr(env, "finish_step", None)
        )

    def step_async(self, actions: np.ndarray) -> None:
        dispatch_started = time.perf_counter()
        if self.closed:
            raise RuntimeError("step_async called on a closed vector environment")
        if self.waiting:
            raise RuntimeError("step_async called while another step is pending")
        self.actions = actions

        # Send every Minecraft action first.  This is the important distinction
        # from DummyVecEnv: no fast instance waits for Python to dispatch to a
        # slower/later array entry.
        split = [self._supports_split_step(env) for env in self.envs]
        def dispatch_one(index: int) -> None:
            if not split[index]:
                return
            env = self.envs[index]
            try:
                env.begin_step(actions[index])
            except BaseException as error:
                instance_id = getattr(env, "instance_id", f"env-{index}")
                raise RuntimeError(
                    f"Minecraft environment {instance_id!r} failed during action dispatch: {error}"
                ) from error

        dispatch_futures = [
            self._executor.submit(dispatch_one, index)
            for index in range(self.num_envs)
        ]
        try:
            # Sending actions concurrently avoids serializing ten tiny JSON
            # writes on the trainer thread.  This waits only for the sends, never
            # for a Minecraft tick or response.
            for future in as_completed(dispatch_futures):
                future.result()
        except BaseException:
            # A partially dispatched vector step cannot be represented safely to
            # SB3.  The caller will close all environments after this exception.
            for future in dispatch_futures:
                future.cancel()
            self.waiting = False
            raise

        self._step_futures = [
            self._executor.submit(
                self._complete_environment_step,
                index,
                split[index],
                actions[index],
            )
            for index in range(self.num_envs)
        ]
        self.last_dispatch_ms = (time.perf_counter() - dispatch_started) * 1000.0
        self.waiting = True

    def _complete_environment_step(
        self, index: int, split_step: bool, action: Any
    ) -> tuple[Any, float, bool, dict[str, Any], dict[str, Any]]:
        env = self.envs[index]
        phase = "receive" if split_step else "step"
        try:
            result = env.finish_step() if split_step else env.step(action)
            observation, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            info["TimeLimit.truncated"] = bool(truncated and not terminated)
            reset_info: dict[str, Any] = {}
            if done:
                info["terminal_observation"] = observation
                phase = "automatic reset"
                observation, reset_info = env.reset()
            return observation, float(reward), done, info, reset_info
        except BaseException as error:
            instance_id = getattr(env, "instance_id", f"env-{index}")
            raise RuntimeError(
                f"Minecraft environment {instance_id!r} failed during {phase}: {error}"
            ) from error

    def step_wait(self) -> VecEnvStepReturn:
        if not self.waiting:
            raise RuntimeError("step_wait called without a pending vector step")
        first_error: BaseException | None = None
        wait_started = time.perf_counter()
        try:
            # Read whichever response becomes ready first.  The result arrays are
            # still indexed in SB3 environment order, but a slow low-index client
            # cannot delay processing a later response or its error.
            future_indices = {
                future: index for index, future in enumerate(self._step_futures)
            }
            for future in as_completed(self._step_futures):
                index = future_indices[future]
                try:
                    observation, reward, done, info, reset_info = future.result()
                except BaseException as error:
                    first_error = error
                    break
                self.buf_rews[index] = reward
                self.buf_dones[index] = done
                self.buf_infos[index] = info
                if done:
                    self.reset_infos[index] = reset_info
                self._save_obs(index, observation)
            if first_error is not None:
                for future in self._step_futures:
                    future.cancel()
                raise first_error
            self.last_wait_ms = (time.perf_counter() - wait_started) * 1000.0
            for info in self.buf_infos:
                profile = info.setdefault("training_profile", {})
                profile["vector_dispatch_ms"] = self.last_dispatch_ms
                profile["vector_wait_ms"] = self.last_wait_ms
            return (
                self._obs_from_buf(),
                np.copy(self.buf_rews),
                np.copy(self.buf_dones),
                deepcopy(self.buf_infos),
            )
        finally:
            self._step_futures = []
            self.waiting = False

    def reset(self) -> VecEnvObs:
        if self.waiting:
            raise RuntimeError("reset called while a vector step is pending")

        def reset_one(index: int) -> tuple[Any, dict[str, Any]]:
            maybe_options = (
                {"options": self._options[index]} if self._options[index] else {}
            )
            try:
                return self.envs[index].reset(
                    seed=self._seeds[index], **maybe_options
                )
            except BaseException as error:
                instance_id = getattr(self.envs[index], "instance_id", f"env-{index}")
                raise RuntimeError(
                    f"Minecraft environment {instance_id!r} failed during reset: {error}"
                ) from error

        futures = [
            self._executor.submit(reset_one, index) for index in range(self.num_envs)
        ]
        try:
            future_indices = {
                future: index for index, future in enumerate(futures)
            }
            for future in as_completed(futures):
                index = future_indices[future]
                observation, self.reset_infos[index] = future.result()
                self._save_obs(index, observation)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            self._reset_seeds()
            self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        # Closing sockets first releases any thread blocked in finish_step().
        for env in self.envs:
            try:
                env.close()
            except (ConnectionError, OSError, RuntimeError):
                pass
        for future in self._step_futures:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._step_futures = []
        self.waiting = False


__all__ = ["BatchedMinecraftVecEnv"]
