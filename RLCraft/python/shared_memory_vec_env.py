"""SB3 VecEnv using fixed shared-memory observation buffers on Windows."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Callable, Sequence
from multiprocessing import shared_memory
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


PROFILE_FIELDS = (
    "env_step_ms",
    "action_encode_ms",
    "pace_ms",
    "send_ms",
    "bridge_wait_ms",
    "protocol_ms",
    "socket_read_ms",
    "json_decode_ms",
    "task_ms",
    "action_client_ms",
    "server_tick_ms",
    "post_client_ms",
    "observation_serialize_ms",
    "java_total_ms",
)


from worker_buffers import _ArrayDescriptor

class _OwnedArray:
    def __init__(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)
        size = max(int(np.prod(shape, dtype=np.int64)) * self.dtype.itemsize, 1)
        self.memory = shared_memory.SharedMemory(create=True, size=size)
        self.array = np.ndarray(shape, dtype=self.dtype, buffer=self.memory.buf)
        self.array.fill(0)

    @property
    def descriptor(self) -> _ArrayDescriptor:
        return _ArrayDescriptor(self.memory.name, self.shape, self.dtype.str)

    def close(self) -> None:
        self.memory.close()
        try:
            self.memory.unlink()
        except FileNotFoundError:
            pass


class _AttachedArray:
    def __init__(self, descriptor: _ArrayDescriptor) -> None:
        self.memory = shared_memory.SharedMemory(name=descriptor.name)
        self.array = np.ndarray(
            descriptor.shape,
            dtype=np.dtype(descriptor.dtype),
            buffer=self.memory.buf,
        )

    def close(self) -> None:
        self.memory.close()


def _observation_leaves(
    observation_space: spaces.Space[Any],
) -> list[tuple[str, spaces.Box]]:
    if isinstance(observation_space, spaces.Box):
        return [("", observation_space)]
    if isinstance(observation_space, spaces.Dict):
        result: list[tuple[str, spaces.Box]] = []
        for key, child in observation_space.spaces.items():
            if not isinstance(child, spaces.Box):
                raise TypeError(
                    "SharedMemoryVecEnv supports Box leaves inside Dict observations"
                )
            result.append((key, child))
        return result
    raise TypeError("SharedMemoryVecEnv supports Box or Dict observation spaces")


def _copy_observation_to_buffers(
    observation: VecEnvObs,
    buffers: dict[str, np.ndarray[Any, Any]],
    env_index: int,
) -> None:
    if "" in buffers:
        np.copyto(buffers[""][env_index], np.asarray(observation))
        return
    assert isinstance(observation, dict)
    for key, target in buffers.items():
        np.copyto(target[env_index], np.asarray(observation[key]))


def _worker(  # noqa: C901
    remote: Any,
    parent_remote: Any,
    env_fn_wrapper: CloudpickleWrapper,
    env_index: int,
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: dict[str, Any] = {}
    attached: list[_AttachedArray] = []
    observations: dict[str, np.ndarray[Any, Any]] = {}
    terminals: dict[str, np.ndarray[Any, Any]] = {}
    rewards: np.ndarray[Any, Any] | None = None
    dones: np.ndarray[Any, Any] | None = None
    profiles: np.ndarray[Any, Any] | None = None
    try:
        while True:
            command, data = remote.recv()
            if command == "configure":
                observation_descriptors, terminal_descriptors, scalar_descriptors = data
                for key, descriptor in observation_descriptors.items():
                    item = _AttachedArray(descriptor)
                    attached.append(item)
                    observations[key] = item.array
                for key, descriptor in terminal_descriptors.items():
                    item = _AttachedArray(descriptor)
                    attached.append(item)
                    terminals[key] = item.array
                scalar_arrays: dict[str, np.ndarray[Any, Any]] = {}
                for key, descriptor in scalar_descriptors.items():
                    item = _AttachedArray(descriptor)
                    attached.append(item)
                    scalar_arrays[key] = item.array
                rewards = scalar_arrays["rewards"]
                dones = scalar_arrays["dones"]
                profiles = scalar_arrays["profiles"]
                remote.send(("configured", None))
            elif command == "step":
                assert rewards is not None and dones is not None and profiles is not None
                observation, reward, terminated, truncated, info = env.step(data)
                done = bool(terminated or truncated)
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                profile = info.pop("training_profile", None)
                profiles[env_index].fill(np.nan)
                if isinstance(profile, dict):
                    for field_index, field in enumerate(PROFILE_FIELDS):
                        if field in profile:
                            profiles[env_index, field_index] = float(profile[field])
                if done:
                    _copy_observation_to_buffers(observation, terminals, env_index)
                    observation, reset_info = env.reset()
                _copy_observation_to_buffers(observation, observations, env_index)
                rewards[env_index] = float(reward)
                dones[env_index] = done
                remote.send(("step", (info, reset_info)))
            elif command == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                _copy_observation_to_buffers(observation, observations, env_index)
                remote.send(("reset", reset_info))
            elif command == "close":
                env.close()
                remote.send(("closed", None))
                break
            elif command == "get_spaces":
                remote.send(("spaces", (env.observation_space, env.action_space)))
            elif command == "get_attr":
                remote.send(("attr", env.get_wrapper_attr(data)))
            elif command == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(("has_attr", True))
                except AttributeError:
                    remote.send(("has_attr", False))
            elif command == "set_attr":
                setattr(env, data[0], data[1])
                remote.send(("set_attr", None))
            elif command == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(("env_method", method(*data[1], **data[2])))
            elif command == "is_wrapped":
                remote.send(("is_wrapped", is_wrapped(env, data)))
            elif command == "render":
                remote.send(("render", env.render()))
            else:
                raise NotImplementedError(f"Unsupported worker command {command!r}")
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException as error:
        try:
            remote.send(("error", (repr(error), traceback.format_exc())))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        for item in attached:
            item.close()
        remote.close()


class SharedMemoryVecEnv(VecEnv):
    """Subprocess VecEnv whose fixed observations never travel through pipes."""

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env[Any, Any]]],
        start_method: str = "spawn",
    ) -> None:
        if not env_fns:
            raise ValueError("SharedMemoryVecEnv requires at least one environment")
        self.waiting = False
        self.closed = False
        self._shared_closed = False
        context = mp.get_context(start_method)
        self.remotes, work_remotes = zip(*[context.Pipe() for _ in env_fns])
        self.processes = []
        for index, (work_remote, remote, env_fn) in enumerate(
            zip(work_remotes, self.remotes, env_fns)
        ):
            process = context.Process(
                target=_worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn), index),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        self._initialize_buffers(len(env_fns))

    def _initialize_buffers(self, num_envs: int) -> None:
        self.remotes[0].send(("get_spaces", None))
        _, (observation_space, action_space) = self._receive(self.remotes[0], "spaces")
        super().__init__(num_envs, observation_space, action_space)

        self._observations: dict[str, _OwnedArray] = {}
        self._terminals: dict[str, _OwnedArray] = {}
        for key, leaf in _observation_leaves(observation_space):
            shape = (self.num_envs, *leaf.shape)
            self._observations[key] = _OwnedArray(shape, leaf.dtype)
            self._terminals[key] = _OwnedArray(shape, leaf.dtype)
        self._rewards = _OwnedArray((self.num_envs,), np.dtype(np.float32))
        self._dones = _OwnedArray((self.num_envs,), np.dtype(np.bool_))
        self._profiles = _OwnedArray(
            (self.num_envs, len(PROFILE_FIELDS)), np.dtype(np.float64)
        )
        self._profiles.array.fill(np.nan)
        descriptors = (
            {key: item.descriptor for key, item in self._observations.items()},
            {key: item.descriptor for key, item in self._terminals.items()},
            {
                "rewards": self._rewards.descriptor,
                "dones": self._dones.descriptor,
                "profiles": self._profiles.descriptor,
            },
        )
        for remote in self.remotes:
            remote.send(("configure", descriptors))
        for remote in self.remotes:
            self._receive(remote, "configured")
        self.instance_ids = self.get_attr("instance_id")

    @staticmethod
    def _receive(remote: Any, expected: str) -> tuple[str, Any]:
        try:
            message_type, payload = remote.recv()
        except EOFError as error:
            raise RuntimeError("Minecraft environment worker exited unexpectedly") from error
        if message_type == "error":
            summary, stack = payload
            raise RuntimeError(f"Minecraft environment worker failed: {summary}\n{stack}")
        if message_type != expected:
            raise RuntimeError(
                f"Expected worker response {expected!r}, received {message_type!r}"
            )
        return message_type, payload

    def _snapshot(self, terminal: bool = False) -> VecEnvObs:
        source = self._terminals if terminal else self._observations
        if "" in source:
            return source[""].array.copy()
        return {key: item.array.copy() for key, item in source.items()}

    def _terminal_observation(self, index: int) -> VecEnvObs:
        if "" in self._terminals:
            return self._terminals[""].array[index].copy()
        return {
            key: item.array[index].copy()
            for key, item in self._terminals.items()
        }

    def step_async(self, actions: np.ndarray) -> None:
        if self.waiting:
            raise RuntimeError("step_async called while another step is pending")
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        infos: list[dict[str, Any]] = []
        reset_infos: list[dict[str, Any]] = []
        for index, remote in enumerate(self.remotes):
            _, payload = self._receive(remote, "step")
            info, reset_info = payload
            if bool(self._dones.array[index]):
                info["terminal_observation"] = self._terminal_observation(index)
            profile_values = self._profiles.array[index]
            if np.isfinite(profile_values[0]):
                info["training_profile"] = {
                    "instance_id": self.instance_ids[index],
                    **{
                        field: float(profile_values[field_index])
                        for field_index, field in enumerate(PROFILE_FIELDS)
                        if np.isfinite(profile_values[field_index])
                    },
                }
            infos.append(info)
            reset_infos.append(reset_info)
        self.waiting = False
        self.reset_infos = reset_infos
        return (
            self._snapshot(),
            self._rewards.array.copy(),
            self._dones.array.copy(),
            infos,
        )

    def reset(self) -> VecEnvObs:
        for index, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[index], self._options[index])))
        reset_infos = []
        for remote in self.remotes:
            _, reset_info = self._receive(remote, "reset")
            reset_infos.append(reset_info)
        self.reset_infos = reset_infos
        self._reset_seeds()
        self._reset_options()
        return self._snapshot()

    def _target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        return [self.remotes[index] for index in self._get_indices(indices)]

    def has_attr(self, attr_name: str) -> bool:
        targets = self._target_remotes(None)
        for remote in targets:
            remote.send(("has_attr", attr_name))
        return all([self._receive(remote, "has_attr")[1] for remote in targets])

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        targets = self._target_remotes(indices)
        for remote in targets:
            remote.send(("get_attr", attr_name))
        return [self._receive(remote, "attr")[1] for remote in targets]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        targets = self._target_remotes(indices)
        for remote in targets:
            remote.send(("set_attr", (attr_name, value)))
        for remote in targets:
            self._receive(remote, "set_attr")

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        targets = self._target_remotes(indices)
        data = (method_name, method_args, method_kwargs)
        for remote in targets:
            remote.send(("env_method", data))
        return [self._receive(remote, "env_method")[1] for remote in targets]

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        targets = self._target_remotes(indices)
        for remote in targets:
            remote.send(("is_wrapped", wrapper_class))
        return [self._receive(remote, "is_wrapped")[1] for remote in targets]

    def get_images(self) -> Sequence[np.ndarray[Any, Any] | None]:
        if self.render_mode != "rgb_array":
            return [None for _ in self.remotes]
        for remote in self.remotes:
            remote.send(("render", None))
        return [self._receive(remote, "render")[1] for remote in self.remotes]

    def _close_shared(self) -> None:
        if self._shared_closed:
            return
        self._shared_closed = True
        for item in (
            *self._observations.values(),
            *self._terminals.values(),
            self._rewards,
            self._dones,
            self._profiles,
        ):
            item.close()

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.waiting:
                for remote in self.remotes:
                    if remote.poll(2.0):
                        try:
                            self._receive(remote, "step")
                        except RuntimeError:
                            pass
                self.waiting = False
            for remote in self.remotes:
                try:
                    remote.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for remote in self.remotes:
                try:
                    if remote.poll(2.0):
                        self._receive(remote, "closed")
                except (BrokenPipeError, EOFError, OSError, RuntimeError):
                    pass
            for process in self.processes:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
        finally:
            self.closed = True
            self._close_shared()


__all__ = ["PROFILE_FIELDS", "SharedMemoryVecEnv"]
