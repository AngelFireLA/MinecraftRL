# ruff: noqa: E402
# Set numerical-library thread limits before importing NumPy/Gymnasium.
"""Standalone Gym worker; deliberately never imports the trainer or SB3."""

from __future__ import annotations
import os
import sys

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
import json
import importlib
import traceback
from typing import Any
from multiprocessing import shared_memory
from multiprocessing.connection import Client
import numpy as np
from worker_buffers import _ArrayDescriptor

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


def _copy_observation_to_buffers(
    observation: Any,
    buffers: dict[str, np.ndarray[Any, Any]],
    env_index: int,
) -> None:
    if "" in buffers:
        np.copyto(buffers[""][env_index], np.asarray(observation))
        return
    assert isinstance(observation, dict)
    for key, target in buffers.items():
        np.copyto(target[env_index], np.asarray(observation[key]))


def run_worker(remote: Any, env: Any, env_index: int) -> None:
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
                assert (
                    rewards is not None and dones is not None and profiles is not None
                )
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
                remote.send(("step", (info, reset_info if done else None)))
            elif command == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                _copy_observation_to_buffers(observation, observations, env_index)
                remote.send(("reset", reset_info))
            elif command == "close":
                env.close()
                remote.send(("closed", None))
                break
            elif command == "diagnostics":
                remote.send(
                    (
                        "diagnostics",
                        {
                            "torch_loaded": "torch" in sys.modules,
                            "sb3_loaded": "stable_baselines3" in sys.modules,
                            "pid": os.getpid(),
                        },
                    )
                )
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
                remote.send(("is_wrapped", False))
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
        env.close()
        for item in attached:
            item.close()
        remote.close()


def main() -> None:
    bootstrap = json.loads(sys.stdin.readline())
    remote = Client(
        bootstrap["address"],
        family="AF_PIPE",
        authkey=bytes.fromhex(bootstrap["authkey"]),
    )
    try:
        spec = bootstrap["spec"]
        if "factory" in spec:
            module, name = spec["factory"].split(":")
            env = getattr(importlib.import_module(module), name)(
                **spec.get("kwargs", {})
            )
        else:
            from minecraft_env import MinecraftEnv
            from task_factory import create_task

            task = create_task(spec["task_id"], spec["max_episode_steps"])
            task.set_training_mode(True)
            task.set_training_progress(0)
            env = MinecraftEnv(task, **spec["environment"])
        run_worker(remote, env, bootstrap["index"])
    except BaseException as error:
        try:
            remote.send(("error", (repr(error), traceback.format_exc())))
        except (OSError, EOFError):
            pass
    finally:
        remote.close()


if __name__ == "__main__":
    main()
