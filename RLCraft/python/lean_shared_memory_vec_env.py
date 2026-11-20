"""SB3 adapter for standalone, Torch-free Minecraft environment workers."""

from __future__ import annotations
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from multiprocessing.connection import Listener
from shared_memory_vec_env import SharedMemoryVecEnv


class _Process:
    def __init__(self, process):
        self.process = process

    @property
    def pid(self):
        return self.process.pid

    @property
    def exitcode(self):
        return self.process.poll()

    def is_alive(self):
        return self.process.poll() is None

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()

    def join(self, timeout=None):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


class LeanSharedMemoryVecEnv(SharedMemoryVecEnv):
    def __init__(self, specs, startup_timeout=30.0, response_timeout=180.0):
        if not specs:
            raise ValueError("At least one environment specification is required")
        if os.name != "nt":
            raise RuntimeError("lean-shared currently requires Windows named pipes")
        self.waiting = False
        self.closed = False
        self._shared_closed = False
        self.response_timeout = response_timeout
        self.remotes = []
        self.processes = []
        self._observations = {}
        self._terminals = {}
        listeners = []
        pending = []
        try:
            for index, spec in enumerate(specs):
                address = r"\\.\pipe\minecraftrl-" + secrets.token_hex(16)
                key = secrets.token_bytes(32)
                listener = Listener(address, family="AF_PIPE", authkey=key)
                listeners.append(listener)
                slot = []

                def accept(listener=listener, slot=slot):
                    try:
                        slot.append(listener.accept())
                    except BaseException as error:
                        slot.append(error)

                thread = threading.Thread(target=accept, daemon=True)
                thread.start()
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).with_name("lean_env_worker.py")),
                    ],
                    stdin=subprocess.PIPE,
                    text=True,
                )
                self.processes.append(_Process(process))
                process.stdin.write(
                    json.dumps(
                        {
                            "address": address,
                            "authkey": key.hex(),
                            "index": index,
                            "spec": spec,
                        }
                    )
                    + "\n"
                )
                process.stdin.close()
                pending.append((thread, slot, process))
            deadline = time.monotonic() + startup_timeout
            for thread, slot, process in pending:
                while (
                    thread.is_alive()
                    and time.monotonic() < deadline
                    and process.poll() is None
                ):
                    thread.join(0.05)
                if not slot:
                    raise RuntimeError(
                        "Lightweight worker failed to connect before startup deadline"
                    )
                if isinstance(slot[0], BaseException):
                    raise RuntimeError("Worker connection failed") from slot[0]
                self.remotes.append(slot[0])
            self._initialize_buffers(len(specs))
        except BaseException:
            self.close()
            # Close connections accepted by threads but not yet registered.
            for _, slot, _ in pending:
                if slot and not isinstance(slot[0], BaseException):
                    slot[0].close()
            raise
        finally:
            for listener in listeners:
                listener.close()

    def _receive(self, remote, expected):
        if not remote.poll(self.response_timeout):
            raise TimeoutError(f"Timed out waiting for lightweight worker {expected}")
        return super()._receive(remote, expected)

    def step_wait(self):
        previous = self.reset_infos
        result = super().step_wait()
        self.reset_infos = [
            old if new is None else new for old, new in zip(previous, self.reset_infos)
        ]
        return result

    def env_is_wrapped(self, wrapper_class, indices=None):
        # Production workers own bare MinecraftEnv objects. Never pickle a trainer
        # wrapper class (e.g. SB3 Monitor) into a Torch-free child interpreter.
        return [False for _ in self._get_indices(indices)]

    def diagnostics(self):
        for remote in self.remotes:
            remote.send(("diagnostics", None))
        return [self._receive(remote, "diagnostics")[1] for remote in self.remotes]

    def _close_shared(self):
        if self._shared_closed:
            return
        self._shared_closed = True
        for item in [
            *self._observations.values(),
            *self._terminals.values(),
            *[
                getattr(self, name)
                for name in ("_rewards", "_dones", "_profiles")
                if hasattr(self, name)
            ],
        ]:
            item.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (OSError, EOFError):
                pass
        deadline = time.monotonic() + 3.0
        for process in self.processes:
            process.join(max(0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(2.0)
            if process.is_alive():
                process.kill()
                process.join(2.0)
        for remote in self.remotes:
            remote.close()
        self.waiting = False
        self._close_shared()
