import json
import socket
import threading
import unittest
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from minecraft_env import MinecraftEnv, MinecraftProtocolError, MinecraftTask, PROTOCOL_VERSION


def observation(message_type: str, sequence: int, *, x: float = 1.0) -> dict[str, Any]:
    # A real Java response must contain only the fields negotiated at reset.
    result = {
        "type": message_type,
        "sequence": sequence,
        "player.x": x,
        "raycast.type": "BLOCK",
    }
    if message_type == "env_reset_result":
        result["world_reset_applied"] = True
    return result


def identify(client: socket.socket, instance_id: str, world: str = "void") -> None:
    messages = (
        {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "instance_id": instance_id,
            "world": world,
        },
        {"type": "client_ready", "instance_id": instance_id, "world": world},
    )
    for message in messages:
        client.sendall((json.dumps(message) + "\n").encode("utf-8"))


class ForwardAndTargetTask(MinecraftTask):
    TARGET_TYPES = {"MISS": 0, "BLOCK": 1, "ENTITY": 2}

    def __init__(self, *, max_episode_steps: int = 1) -> None:
        super().__init__(
            task_id="forward_and_target",
            action_space=spaces.Discrete(2),
            observation_space=spaces.Box(
                low=np.asarray([-np.inf, 0.0], dtype=np.float32),
                high=np.asarray([np.inf, 3.0], dtype=np.float32),
                dtype=np.float32,
            ),
            action_fields=("forward",),
            observation_fields=("player.x", "raycast.type"),
            max_episode_steps=max_episode_steps,
        )

    def encode_action(self, action: int) -> dict[str, Any]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")
        return {"forward": bool(action)}

    def observation(self, raw: Mapping[str, Any]) -> np.ndarray:
        target_type = self.TARGET_TYPES.get(str(raw["raycast.type"]), 3)
        return np.asarray([raw["player.x"], target_type], dtype=np.float32)

    def reward(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        return float(current["player.x"] - previous["player.x"])


class MinecraftEnvTest(unittest.TestCase):
    def test_fast_training_path_preserves_deterministic_task_trace(self) -> None:
        def run_trace(
            *, strict_validation: bool, info_mode: str
        ) -> tuple[list[Any], list[dict[str, Any]]]:
            env = MinecraftEnv(
                ForwardAndTargetTask(max_episode_steps=2),
                port=0,
                connect_timeout=2.0,
                step_timeout=2.0,
                include_raw_info=True,
                strict_validation=strict_validation,
                info_mode=info_mode,
            )

            def fake_java() -> None:
                with socket.create_connection(env.address, timeout=2.0) as client:
                    identify(client, "rl-0")
                    stream = client.makefile("rb")
                    json.loads(stream.readline())
                    client.sendall(
                        (json.dumps(observation("env_reset_result", 0)) + "\n").encode()
                    )
                    for sequence, x in enumerate((1.25, 2.0)):
                        json.loads(stream.readline())
                        client.sendall(
                            (json.dumps(observation(
                                "env_step_result", sequence, x=x
                            )) + "\n").encode()
                        )
                    self.assertEqual(
                        json.loads(stream.readline()), {"type": "env_close"}
                    )
                    client.sendall(b'{"type":"env_closed"}\n')

            thread = threading.Thread(target=fake_java, daemon=True)
            thread.start()
            infos: list[dict[str, Any]] = []
            trace: list[Any] = []
            try:
                initial, reset_info = env.reset(seed=99)
                infos.append(reset_info)
                trace.append((initial.tolist(), reset_info["raw_observation"]))
                for action in (1, 0):
                    current, reward, terminated, truncated, info = env.step(action)
                    infos.append(info)
                    trace.append((
                        current.tolist(), reward, terminated, truncated,
                        info["raw_observation"],
                    ))
            finally:
                env.close()
                thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            return trace, infos

        strict_trace, strict_infos = run_trace(
            strict_validation=True, info_mode="full"
        )
        fast_trace, fast_infos = run_trace(
            strict_validation=False, info_mode="episode"
        )
        self.assertEqual(strict_trace, fast_trace)
        self.assertIn("task_id", strict_infos[1])
        self.assertNotIn("task_id", fast_infos[1])
        self.assertEqual(fast_infos[2]["task_id"], "forward_and_target")

    def test_reset_step_and_close_use_task_schema(self) -> None:
        task = ForwardAndTargetTask()
        env = MinecraftEnv(
            task,
            port=0,
            connect_timeout=2.0,
            step_timeout=2.0,
            profile=True,
        )
        captured: list[dict[str, Any]] = []

        def fake_java() -> None:
            with socket.create_connection(env.address, timeout=2.0) as client:
                identify(client, "rl-0")
                stream = client.makefile("rb")
                reset = json.loads(stream.readline())
                captured.append(reset)
                client.sendall((json.dumps(observation("env_reset_result", 0)) + "\n").encode())

                action = json.loads(stream.readline())
                captured.append(action)
                client.sendall(
                    (json.dumps(observation("env_step_result", 0, x=1.75)) + "\n").encode()
                )

                close = json.loads(stream.readline())
                captured.append(close)
                client.sendall(b'{"type":"env_closed"}\n')

        client_thread = threading.Thread(target=fake_java, daemon=True)
        client_thread.start()
        try:
            initial, reset_info = env.reset(seed=123)
            self.assertTrue(env.observation_space.contains(initial))
            self.assertEqual(initial.tolist(), [1.0, 1.0])
            self.assertEqual(reset_info["episode_seed"], 123)
            self.assertTrue(reset_info["world_reset_applied"])
            self.assertEqual(reset_info["instance_id"], "rl-0")
            self.assertEqual(reset_info["client_world"], "void")
            self.assertEqual(reset_info["protocol_version"], PROTOCOL_VERSION)
            self.assertIn("raw_observation", reset_info)
            self.assertGreaterEqual(
                reset_info["training_profile"]["reset_ms"], 0.0
            )
            self.assertGreaterEqual(
                reset_info["training_profile"]["reset_bridge_ms"], 0.0
            )

            env.include_raw_info = False
            current, reward, terminated, truncated, info = env.step(1)
            self.assertTrue(env.observation_space.contains(current))
            self.assertAlmostEqual(reward, 0.75)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertNotIn("raw_observation", info)
            profile = info["training_profile"]
            self.assertEqual(profile["instance_id"], "rl-0")
            for key in (
                "env_step_ms",
                "action_encode_ms",
                "pace_ms",
                "send_ms",
                "bridge_wait_ms",
                "protocol_ms",
                "task_ms",
            ):
                self.assertGreaterEqual(profile[key], 0.0)
        finally:
            env.close()
            client_thread.join(timeout=2.0)

        self.assertFalse(client_thread.is_alive())
        self.assertEqual(captured[0]["task"], "forward_and_target")
        self.assertFalse(captured[0]["visualized"])
        self.assertEqual(captured[0]["action_fields"], "forward")
        self.assertEqual(captured[0]["observation_fields"], "player.x,raycast.type")
        self.assertEqual(
            captured[1],
            {"type": "env_action", "sequence": 0, "forward": True},
        )
        self.assertEqual(captured[2], {"type": "env_close"})

    def test_invalid_action_is_rejected_before_network_send(self) -> None:
        task = ForwardAndTargetTask()
        with self.assertRaises(ValueError):
            task.encode_action(2)

    def test_recoverable_reset_sync_error_is_retried_once(self) -> None:
        env = MinecraftEnv(
            ForwardAndTargetTask(), port=0, connect_timeout=2.0, step_timeout=2.0
        )
        resets: list[dict[str, Any]] = []

        def recovering_java() -> None:
            with socket.create_connection(env.address, timeout=2.0) as client:
                identify(client, "rl-0")
                stream = client.makefile("rb")
                resets.append(json.loads(stream.readline()))
                client.sendall(
                    b'{"type":"env_error","text":"multi-block reset did not reach the client: target"}\n'
                )
                resets.append(json.loads(stream.readline()))
                client.sendall(
                    (json.dumps(observation("env_reset_result", 0)) + "\n").encode()
                )
                self.assertEqual(json.loads(stream.readline()), {"type": "env_close"})
                client.sendall(b'{"type":"env_closed"}\n')

        thread = threading.Thread(target=recovering_java, daemon=True)
        thread.start()
        try:
            _, info = env.reset(seed=456)
            self.assertEqual(info["reset_retries"], 1)
        finally:
            env.close()
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(resets), 2)
        self.assertEqual(resets[0]["episode"], resets[1]["episode"])
        self.assertEqual(resets[0]["seed"], resets[1]["seed"])

    def test_target_tps_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_tps must be positive"):
            MinecraftEnv(ForwardAndTargetTask(), port=0, target_tps=0.0)

    def test_wrong_instance_is_rejected_during_handshake(self) -> None:
        env = MinecraftEnv(
            ForwardAndTargetTask(), port=0, instance_id="rl-1", connect_timeout=0.2
        )
        reset_errors: list[BaseException] = []

        def reset_environment() -> None:
            try:
                env.reset()
            except BaseException as error:
                reset_errors.append(error)

        reset_thread = threading.Thread(target=reset_environment, daemon=True)
        reset_thread.start()
        try:
            with socket.create_connection(env.address, timeout=1.0) as client:
                client.sendall(
                    (json.dumps({
                        "type": "hello",
                        "protocol_version": PROTOCOL_VERSION,
                        "instance_id": "rl-0",
                        "world": "void",
                    }) + "\n").encode("utf-8")
                )
                response = json.loads(client.makefile("rb").readline())
                self.assertEqual(response["type"], "env_error")
                self.assertIn("expected 'rl-1'", response["text"])
            reset_thread.join(timeout=1.0)
            self.assertFalse(reset_thread.is_alive())
            self.assertEqual(len(reset_errors), 1)
            self.assertIsInstance(reset_errors[0], TimeoutError)
        finally:
            env.close()

    def test_client_error_while_waiting_for_ready_is_reported(self) -> None:
        env = MinecraftEnv(
            ForwardAndTargetTask(), port=0, instance_id="rl-0", connect_timeout=1.0
        )
        release_client = threading.Event()

        def failing_java() -> None:
            with socket.create_connection(env.address, timeout=1.0) as client:
                hello = {
                    "type": "hello",
                    "protocol_version": PROTOCOL_VERSION,
                    "instance_id": "rl-0",
                    "world": "void",
                }
                client.sendall((json.dumps(hello) + "\n").encode("utf-8"))
                client.sendall(b'{"type":"client_error","text":"world missing"}\n')
                release_client.wait(timeout=2.0)

        thread = threading.Thread(target=failing_java, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(MinecraftProtocolError, "world missing"):
                env.reset()
        finally:
            release_client.set()
            env.close()
            thread.join(timeout=1.0)

    def test_task_rejects_unsupported_java_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported observation"):
            MinecraftTask.__init__(
                object.__new__(ForwardAndTargetTask),
                task_id="bad",
                action_space=spaces.Discrete(1),
                observation_space=spaces.Discrete(1),
                action_fields=(),
                observation_fields=("world.weather",),
            )


if __name__ == "__main__":
    unittest.main()
