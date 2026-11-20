#!/usr/bin/env python3
"""Gymnasium environment and task template for the MinecraftRL Forge client."""

from __future__ import annotations

import json
import re
import socket
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np


MAX_LINE_BYTES = 1_048_576
MAX_TASK_BLOCKS = 5
PROTOCOL_VERSION = 1
INSTANCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
RESET_RETRY_ERROR_PREFIXES = (
    "multi-block reset did not reach the client:",
    "environment reset block resync failed:",
    "environment reset state did not reach the client:",
)


def _task_block_observation_fields() -> set[str]:
    attributes = (
        "x", "y", "z", "distance", "yaw_delta", "pitch_delta", "block_id",
        "tool_speed_0", "tool_speed_1", "tool_speed_2",
    )
    return {
        f"task_block_{index}.{attribute}"
        for index in range(MAX_TASK_BLOCKS)
        for attribute in attributes
    }


class MinecraftProtocolError(RuntimeError):
    """Raised when Java disconnects or violates the environment protocol."""


@dataclass(frozen=True)
class _PendingStep:
    """Timing/state captured after an action has been dispatched.

    Splitting dispatch from receive lets a vector environment put every Minecraft
    instance in flight before waiting for the slowest one.  The ordinary Gym
    ``step`` method still exposes exactly the same synchronous contract.
    """

    step_started: float
    encoded_at: float
    paced_at: float
    sent_at: float


class MinecraftTask(ABC):
    """Per-task Gym spaces and the exact Java data contract they require."""

    SUPPORTED_ACTION_FIELDS = frozenset(
        {
            "mouse_dx", "mouse_dy", "forward", "back", "left", "right", "jump",
            "attack", "sneak", "sprint", "use", "attack_once", "use_once",
            "hotbar_slot",
        }
    )
    SUPPORTED_OBSERVATION_FIELDS = frozenset(
        {
            "player.x",
            "player.y",
            "player.z",
            "player.yaw",
            "player.pitch",
            "player.health",
            "player.on_ground",
            "player.motion_x", "player.motion_y", "player.motion_z",
            "player.sneaking", "player.sprinting", "player.using_item",
            "player.item_use_ticks", "player.selected_slot",
            "player.collided_horizontally", "player.collided_vertically",
            "player.fall_distance", "player.block_break_progress",
            "held.item_id", "held.item_name", "held.count", "held.damage", "held.max_damage",
            "inventory.arrow_count", "inventory.hotbar",
            "raycast.type", "raycast.name", "raycast.id", "raycast.x", "raycast.y",
            "raycast.z", "raycast.side", "raycast.distance", "raycast.health",
            "raycast.block_hardness", "raycast.can_harvest",
            "raycast.category", "raycast.safe_yaw_correction",
            "raycast.safe_pitch_correction",
            "nearest_living.type", "nearest_living.id", "nearest_living.x",
            "nearest_living.y", "nearest_living.z", "nearest_living.relative_x",
            "nearest_living.relative_y", "nearest_living.relative_z",
            "nearest_living.distance", "nearest_living.yaw_delta",
            "nearest_living.pitch_delta", "nearest_living.motion_x",
            "nearest_living.motion_y", "nearest_living.motion_z",
            "nearest_living.health", "nearest_living.alive",
            "task_entity.type", "task_entity.id", "task_entity.x", "task_entity.y",
            "task_entity.z", "task_entity.relative_x", "task_entity.relative_y",
            "task_entity.relative_z", "task_entity.distance", "task_entity.yaw_delta",
            "task_entity.pitch_delta", "task_entity.motion_x", "task_entity.motion_y",
            "task_entity.motion_z", "task_entity.health", "task_entity.alive",
            "task_goal.x", "task_goal.y", "task_goal.z", "task_goal.relative_x",
            "task_goal.relative_y", "task_goal.relative_z", "task_goal.distance",
            "task_goal.horizontal_distance", "task_goal.yaw_delta", "task_goal.pitch_delta",
            "task_goal.block_id", "task_goal.tool_speed_0", "task_goal.tool_speed_1",
            "task_goal.tool_speed_2",
            "local.blocks_3x3x3", "task_blocks.count",
            "bed_defense.origin_x", "bed_defense.origin_y",
            "bed_defense.origin_z", "bed_defense.block_grid",
            "bed_defense.visible_mask", "bed_defense_v2.block_grid",
            "bed_defense_v2.visible_mask",
            "bridging.ground_fraction", "bridging.distance_forward_edge",
            "bridging.distance_left_edge", "bridging.distance_right_edge",
            "bridging.distance_rear_edge", "bridging.can_place",
            "bridging.block_grid", "bridging.above_level_count",
        }
    ).union(_task_block_observation_fields())

    def __init__(
        self,
        *,
        task_id: str,
        action_space: gym.Space[Any],
        observation_space: gym.Space[Any],
        action_fields: Sequence[str],
        observation_fields: Sequence[str],
        max_episode_steps: int = 1_000,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self._validate_fields("action", action_fields, self.SUPPORTED_ACTION_FIELDS)
        self._validate_fields(
            "observation", observation_fields, self.SUPPORTED_OBSERVATION_FIELDS
        )
        self.task_id = task_id
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_fields = tuple(action_fields)
        self.observation_fields = tuple(observation_fields)
        self.max_episode_steps = max_episode_steps

    @staticmethod
    def _validate_fields(kind: str, fields: Sequence[str], supported: frozenset[str]) -> None:
        if len(fields) != len(set(fields)):
            raise ValueError(f"duplicate {kind} fields are not allowed")
        unsupported = set(fields) - supported
        if unsupported:
            raise ValueError(f"unsupported {kind} fields: {sorted(unsupported)}")

    def reset_payload(self, seed: int, options: Mapping[str, Any]) -> dict[str, Any]:
        """Extra JSON fields for a future task-specific deterministic reset."""
        return {}

    def set_training_mode(self, enabled: bool) -> None:
        """Let a task enable training-only behavior such as a curriculum."""

    def set_training_progress(self, timesteps: int) -> None:
        """Restore task-local curriculum progress when resuming training."""

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Select a mastery-gated curriculum stage when the task defines one."""

    def set_curriculum_validation_mode(self, enabled: bool) -> None:
        """Select active-stage-only resets for curriculum validation."""

    def set_curriculum_rehearsal(
        self, percent: int, focus_stage: int = -1
    ) -> None:
        """Configure training-only rehearsal sampling when supported."""

    def on_reset(self, initial: Mapping[str, Any]) -> None:
        """Reset task-local reward state."""

    @abstractmethod
    def encode_action(self, action: Any) -> dict[str, Any]:
        """Validate/decode one task action into only the declared Java primitives."""

    @abstractmethod
    def observation(self, raw: Mapping[str, Any]) -> Any:
        """Convert the selected raw Java fields to this task's Gym observation."""

    def reward(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> float:
        return 0.0

    def terminated(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> bool:
        return False

    def truncated(self, step_count: int, current: Mapping[str, Any]) -> bool:
        return step_count >= self.max_episode_steps

    def info(
        self, raw: Mapping[str, Any], *, episode_end: bool = True
    ) -> dict[str, Any]:
        """Return task metadata; expensive diagnostics belong at episode boundaries."""
        return {"task_id": self.task_id} if episode_end else {}

    def describe_action(self, action: Any) -> str:
        """Human-readable action for generic evaluation logs."""
        return repr(action)

    def describe_reset(self, raw: Mapping[str, Any]) -> str:
        """Human-readable initial state for generic evaluation logs."""
        return f"task={self.task_id}"


class _EnvironmentConnection:
    """One synchronous localhost connection owned by its Gym worker process."""

    def __init__(self, address: tuple[str, int], expected_instance_id: str) -> None:
        self.expected_instance_id = expected_instance_id
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(address)
        self._listener.listen(1)
        self._client: socket.socket | None = None
        self._reader: Any | None = None
        self._client_info: dict[str, Any] | None = None
        self._epoch = 0
        self.socket_read_ms = 0.0
        self.json_decode_ms = 0.0

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._listener.getsockname()
        return str(host), int(port)

    def _drop_client(self) -> None:
        reader, client = self._reader, self._client
        self._reader = None
        self._client = None
        self._client_info = None
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        if client is not None:
            try:
                client.close()
            except OSError:
                pass

    @staticmethod
    def _validate_hello(
        hello: Mapping[str, Any], expected_instance_id: str
    ) -> str | None:
        if hello.get("type") != "hello":
            return "the first message must be a hello"
        if type(hello.get("protocol_version")) is not int:
            return "hello.protocol_version must be an integer"
        if hello["protocol_version"] != PROTOCOL_VERSION:
            return (
                f"protocol version mismatch: expected {PROTOCOL_VERSION}, "
                f"received {hello['protocol_version']}"
            )
        if hello.get("instance_id") != expected_instance_id:
            return (
                f"instance mismatch: expected {expected_instance_id!r}, "
                f"received {hello.get('instance_id')!r}"
            )
        if not isinstance(hello.get("world"), str):
            return "hello.world must be a string"
        return None

    def wait_connected(self, timeout: float) -> int:
        if self._client is not None:
            return self._epoch
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("timed out waiting for the Minecraft client to connect")
            self._listener.settimeout(remaining)
            try:
                client, _ = self._listener.accept()
            except socket.timeout as error:
                raise TimeoutError(
                    "timed out waiting for the Minecraft client to connect"
                ) from error
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._client = client
            self._reader = client.makefile("rb")
            try:
                hello = self._read_message(deadline)
            except (MinecraftProtocolError, TimeoutError):
                self._drop_client()
                continue
            error = self._validate_hello(hello, self.expected_instance_id)
            if error is not None:
                try:
                    self._send_unchecked({"type": "env_error", "text": error})
                finally:
                    self._drop_client()
                continue
            self._epoch += 1
            self._client_info = dict(hello)
            return self._epoch

    def client_info(self, epoch: int) -> dict[str, Any]:
        if self._client is None or epoch != self._epoch:
            raise MinecraftProtocolError("Minecraft client disconnected")
        if self._client_info is None:
            raise MinecraftProtocolError("Minecraft client did not identify itself")
        return dict(self._client_info)

    def _send_unchecked(self, message: Mapping[str, Any]) -> None:
        if self._client is None:
            raise MinecraftProtocolError("Minecraft client disconnected")
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self._client.sendall(payload)
        except (ConnectionError, OSError) as error:
            self._drop_client()
            raise MinecraftProtocolError("failed to send message to Minecraft") from error

    def send(self, epoch: int, message: Mapping[str, Any]) -> None:
        if self._client is None or epoch != self._epoch:
            raise MinecraftProtocolError("Minecraft client disconnected")
        self._send_unchecked(message)

    def _read_message(self, deadline: float) -> dict[str, Any]:
        if self._client is None or self._reader is None:
            raise MinecraftProtocolError("Minecraft client disconnected")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("timed out waiting for Minecraft")
            self._client.settimeout(remaining)
            try:
                read_started = time.perf_counter()
                raw_line = self._reader.readline(MAX_LINE_BYTES + 1)
                self.socket_read_ms += (time.perf_counter() - read_started) * 1000.0
            except socket.timeout as error:
                raise TimeoutError("timed out waiting for Minecraft") from error
            except (ConnectionError, OSError) as error:
                self._drop_client()
                raise MinecraftProtocolError("Minecraft client disconnected") from error
            if not raw_line or len(raw_line) > MAX_LINE_BYTES:
                self._drop_client()
                raise MinecraftProtocolError("Minecraft client disconnected")
            try:
                decode_started = time.perf_counter()
                message = json.loads(raw_line.decode("utf-8"))
                self.json_decode_ms += (time.perf_counter() - decode_started) * 1000.0
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(message, dict):
                return message

    def receive(self, epoch: int, expected_type: str, timeout: float) -> dict[str, Any]:
        if self._client is None or epoch != self._epoch:
            raise MinecraftProtocolError("Minecraft client disconnected")
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self._read_message(deadline)
            except TimeoutError as error:
                raise TimeoutError(f"timed out waiting for {expected_type}") from error
            message_type = message.get("type")
            if message_type in {"env_error", "client_error"}:
                raise MinecraftProtocolError(str(message.get("text", "unknown Java error")))
            if message_type == expected_type:
                return message
            if message_type in {"message", "client_ready"}:
                continue
            raise MinecraftProtocolError(
                f"expected {expected_type}, received {message_type or '<missing type>'}"
            )

    def close_client(self) -> None:
        self._drop_client()

    def close(self) -> None:
        self._drop_client()
        self._listener.close()


class MinecraftEnv(gym.Env[Any, Any]):
    """One persistent Minecraft client exposed as a synchronous Gymnasium Env."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: MinecraftTask,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        connect_timeout: float = 120.0,
        step_timeout: float = 30.0,
        visualized: bool = False,
        target_tps: float = 20.0,
        instance_id: str = "rl-0",
        profile: bool = False,
        profile_java_phases: bool = False,
        include_raw_info: bool = True,
        strict_validation: bool = True,
        info_mode: str = "full",
    ) -> None:
        super().__init__()
        if target_tps <= 0.0:
            raise ValueError("target_tps must be positive")
        if not INSTANCE_ID_PATTERN.fullmatch(instance_id):
            raise ValueError(
                "instance_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"
            )
        if info_mode not in {"episode", "full"}:
            raise ValueError("info_mode must be 'episode' or 'full'")
        self.task = task
        self.action_space = self.task.action_space
        self.observation_space = self.task.observation_space
        self.connect_timeout = connect_timeout
        self.step_timeout = step_timeout
        self.visualized = bool(visualized)
        self.target_tps = float(target_tps)
        self.instance_id = instance_id
        self.profile = bool(profile)
        self.profile_java_phases = bool(profile_java_phases)
        self.include_raw_info = bool(include_raw_info)
        self.strict_validation = bool(strict_validation)
        self.info_mode = info_mode
        self._step_interval = 1.0 / self.target_tps
        self._server = _EnvironmentConnection((host, port), instance_id)
        self._epoch: int | None = None
        self._active = False
        self._closed = False
        self._episode = 0
        self._next_sequence = 0
        self._step_count = 0
        self._previous_raw: dict[str, Any] | None = None
        self._next_step_deadline: float | None = None
        self._pending_step: _PendingStep | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address
        return str(host), int(port)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        reset_started = time.perf_counter()
        if self._pending_step is not None:
            raise RuntimeError("cannot reset while an environment step is pending")
        super().reset(seed=seed)
        self._require_open()
        reset_options = options or {}
        episode_seed = int(seed) if seed is not None else int(
            self.np_random.integers(0, 2**63 - 1)
        )
        connected_epoch = self._server.wait_connected(self.connect_timeout)
        if connected_epoch != self._epoch:
            ready = self._server.receive(
                connected_epoch, "client_ready", self.connect_timeout
            )
            if ready.get("instance_id") != self.instance_id:
                raise MinecraftProtocolError(
                    f"ready message came from {ready.get('instance_id')!r}, "
                    f"expected {self.instance_id!r}"
                )
            self._epoch = connected_epoch
        client_info = self._server.client_info(self._epoch)
        connected_at = time.perf_counter()
        self._episode += 1
        request: dict[str, Any] = {
            "type": "env_reset",
            "task": self.task.task_id,
            "episode": self._episode,
            "seed": episode_seed,
            "visualized": self.visualized,
            "action_fields": ",".join(self.task.action_fields),
            "observation_fields": ",".join(self.task.observation_fields),
            "payload": "profile_java_phases" if self.profile_java_phases else "",
        }
        extra_reset_fields = self.task.reset_payload(episode_seed, reset_options)
        reserved_fields = request.keys() & extra_reset_fields.keys()
        if reserved_fields:
            raise ValueError(
                f"reset_payload must not override protocol fields: {sorted(reserved_fields)}"
            )
        request.update(extra_reset_fields)
        reset_retries = 0
        bridge_started = time.perf_counter()
        for attempt in range(2):
            self._server.send(self._epoch, request)
            try:
                raw = self._server.receive(
                    self._epoch, "env_reset_result", self.step_timeout
                )
                break
            except MinecraftProtocolError as error:
                recoverable = str(error).startswith(RESET_RETRY_ERROR_PREFIXES)
                if attempt > 0 or not recoverable:
                    raise
                reset_retries += 1
                # Java has already aborted the failed synchronized session. Give
                # its client loop a brief chance to return to IDLE before sending
                # the same deterministic reset again on the existing connection.
                time.sleep(0.05)
        bridge_finished = time.perf_counter()
        self._validate_sequence(raw, 0)
        self._validate_raw_fields(raw, reset=True)
        validated_at = time.perf_counter()

        self._active = True
        self._next_sequence = 0
        self._step_count = 0
        self._previous_raw = raw
        self._next_step_deadline = time.monotonic() if self.visualized else None
        self.task.on_reset(raw)
        observation = self.task.observation(raw)
        self._validate_observation(observation)
        info = self.task.info(raw, episode_end=True)
        if self.include_raw_info:
            info["raw_observation"] = dict(raw)
        info.update(
            {
                "episode": self._episode,
                "episode_seed": episode_seed,
                "world_reset_applied": bool(raw["world_reset_applied"]),
                "visualized": self.visualized,
                "target_tps": self.target_tps if self.visualized else None,
                "instance_id": self.instance_id,
                "client_world": client_info["world"],
                "protocol_version": client_info["protocol_version"],
                "reset_retries": reset_retries,
            }
        )
        if self.profile:
            reset_finished = time.perf_counter()
            info["training_profile"] = {
                "reset_ms": (reset_finished - reset_started) * 1000.0,
                "reset_connect_ms": (connected_at - reset_started) * 1000.0,
                "reset_bridge_ms": (bridge_finished - bridge_started) * 1000.0,
                "reset_protocol_ms": (validated_at - bridge_finished) * 1000.0,
                "reset_task_ms": (reset_finished - validated_at) * 1000.0,
            }
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.begin_step(action)
        return self.finish_step()

    def begin_step(self, action: Any) -> None:
        """Validate, encode, and send one action without waiting for its result."""
        step_started = time.perf_counter()
        self._require_open()
        if self._pending_step is not None:
            raise RuntimeError("begin_step() called while another step is pending")
        if not self._active or self._epoch is None or self._previous_raw is None:
            raise RuntimeError("reset() must complete before step()")
        if self.strict_validation and not self.action_space.contains(action):
            raise ValueError(f"action is outside {self.action_space}: {action!r}")
        encoded_action = self.task.encode_action(action)
        undeclared_fields = (
            set(encoded_action) - set(self.task.action_fields)
            if self.strict_validation else set()
        )
        if undeclared_fields:
            raise ValueError(
                f"encode_action returned undeclared fields: {sorted(undeclared_fields)}"
            )
        encoded_at = time.perf_counter()
        self._pace_visualized_step()
        paced_at = time.perf_counter()
        message = {
            "type": "env_action",
            "sequence": self._next_sequence,
            **encoded_action,
        }
        self._server.send(self._epoch, message)
        sent_at = time.perf_counter()
        self._pending_step = _PendingStep(
            step_started=step_started,
            encoded_at=encoded_at,
            paced_at=paced_at,
            sent_at=sent_at,
        )

    def finish_step(self) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Wait for and process the result of the action sent by ``begin_step``."""
        pending = self._pending_step
        if pending is None:
            raise RuntimeError("finish_step() called without a pending step")
        if self._epoch is None or self._previous_raw is None:
            self._pending_step = None
            raise RuntimeError("environment became inactive during a pending step")
        self._server.socket_read_ms = self._server.json_decode_ms = 0.0
        raw = self._server.receive(self._epoch, "env_step_result", self.step_timeout)
        received_at = time.perf_counter()
        self._validate_sequence(raw, self._next_sequence)
        if self.strict_validation:
            self._validate_raw_fields(raw)
        validated_at = time.perf_counter()

        self._next_sequence += 1
        self._step_count += 1
        reward = self.task.reward(self._previous_raw, raw)
        terminated = self.task.terminated(self._previous_raw, raw)
        truncated = self.task.truncated(self._step_count, raw)
        # Reward calculation may advance task-local state (for example, the
        # designated target in a curriculum). Build the returned observation
        # afterwards so it always describes the state used for the next action.
        observation = self.task.observation(raw)
        if self.strict_validation:
            self._validate_observation(observation)
        self._previous_raw = raw
        episode_end = bool(terminated or truncated)
        info = self.task.info(
            raw, episode_end=episode_end or self.info_mode == "full"
        )
        if self.include_raw_info:
            info["raw_observation"] = dict(raw)
        info["step"] = self._step_count
        if self.profile:
            step_finished = time.perf_counter()
            info["training_profile"] = {
                "instance_id": self.instance_id,
                "env_step_ms": (step_finished - pending.step_started) * 1000.0,
                "action_encode_ms": (pending.encoded_at - pending.step_started)
                * 1000.0,
                "pace_ms": (pending.paced_at - pending.encoded_at) * 1000.0,
                "send_ms": (pending.sent_at - pending.paced_at) * 1000.0,
                "bridge_wait_ms": (received_at - pending.sent_at) * 1000.0,
                "protocol_ms": (validated_at - received_at) * 1000.0,
                "socket_read_ms": self._server.socket_read_ms,
                "json_decode_ms": self._server.json_decode_ms,
                "task_ms": (step_finished - validated_at) * 1000.0,
            }
            for field in (
                "action_client_ms",
                "server_tick_ms",
                "post_client_ms",
                "observation_serialize_ms",
                "java_total_ms",
            ):
                key = f"java_profile.{field}"
                if key in raw:
                    info["training_profile"][field] = float(raw[key])
        self._pending_step = None
        return observation, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        had_pending_step = self._pending_step is not None
        self._pending_step = None
        if self._active and self._epoch is not None and not had_pending_step:
            try:
                self._server.send(self._epoch, {"type": "env_close"})
                self._server.receive(self._epoch, "env_closed", min(self.step_timeout, 2.0))
            except (MinecraftProtocolError, TimeoutError):
                pass
        self._active = False
        self._next_step_deadline = None
        self._server.close_client()
        self._server.close()

    def set_training_progress(self, timesteps: int) -> None:
        """Synchronize task-local curricula from the SB3 global timestep count."""
        self.task.set_training_progress(timesteps)

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Synchronize mastery-gated curricula across vector workers."""
        self.task.set_curriculum_stage(stage_index)

    def set_curriculum_validation_mode(self, enabled: bool) -> None:
        """Temporarily disable task rehearsal during validation."""
        self.task.set_curriculum_validation_mode(enabled)

    def set_curriculum_rehearsal(
        self, percent: int, focus_stage: int = -1
    ) -> None:
        """Configure adaptive curriculum rehearsal across vector workers."""
        self.task.set_curriculum_rehearsal(percent, focus_stage)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")

    def _pace_visualized_step(self) -> None:
        if not self.visualized:
            return
        now = time.monotonic()
        deadline = self._next_step_deadline
        if deadline is not None and deadline > now:
            time.sleep(deadline - now)
        # Schedule from the actual action start so a slow frame or model decision
        # never causes a burst of catch-up ticks.
        self._next_step_deadline = time.monotonic() + self._step_interval

    @staticmethod
    def _validate_sequence(message: Mapping[str, Any], expected: int) -> None:
        if message.get("sequence") != expected:
            raise MinecraftProtocolError(
                f"expected sequence {expected}, received {message.get('sequence')!r}"
            )

    def _validate_observation(self, observation: Any) -> None:
        if not self.observation_space.contains(observation):
            diagnostics: list[str] = []
            if isinstance(self.observation_space, gym.spaces.Dict) and isinstance(
                observation, Mapping
            ):
                for key, field_space in self.observation_space.spaces.items():
                    if key not in observation:
                        diagnostics.append(f"{key}: missing")
                        continue
                    value = np.asarray(observation[key])
                    if value.shape != field_space.shape:
                        diagnostics.append(
                            f"{key}: shape={value.shape}, expected={field_space.shape}"
                        )
                        continue
                    if isinstance(field_space, gym.spaces.Box):
                        invalid = (
                            ~np.isfinite(value)
                            | (value < field_space.low)
                            | (value > field_space.high)
                        )
                        for index_array in np.argwhere(invalid)[:8]:
                            index = tuple(int(part) for part in index_array)
                            diagnostics.append(
                                f"{key}{index}: value={value[index]!r}, "
                                f"range=[{field_space.low[index]!r}, "
                                f"{field_space.high[index]!r}]"
                            )
                        if value.dtype != field_space.dtype:
                            diagnostics.append(
                                f"{key}: dtype={value.dtype}, "
                                f"expected={field_space.dtype}"
                            )
            detail = "; ".join(diagnostics) or "no field-level diagnostic"
            raise MinecraftProtocolError(
                f"task produced an observation outside {self.observation_space}; "
                f"violations: {detail}"
            )

    def _validate_raw_fields(
        self, raw: Mapping[str, Any], *, reset: bool = False
    ) -> None:
        expected = {"type", "sequence", *self.task.observation_fields}
        if reset:
            expected.add("world_reset_applied")
        actual = set(raw)
        permitted_profile_fields = {
            key for key in actual if key.startswith("java_profile.")
        }
        if actual - permitted_profile_fields != expected:
            raise MinecraftProtocolError(
                f"Java observation fields differ from task schema; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected - permitted_profile_fields)}"
            )


__all__ = [
    "MinecraftEnv",
    "MinecraftProtocolError",
    "MinecraftTask",
    "PROTOCOL_VERSION",
]
