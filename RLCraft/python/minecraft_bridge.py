#!/usr/bin/env python3
"""Small NDJSON-over-TCP bridge for the MinecraftRL Forge prototype."""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import threading
import time
from typing import Any


MAX_LINE_BYTES = 1_048_576


class BridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, BridgeHandler)
        self._clients: set[BridgeHandler] = set()
        self._clients_lock = threading.Lock()

    def add_client(self, client: "BridgeHandler") -> None:
        with self._clients_lock:
            self._clients.add(client)

    def remove_client(self, client: "BridgeHandler") -> None:
        with self._clients_lock:
            self._clients.discard(client)

    def broadcast(self, message: dict[str, Any]) -> int:
        with self._clients_lock:
            clients = list(self._clients)
        delivered = 0
        for client in clients:
            if client.send(message):
                delivered += 1
        return delivered

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    @property
    def client_instances(self) -> list[str]:
        with self._clients_lock:
            return sorted(client.instance_id for client in self._clients)


class BridgeHandler(socketserver.StreamRequestHandler):
    server: BridgeServer

    def setup(self) -> None:
        super().setup()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._write_lock = threading.Lock()
        self._sync_benchmark: dict[str, int] | None = None
        self.instance_id = "unidentified"
        self.server.add_client(self)
        print(f"[connected] Minecraft at {self.client_address[0]}:{self.client_address[1]}")

    def handle(self) -> None:
        while True:
            raw_line = self.rfile.readline(MAX_LINE_BYTES + 1)
            if not raw_line:
                return
            if len(raw_line) > MAX_LINE_BYTES:
                print("[warning] ignored oversized message")
                return
            try:
                message = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                print(f"[warning] invalid message: {error}")
                continue
            if self._handle_sync_benchmark(message, len(raw_line)):
                continue
            if message.get("type") == "hello":
                instance_id = message.get("instance_id")
                if isinstance(instance_id, str) and instance_id:
                    self.instance_id = instance_id
                print(
                    f"[identified] {self.instance_id} protocol="
                    f"{message.get('protocol_version')} world={message.get('world', '')}"
                )
            elif message.get("type") == "client_ready":
                print(f"[ready/{self.instance_id}] world={message.get('world', '')}")
            elif message.get("type") == "benchmark_echo":
                message_id = message.get("id")
                payload = message.get("payload")
                if isinstance(message_id, str) and isinstance(payload, str):
                    self.send({"type": "benchmark_echo_reply", "id": message_id, "payload": payload})
                else:
                    print("[warning] invalid benchmark echo")
            elif message.get("type") == "message":
                print(f"[Minecraft/{self.instance_id}] {message.get('text', '')}")
            else:
                print(f"[Minecraft/{self.instance_id}/raw] {message}")

    def _handle_sync_benchmark(self, message: dict[str, Any], wire_bytes: int) -> bool:
        message_type = message.get("type")
        if message_type == "sync_benchmark_start":
            steps = message.get("steps")
            if not isinstance(steps, int) or not 1 <= steps <= 100_000:
                self.send({"type": "sync_benchmark_error", "text": "invalid step count"})
                return True
            self._sync_benchmark = {
                "steps": steps,
                "sequence": 0,
                "started_ns": time.perf_counter_ns(),
                "observation_bytes": 0,
            }
            print(f"[sync benchmark] starting {steps} causal RL steps")
            self._send_sync_action(0)
            return True

        if message_type == "sync_observation":
            state = self._sync_benchmark
            if state is None:
                self.send({"type": "sync_benchmark_error", "text": "observation without active benchmark"})
                return True
            sequence = message.get("sequence")
            if sequence != state["sequence"]:
                self.send({"type": "sync_benchmark_error", "text": "unexpected observation sequence"})
                self._sync_benchmark = None
                return True
            state["observation_bytes"] += wire_bytes
            state["sequence"] += 1
            if state["sequence"] < state["steps"]:
                self._send_sync_action(state["sequence"])
            else:
                elapsed_ns = time.perf_counter_ns() - state["started_ns"]
                steps_per_second = state["steps"] / (elapsed_ns / 1_000_000_000)
                self.send({
                    "type": "sync_benchmark_complete",
                    "steps": state["steps"],
                    "elapsed_nanos": elapsed_ns,
                    "observation_bytes": state["observation_bytes"],
                })
                print(f"[sync benchmark] complete: {steps_per_second:.1f} step/s")
                self._sync_benchmark = None
            return True

        if message_type == "sync_benchmark_abort":
            print(f"[sync benchmark] aborted by Minecraft: {message.get('text', '')}")
            self._sync_benchmark = None
            return True
        return False

    def _send_sync_action(self, sequence: int) -> None:
        # Alternating raw mouse deltas exercise vanilla Entity.setAngles without drifting far.
        mouse_dx = 1.0 if sequence % 2 == 0 else -1.0
        self.send({
            "type": "sync_action",
            "sequence": sequence,
            "mouse_dx": mouse_dx,
            "mouse_dy": 0.0,
            "forward": False,
            "back": False,
            "left": False,
            "right": False,
            "jump": False,
            "attack": False,
        })

    def finish(self) -> None:
        self.server.remove_client(self)
        print(
            f"[disconnected] {self.instance_id} at "
            f"{self.client_address[0]}:{self.client_address[1]}"
        )
        super().finish()

    def send(self, message: dict[str, Any]) -> bool:
        payload = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                self.wfile.write(payload)
                self.wfile.flush()
            return True
        except (ConnectionError, OSError):
            return False


def parse_console_line(line: str) -> dict[str, Any] | None:
    words = line.strip().split()
    if not words:
        return None
    if words[0] == "/step":
        ticks = int(words[1]) if len(words) > 1 else 1
        return {"type": "control", "action": "step", "ticks": max(1, min(ticks, 100_000))}
    if words[0] == "/mode" and len(words) == 2:
        mode = words[1].lower()
        if mode not in {"vanilla", "fast", "step"}:
            raise ValueError("mode must be vanilla, fast, or step")
        return {"type": "control", "action": "mode", "mode": mode}
    return {"type": "message", "text": line}


def console_loop(server: BridgeServer) -> None:
    print("Type text, /step [n], /mode <vanilla|fast|step>, /clients, or /quit.")
    while True:
        try:
            line = input("python> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() == "/quit":
            break
        if line.strip() == "/clients":
            print(f"{server.client_count} client(s): {', '.join(server.client_instances)}")
            continue
        try:
            message = parse_console_line(line)
        except (ValueError, IndexError) as error:
            print(f"[error] {error}")
            continue
        if message is not None:
            print(f"[sent to {server.broadcast(message)} client(s)]")
    server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    with BridgeServer((args.host, args.port)) as server:
        network_thread = threading.Thread(target=server.serve_forever, name="bridge-network", daemon=True)
        network_thread.start()
        print(f"MinecraftRL bridge listening on {args.host}:{args.port}")
        console_loop(server)
        network_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
