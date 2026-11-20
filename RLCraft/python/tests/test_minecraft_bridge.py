import json
import socket
import threading
import time
import unittest

from minecraft_bridge import BridgeServer, parse_console_line


class ProtocolTest(unittest.TestCase):
    def test_console_commands(self) -> None:
        self.assertEqual(
            parse_console_line("/step 3"),
            {"type": "control", "action": "step", "ticks": 3},
        )
        self.assertEqual(
            parse_console_line("héllo Minecraft"),
            {"type": "message", "text": "héllo Minecraft"},
        )

    def test_server_can_receive_and_send_ndjson(self) -> None:
        with BridgeServer(("127.0.0.1", 0)) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            with socket.create_connection(server.server_address, timeout=1.0) as client:
                client.settimeout(1.0)
                client.sendall(
                    b'{"type":"hello","protocol_version":1,'
                    b'"instance_id":"rl-0","world":"void"}\n'
                )
                client.sendall(b'{"type":"message","text":"from Java"}\n')
                deadline = time.monotonic() + 1.0
                while server.client_count != 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(server.client_count, 1)
                deadline = time.monotonic() + 1.0
                while server.client_instances != ["rl-0"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(server.client_instances, ["rl-0"])
                self.assertEqual(
                    server.broadcast({"type": "message", "text": "depuis Python"}),
                    1,
                )
                received = client.makefile("rb").readline()
                self.assertEqual(json.loads(received), {"type": "message", "text": "depuis Python"})
            server.shutdown()
            thread.join(timeout=1.0)

    def test_server_echoes_benchmark_payload(self) -> None:
        with BridgeServer(("127.0.0.1", 0)) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            with socket.create_connection(server.server_address, timeout=1.0) as client:
                client.settimeout(1.0)
                request = {"type": "benchmark_echo", "id": "test-1", "payload": "x" * 65_536}
                client.sendall((json.dumps(request) + "\n").encode("utf-8"))
                received = json.loads(client.makefile("rb").readline())
                self.assertEqual(received["type"], "benchmark_echo_reply")
                self.assertEqual(received["id"], "test-1")
                self.assertEqual(received["payload"], request["payload"])
            server.shutdown()
            thread.join(timeout=1.0)

    def test_synchronized_benchmark_driver(self) -> None:
        with BridgeServer(("127.0.0.1", 0)) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            with socket.create_connection(server.server_address, timeout=1.0) as client:
                client.settimeout(1.0)
                stream = client.makefile("rb")
                client.sendall(b'{"type":"sync_benchmark_start","steps":2}\n')
                first_action = json.loads(stream.readline())
                self.assertEqual((first_action["type"], first_action["sequence"]), ("sync_action", 0))
                client.sendall(b'{"type":"sync_observation","sequence":0,"target_type":"BLOCK"}\n')
                second_action = json.loads(stream.readline())
                self.assertEqual((second_action["type"], second_action["sequence"]), ("sync_action", 1))
                client.sendall(b'{"type":"sync_observation","sequence":1,"target_type":"ENTITY"}\n')
                complete = json.loads(stream.readline())
                self.assertEqual(complete["type"], "sync_benchmark_complete")
                self.assertEqual(complete["steps"], 2)
                self.assertGreater(complete["observation_bytes"], 0)
            server.shutdown()
            thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
