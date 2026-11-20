import subprocess
import sys
from pathlib import Path
import psutil
from benchmark_matrix import launch_client, stop_clients, DEFAULT_CONFIG

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "benchmarks/optimization_fixture"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "check.log").open("w", encoding="utf-8") as log:
        check = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(root / "python/optimization_fixture_check.py"),
                "--output",
                str(output / "results.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        client = launch_client(
            root.parent / "HeadlessMC",
            0,
            {**DEFAULT_CONFIG, "idle_wait_mode": "event"},
            output,
        )
        owned = []
        try:
            check.wait(timeout=240)
            print("Fixture check exit", check.returncode, flush=True)
        finally:
            for process in [check, client[0]]:
                try:
                    parent = psutil.Process(process.pid)
                    owned.extend([parent, *parent.children(recursive=True)])
                except psutil.Error:
                    pass
            stop_clients([client])
            for process in owned:
                try:
                    if process.is_running():
                        process.terminate()
                except psutil.Error:
                    pass
            _, alive = psutil.wait_procs(owned, timeout=5)
            for process in alive:
                try:
                    process.kill()
                except psutil.Error:
                    pass

    raise SystemExit(check.returncode)
