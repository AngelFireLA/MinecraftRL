"""Opt-in inclusive wall timers around existing SB3 operations (no algorithm fork)."""

from collections import defaultdict
from time import perf_counter
import psutil


class PhaseTimers:
    def __init__(self, model, vector):
        self.elapsed = defaultdict(float)
        self.resources = {}
        parent = psutil.Process()
        self.processes = [(parent, "trainer")] + [
            (child, "worker") for child in parent.children(recursive=True)
        ]
        for process in psutil.process_iter(["name", "cmdline"]):
            if (
                process.info["name"]
                and process.info["name"].lower().startswith("java")
                and any(
                    "-Dminecraftrl.instanceId=" in arg
                    for arg in (process.info["cmdline"] or [])
                )
            ):
                self.processes.append((process, "minecraft"))
        self.restore = []
        excluded_was_local = "_excluded_save_params" in model.__dict__
        excluded = model._excluded_save_params
        model._excluded_save_params = lambda: [
            *excluded(),
            "train",
            "collect_rollouts",
            "_excluded_save_params",
        ]
        self.restore.append(
            (model, "_excluded_save_params", excluded, excluded_was_local)
        )
        self.wrap(model.policy, "forward", "policy_inference_ms")
        self.wrap(model.policy, "predict_values", "value_inference_ms")
        self.wrap(vector, "step_async", "dispatch_ms")
        self.wrap(vector, "step_wait", "receive_ms")
        self.wrap(model.rollout_buffer, "add", "buffer_add_ms")
        self.wrap(
            model.rollout_buffer, "compute_returns_and_advantage", "advantages_ms"
        )
        self.wrap(model.rollout_buffer, "_get_samples", "minibatch_materialize_ms")
        self.wrap(model, "train", "ppo_train_ms")
        self.wrap(model, "collect_rollouts", "collect_total_ms")

    def wrap(self, owner, name, label):
        was_local = name in owner.__dict__
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            resource_start = (
                self.snapshot()
                if label in ("ppo_train_ms", "collect_total_ms")
                else None
            )
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.elapsed[label] += (perf_counter() - started) * 1000
                if resource_start is not None:
                    end = self.snapshot()
                    self.resources[label] = {
                        role: {
                            "cpu_seconds": values["cpu_seconds"]
                            - resource_start.get(role, {}).get("cpu_seconds", 0),
                            "rss_bytes": values["rss_bytes"],
                            "private_bytes": values["private_bytes"],
                        }
                        for role, values in end.items()
                    }

        setattr(owner, name, timed)
        self.restore.append((owner, name, original, was_local))

    def snapshot(self):
        result = {}
        for process, role in self.processes:
            try:
                cpu = process.cpu_times()
                memory = process.memory_info()
                row = result.setdefault(
                    role, {"cpu_seconds": 0, "rss_bytes": 0, "private_bytes": 0}
                )
                row["cpu_seconds"] += cpu.user + cpu.system
                row["rss_bytes"] += memory.rss
                row["private_bytes"] += getattr(memory, "private", 0)
            except psutil.Error:
                pass
        return result

    def take(self):
        result = dict(self.elapsed)
        self.elapsed.clear()
        return result

    def close(self):
        for owner, name, original, was_local in reversed(self.restore):
            if was_local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.restore.clear()
