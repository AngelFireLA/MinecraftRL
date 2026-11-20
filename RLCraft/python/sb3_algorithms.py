"""Lazy Stable-Baselines3 algorithm lookup used by the generic runners."""

from __future__ import annotations

from typing import Any


def checkpoint_custom_objects(
    observation_space: Any, action_space: Any
) -> dict[str, Any]:
    """Replace environment metadata that is fragile across NumPy versions."""
    return {
        "observation_space": observation_space,
        "action_space": action_space,
        "_last_obs": None,
        "_last_episode_starts": None,
    }


def get_algorithm_class(name: str) -> type[Any]:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing RL dependency {error.name!r}. Run: "
            "python -m pip install -r python/requirements.txt"
        ) from error

    algorithms: dict[str, type[Any]] = {
        "ppo": PPO,
    }
    try:
        return algorithms[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported SB3 algorithm: {name!r}") from error


__all__ = ["checkpoint_custom_objects", "get_algorithm_class"]
