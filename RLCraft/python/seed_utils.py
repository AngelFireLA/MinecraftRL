"""Deterministic seed mixing shared by curricula and evaluation."""

from __future__ import annotations


UINT64_MASK = (1 << 64) - 1
POSITIVE_INT64_MASK = (1 << 63) - 1


def splitmix64(value: int) -> int:
    """Return a well-dispersed unsigned 64-bit value for an integer input."""

    mixed = (int(value) + 0x9E3779B97F4A7C15) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (mixed ^ (mixed >> 31)) & UINT64_MASK


def evaluation_seed(base_seed: int, episode_index: int) -> int:
    """Create a reproducible, non-consecutive positive Java-compatible seed."""

    return splitmix64(int(base_seed) + int(episode_index)) & POSITIVE_INT64_MASK


__all__ = ["evaluation_seed", "splitmix64"]
