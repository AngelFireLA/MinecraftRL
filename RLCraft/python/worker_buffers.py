"""Pickle-safe buffer descriptors without trainer dependencies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _ArrayDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str
