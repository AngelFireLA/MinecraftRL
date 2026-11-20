"""Reusable MinecraftRL task definitions."""

from .aim_break_tnt import AimBreakTntTask
from .aim_tool_break import AimToolBreakTask
from .bed_defense_break import BedDefenseBreakTask
from .bed_defense_break_v2 import BedDefenseBreakV2Task
from .bridging import BridgingTask
from .multi_tool_break import MultiToolBreakTask
from .pvp_benchmark import PvpBenchmarkTask

__all__ = [
    "AimBreakTntTask",
    "AimToolBreakTask",
    "BedDefenseBreakTask",
    "BedDefenseBreakV2Task",
    "BridgingTask",
    "MultiToolBreakTask",
    "PvpBenchmarkTask",
]
