"""Environment-only task construction, independent of training presets."""

from tasks import (
    AimBreakTntTask,
    AimToolBreakTask,
    MultiToolBreakTask,
    BedDefenseBreakTask,
    BedDefenseBreakV2Task,
    BridgingTask,
    PvpBenchmarkTask,
)

TASK_FACTORIES = {
    "aim_break_tnt": AimBreakTntTask,
    "aim_tool_break": AimToolBreakTask,
    "multi_tool_break": MultiToolBreakTask,
    "bed_defense_break": BedDefenseBreakTask,
    "bed_defense_break_v2": BedDefenseBreakV2Task,
    "bridging": BridgingTask,
    "pvp_benchmark": PvpBenchmarkTask,
}


def create_task(task_id, max_episode_steps):
    return TASK_FACTORIES[task_id](max_episode_steps)
