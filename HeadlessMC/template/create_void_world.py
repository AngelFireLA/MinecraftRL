"""Write the initial level.dat for an empty Minecraft 1.8.9 superflat world."""

import gzip
import struct
import sys
from pathlib import Path


def nbt_string(value: str) -> bytes:
    encoded = value.encode("ascii")
    return struct.pack(">H", len(encoded)) + encoded


def tag(name: str, value: int | str | bytes) -> bytes:
    if isinstance(value, int):
        kind, payload = 3, struct.pack(">i", value)
    elif isinstance(value, str):
        kind, payload = 8, nbt_string(value)
    else:
        kind, payload = 10, value + b"\x00"
    return bytes([kind]) + nbt_string(name) + payload


def create_world(destination: Path) -> None:
    game_rules = b"".join(
        tag(rule, value)
        for rule, value in {
            "doMobSpawning": "false",
            "doDaylightCycle": "false",
            "doFireTick": "false",
            "randomTickSpeed": "0",
        }.items()
    )
    # The player joins in Creative above the void. The mod builds the task's
    # platform and switches the player to Survival on the first reset.
    fields = {
        "LevelName": "void",
        "version": 19133,  # Anvil save format
        "generatorName": "flat",
        "generatorVersion": 0,
        "generatorOptions": "3;minecraft:air;1;",
        "SpawnX": 0,
        "SpawnY": 64,
        "SpawnZ": 0,
        "GameType": 1,
        "GameRules": game_rules,
    }
    # Minecraft reads these world flags as NBT bytes (TAG_Byte).
    flags = {"initialized": 1, "allowCommands": 1, "MapFeatures": 0, "Difficulty": 0}
    data = b"".join(tag(name, value) for name, value in fields.items())
    data += b"".join(b"\x01" + nbt_string(name) + bytes([value]) for name, value in flags.items())
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "level.dat").open("xb") as level:
        level.write(gzip.compress(tag("", tag("Data", data)), mtime=0))


if __name__ == "__main__":
    create_world(Path(sys.argv[1]))
