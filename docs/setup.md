# Run MinecraftRL on Windows

## Prerequisites

- Windows with PowerShell. The default multi-environment worker backend uses Windows named pipes.
- Python 3.11 or 3.12 with `python` on PATH.
- A JDK 17 for Gradle and the HeadlessMC launcher, and a 64-bit Java 8 installation for the Minecraft 1.8.9 process. Install Java 8 where the launcher and Gradle can discover it, such as the JDK vendor's default Windows install location.
- Minecraft Java Edition. These scripts run local single-player training worlds.

Run the commands below from the repository root in Windows PowerShell. Replace the example Java path with your installed JDK 17 directory. Set `JAVA_HOME` in the client terminal as well if you open a new terminal.

## Build and create one environment

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-17'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\RLCraft\python\requirements.txt

Push-Location .\RLCraft
.\gradlew.bat build
Pop-Location

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\HeadlessMC\Initialize-Cache.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\HeadlessMC\New-Instance.ps1 -Name rl-0
```

Cache initialization fetches the pinned HeadlessMC and HMC-Specifics releases, checks the newly downloaded JARs against their upstream SHA-256 digests, and installs Minecraft and Forge into a shared cache. The first client launch fetches any remaining libraries. Each instance gets a fresh void world, options, mods, and its own game directory. To reuse a local world as a template, pass its directory to `-WorldTemplate`. Run `New-Instance.ps1` once per instance name.

## Check the Python–Minecraft connection

In terminal 1, from the repository root:

```powershell
.\.venv\Scripts\python.exe .\RLCraft\python\smoke_multi_env.py --task aim_break_tnt --num-envs 1 --steps 3
```

In terminal 2, from the repository root after setting `JAVA_HOME`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\HeadlessMC\Start-Instance.ps1 -Name rl-0
```

The smoke test prints `reset ok` and reports three transitions. The client reconnects automatically to the next Python command, so you can leave it running.

## Train and evaluate

In terminal 1:

```powershell
Set-Location .\RLCraft
..\.venv\Scripts\python.exe .\python\train.py --task aim_break_tnt --num-envs 1
..\.venv\Scripts\python.exe .\python\evaluate.py --task aim_break_tnt --episodes 5 --fast
```

The trainer uses the task's default step budget and writes `models/aim_break_tnt/final_model.zip` on completion. With a different output path, pass the checkpoint explicitly with `evaluate.py --model`. Ctrl+C in the trainer saves `latest_model.zip`, which you can load with `train.py --resume`. Allow the trainer to exit before starting evaluation on the same port. Enter `quit` in the client console to shut it down cleanly.

## Use multiple clients

From the repository root, create any missing instances. Each `rl-N` uses port `8765 + N`.

```powershell
0..3 | ForEach-Object {
    $name = "rl-$_"
    if (-not (Test-Path ".\HeadlessMC\instances\$name")) {
        powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\HeadlessMC\New-Instance.ps1 -Name $name
    }
}
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\HeadlessMC\Start-Instances.ps1 -Count 4
```

In the Python terminal, from `RLCraft`:

```powershell
..\.venv\Scripts\python.exe .\python\train.py --task bed_defense_break_v2 --num-envs 4
```

The launcher starts one console per client. The trainer uses CPU and the lightweight shared-memory backend by default. Set `--vec-env shared` and launch clients with `-IdleWaitMode off` to select the benchmark reference configuration. The [benchmark report](../RLCraft/benchmarks/optimization_summary.md#repeat-the-paired-comparison) includes a supervised command for running both configurations across the same three seeds.

For rendered evaluation, use a graphical Forge client with the mod loaded and a local world open, then run `evaluate.py` without `--fast`. The [mod reference](../RLCraft/README.md) documents the development client, TCP handshake, task contracts, and in-game controls.
