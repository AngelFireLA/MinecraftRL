# HeadlessMC launcher reference

HeadlessMC 2.10.0 runs Forge 1.8.9 without an OpenGL window. Each client
has its own game directory, save, port, and transformed launcher JARs.

| Directory | Contents |
| --- | --- |
| `shared/.minecraft` | Shared Minecraft versions, libraries, and assets |
| `instances/rl-N` | One client's game directory with its world, mods, options, config, and logs |
| `instances/rl-N/.headlessmc-runtime` | Temporary transformed JARs for that client |

Give each running client a separate game directory and save. Sharing the
Minecraft cache reduces downloads and disk usage. The [Windows setup
guide](../docs/setup.md) covers prerequisites and the first launch.

## Initialize the shared cache

Run from `HeadlessMC`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Initialize-Cache.ps1
```

This fetches the pinned HeadlessMC and HMC-Specifics releases, then installs
Minecraft 1.8.9 and Forge `11.15.1.2318-1.8.9` in `shared/.minecraft`. The
first launch fetches additional libraries and dummy assets. Keeping this
cache under `HeadlessMC` isolates dummy assets from regular Minecraft or
MultiMC installations.

## Create instances

`New-Instance.ps1` creates a fresh `void` world using
`template/create_void_world.py`, then installs the two mods and minimal
rendering settings for the new instance. To start from a local save, pass
its directory with `-WorldTemplate`. Choose an unused instance name.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\New-Instance.ps1 -Name rl-1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\New-Instance.ps1 -Name rl-2
```

By default, `rl-0` uses port 8765, `rl-1` uses port 8766, and `rl-N` uses
port `8765 + N`. Each `instance.json` records that mapping and a unique
offline username and UUID. On launch, `Start-Instance.ps1` migrates older
instances with a shared UUID to the UUID for that instance's username.

## Launch one client

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Instance.ps1 -Name rl-0
```

The client stays attached to its terminal. After Forge starts,
HMC-Specifics accepts these commands.

```text
gui       show the current screen and its buttons
render    show the currently rendered text
click 1   click the element with ID 1
memory    show JVM heap statistics
quit      shut down Minecraft cleanly
```

Each game JVM starts with a 256 MiB heap and can grow to 1,024 MiB
(`-Xms256m -Xmx1024m`). `-XX:ActiveProcessorCount=4` guides internal thread
pool sizing. The launcher uses dummy assets by default. To change the heap
and reported processor count for an experiment, use launcher parameters.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Instance.ps1 -Name rl-0 -HeapMb 1536 -ActiveProcessorCount 2
```

Use `-GarbageCollector default` or `-GarbageCollector g1` to compare
collectors. The script passes game JVM flags through `hmc.jvmargs`. In the
tested 1.8.9 launch, `launch ... --jvm` forwarded only the first flag when
given several.

### Startup messages

Forge 1.8.9 can print an ASM 5 error about
`META-INF/versions/9/module-info.class` during mod scanning and skip
`headlessmc-lwjgl.jar` as a mod candidate. The HMC logger may also report
that its legacy Log4j pattern lacks `{nolookups}` after the launcher reports
patching `Interpolator` and `JndiLookup`. Both messages appeared in a
successful smoke test. Look for `HMC-Specifics initialized` and
`successfully loaded 4 mods`. Use `gui` to see `GuiMainMenu` before the
training world opens.

## Launch multiple clients

After creating the matching instance directories, start the clients.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Instances.ps1 -Count 2
```

The script opens one interactive console per client and staggers launches
by two seconds to ease startup CPU and disk load. If a client exits with an
error, its console waits for Enter. Use a matching `train.py --num-envs N`
on the Python side. `--vec-env subproc` provides the standard SB3 subprocess
backend for comparison. A two-client smoke test opened both `void` saves,
reset and stepped each environment three times, and shut down both clients
with exit code 0.

### Identity and world loading

The launcher sets these mod properties for each client.

```text
minecraftrl.instanceId=rl-N
minecraftrl.port=8765+N
minecraftrl.world=void
```

The bridge verifies the protocol version and instance ID. The mod opens the
existing save named by `minecraftrl.world` and emits `client_ready` once the
world, player, and integrated server are available. Python waits for that
message before resetting the environment.

For a two-client smoke test, start Python from `RLCraft`.

```powershell
python .\python\smoke_multi_env.py --num-envs 2 --steps 3
```

Then start the clients from `HeadlessMC`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Instances.ps1 -Count 2
```

## Runtime and benchmark settings

The ten-client tests used 1,024 MiB maximum heap and four logical processors
reported per client. A 1,024 MiB heap matched the throughput of 1,280 MiB
with lower memory use, while 768 MiB ran slower. Process RSS includes
metaspace, thread stacks, native buffers, and JIT code in addition to the
heap. Headless throughput depends mainly on CPU time for client and
integrated-server ticks.

`Start-Instances.ps1` applies `-ActiveProcessorCount` to every client.
With ten instances, the four-processor setting and the default Java 8
collector performed best in the local screen. Reporting two processors
failed to improve collection, bridge latency, and reset latency together.
G1 reduced throughput and increased latency in that screen. Use consistent
heap, processor, and collector settings across clients when comparing them.

HeadlessMC's display loop can yield the client thread on Windows even with
`-DisplaySleepMilliseconds 0`. Multiple passes through that loop per
synchronized step had limited a client to about 15–25 transitions per
second. MinecraftRL skips `Display.update()` during renderless synchronized
training and keeps it in vanilla and visualized modes. Set
`-DisplaySleepMilliseconds 1` to reduce CPU use outside the renderless
training path.

In renderless synchronized mode, clients wait for work between actions.
TCP messages wake the client, with a bounded 10 ms wake-up for maintenance.
Use `-IdleWaitMode off` for the reference configuration. The default CPU
trainer uses the `lean-shared` worker backend independently of the idle
mode.

The [benchmark supervisor](../RLCraft/python/benchmark_matrix.py) starts
clients without extra windows, logs each instance, records process-tree
memory, and sends `quit` to its own clients before cleanup. The
[optimization report](../RLCraft/benchmarks/optimization_summary.md)
documents the paired run, the local ten-client recommendation, and the
limits of the long-run validation.

### Upstream documentation

- [Launching](https://headlesshq.github.io/headlessmc/launch/)
- [Configuration](https://headlesshq.github.io/headlessmc/configuration/)
- [HMC-Specifics](https://headlesshq.github.io/headlessmc/specifics/)
