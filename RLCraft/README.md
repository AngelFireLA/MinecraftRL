# MinecraftRL Forge mod and Python API

The Forge 1.8.9 mod controls client and integrated-server ticks and exchanges
actions and observations with Python over TCP. The bridge keeps blocking network
I/O off the Minecraft threads. For headless training, follow the [Windows setup
guide](../docs/setup.md).

## Graphical development client

Run the following commands from `RLCraft`. Gradle uses JDK 17 and launches
Minecraft with Java 8. Start the interactive bridge in one terminal.

```powershell
python .\python\minecraft_bridge.py
```

In a second terminal, launch the Forge client and open a disposable single-player
world.

```powershell
.\gradlew.bat runClient
```

By default, the mod connects to `127.0.0.1:8765` as `rl-0` and retries once a
second. HeadlessMC passes a separate identity, port, and world to each process.
When `minecraftrl.world` is set, the mod opens the existing save from the main menu
and reports an error if the save is missing.

## TCP protocol

Use the chat console to check the connection.

```text
/rl status
/rl send hello from Minecraft
```

Messages appear in the Python terminal. Text entered at the `python>` prompt
appears in Minecraft chat. The prompt blocks only the Python console thread.
The bridge uses UTF-8 NDJSON, with one JSON object per line.

```json
{"type":"message","text":"hello"}
{"type":"control","action":"mode","mode":"step"}
{"type":"control","action":"step","ticks":1}
```

A connection starts with an identity handshake. The client then reports readiness
once the world, player, and integrated server are available.

```json
{"type":"hello","protocol_version":1,"instance_id":"rl-0","world":"void"}
{"type":"client_ready","instance_id":"rl-0","world":"void"}
```

`MinecraftEnv.reset()` waits for both messages and rejects clients with a
different protocol version or instance ID. TCP preserves byte order. Newlines
delimit messages regardless of how `recv()` splits the stream.

## Tick controls

```text
/rl mode vanilla   # normal operation at 20 ticks/s
/rl mode fast      # virtual clock with no 50 ms sleep
/rl mode step      # integrated server waits for explicit permits
/rl step [n]       # allow n integrated-server ticks
/rl benchmark [s]  # s seconds with rendering, s without, then a TCP benchmark
/rl syncbench [n]  # n synchronized Python/client/server transitions
/rl stats          # measured client and server tick rates
/rl resetstats     # reset the measurement window
```

The interactive Python bridge exposes the corresponding controls.

```text
/mode vanilla
/mode fast
/mode step
/step 1
```

### Render and TCP benchmark

Run this command in single-player with the bridge connected.

```text
/rl benchmark 5
```

The benchmark runs three phases.

1. Run FAST mode with rendering for the requested duration.
2. Run FAST mode for the same duration with `EntityRenderer.updateCameraAndRender` skipped.
3. Exchange 100 sequential 64 KiB TCP round trips between Java and Python.

Disable VSync to measure the first phase without a display cap. During the
renderless phase, the picture freezes while client ticks and networking continue.
The benchmark restores rendering and vanilla mode when it finishes. It reports
client/server ticks per second, mean and p95 TCP latency, and payload throughput.
Restart the Python bridge after updating it so it handles `benchmark_echo`.

### Synchronized transition benchmark

Run the benchmark from a single-player world with the interactive bridge connected.

```text
/rl syncbench 1000
```

Each transition follows this order.

1. Java waits for the integrated server to park at the start-of-tick barrier.
2. Python sends mouse deltas and button states.
3. The client applies camera inputs via `Entity.setAngles`, updates the vanilla
   raycast, and completes its tick to send packets. Movement controls are
   released afterward. Held attack and use persist until the next action.
4. Java permits one integrated-server tick and waits for it to finish.
5. A client tick processes the server update and recomputes the raycast.
6. Java returns the observation to Python, which can then send the next action.

The benchmark records position, rotation, health, ground contact, and details of
the block or entity under the crosshair, including its type, ID or state,
position, distance, and health where applicable. It skips OpenGL rendering and
reports transition rates on both sides plus mean observation size.

### Gymnasium environment

`python/minecraft_env.py` exposes this loop through `reset()`, `step()`, and
`close()`. Each environment connects to one identified Minecraft client over its
own port. Run one Python listener per port, whether it is the interactive bridge,
a Gymnasium task script, or a training worker.

Each task declares its Gymnasium spaces, action encoder, and required raw
observation fields. During `reset()`, Java validates that schema, accepts the
declared action primitives, and serializes the requested observations.

| Input | Primitives |
| --- | --- |
| Camera | `mouse_dx`, `mouse_dy` |
| Movement | `forward`, `back`, `left`, `right`, `jump`, `sneak`, `sprint` |
| Held buttons | `attack`, `use` |
| Single clicks | `attack_once`, `use_once` |
| Inventory | `hotbar_slot` (0–8) |

Consecutive actions with `attack=True` keep the button down across ticks, so block
breaking progresses continuously. `attack=False` releases it and resets vanilla
break progress. Use `attack_once` to strike an entity once per decision, `use` for
actions such as drawing a bow, and `use_once` to place a block or interact once.
Client ticks pause while waiting for Python, so held inputs advance only when
the environment steps.

The scripted task controllers accept `--visualized` to retain rendering, pace
actions at 20 server ticks per second, and pause for one second around resets.
Adjust the pace and reset pause with `--tps` and `--episode-pause`. Renderless
training runs without this pacing. The same mode can be enabled from Python.

```python
env = MinecraftEnv(task, visualized=True, target_tps=20.0)
```

Pacing limits when steps can start. Each action still advances exactly one
integrated-server tick before its observation.

| Observation | Available fields |
| --- | --- |
| Player (`player.*`) | Position, rotation, health, ground contact, velocity, sneak/sprint, collisions, falling, selected slot, item use and charge, and block-breaking progress |
| Held item and inventory (`held.*`, `inventory.*`) | Item ID, count, damage, hotbar, and arrow count |
| Crosshair raycast (`raycast.*`) | Hit type, target ID and position, face, distance, health, hardness, and whether the held tool can harvest it |
| Nearest living entity (`nearest_living.*`) | World and relative position, distance, relative yaw/pitch, velocity, health, and alive status |
| Task entity (`task_entity.*`) | Same fields as `nearest_living.*`, with the entity identity locked at reset |
| Task goal (`task_goal.*`) | World position, relative vector, distances, and angles derived from `goal_x/y/z` in `reset_payload()` |
| Nearby blocks (`local.blocks_3x3x3`) | 27 block-state IDs around the player's feet, with X varying fastest, then Z, then Y |

`raycast` reflects `Minecraft.objectMouseOver` from the player's eyes and
orientation. It returns `MISS` or the visible block or entity directly under the
crosshair within vanilla reach. For an aiming task, use
`task_entity.yaw_delta` and `task_entity.pitch_delta` as the aiming signal and
`raycast.id == task_entity.id` to check whether the crosshair reaches the target.
`nearest_living` tracks whichever living entity is currently closest.

Java computes the raycast when a requested field or attack/use action needs it.
Entity, hotbar, and local block queries also run on demand. To add a field,
implement its Java extractor and register the field ID.

This example uses one yaw bin as its action and the target's yaw error as its
policy observation. The reward and termination methods also read the raw
target and raycast IDs.

```python
import numpy as np
from gymnasium import spaces
from minecraft_env import MinecraftTask

class AimYawTask(MinecraftTask):
    def __init__(self):
        self.yaw_bins = (-4.0, -1.0, 0.0, 1.0, 4.0)
        super().__init__(
            task_id="aim_yaw",
            action_space=spaces.Discrete(len(self.yaw_bins)),
            observation_space=spaces.Box(
                low=np.array([-180], dtype=np.float32),
                high=np.array([180], dtype=np.float32),
            ),
            action_fields=("mouse_dx",),
            observation_fields=("task_entity.yaw_delta", "task_entity.id", "raycast.id"),
            max_episode_steps=100,
        )

    def encode_action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(action)
        return {"mouse_dx": self.yaw_bins[action]}

    def observation(self, raw):
        return np.array([raw["task_entity.yaw_delta"]], dtype=np.float32)

    def reward(self, previous, current):
        return 1.0 if current["raycast.id"] == current["task_entity.id"] else -0.01

    def terminated(self, previous, current):
        return current["raycast.id"] == current["task_entity.id"]
```

Other tasks can use `MultiDiscrete`, a continuous `Box`, or another space the
chosen algorithm supports. `reward`, `terminated`, `truncated`, `on_reset`, and
`reset_payload` are task-specific. Python dependencies are listed in
`python/requirements.txt`.

#### Physical resets

`reset()` negotiates the schema and waits for Java to finish resetting. The
`hold_attack_block` fixture anchors a small arena at the player's initial
position. Between episodes, it clears the arena, restores a bedrock support
and stone target, removes leftover entities, and resets player position,
rotation, velocity, health, hunger, effects, inventory, and survival mode.

The reset runs on the integrated-server thread at the synchronized tick barrier.
The server applies the changes and sends packets. A client tick processes them
before Python receives the first observation. `info["world_reset_applied"]`
reports whether Java applied a task-specific physical reset.

Use a disposable single-player world for physical resets. They alter the arena
around the player and clear the inventory. Reuse each HeadlessMC world across
episodes so the client can reset the bounded arena in place.

## Tasks and training

Commands in this section run from `RLCraft`, except for the HeadlessMC
launcher commands, which run from `HeadlessMC`.

### Aim and break TNT

With `aim_break_tnt`, the player stays in place and a TNT block spawns at one
of 30 reachable positions in front of them. The Gymnasium seed determines the
spawn position. The policy observes yaw and pitch error, distance, whether
the crosshair is on the target, and whether the target is still present. A
`MultiDiscrete` action selects yaw and pitch bins and a single-click attack.
Success is determined from the block state at the target position.

Install the Python dependencies.

```powershell
python -m pip install -r .\python\requirements.txt
```

Check the environment with the deterministic controller.

```powershell
python .\python\examples\example_aim_break_tnt_loop.py --episodes 10 --visualized
```

Train PPO with rendering disabled.

```powershell
python .\python\train.py --task aim_break_tnt --total-timesteps 100000
```

### Parallel HeadlessMC training

The training defaults are `lean-shared` and CPU. HeadlessMC uses event-driven
waiting between synchronized actions. The `lean-shared` backend keeps one worker
per instance with shared observation buffers. The workers avoid loading
PyTorch and Stable-Baselines3. Other backends are `shared`, `batched`, and
`subproc`. Pass `--device cuda` to select a GPU.

To check two instances, start the Python listener in the first terminal.

```powershell
python .\python\smoke_multi_env.py --num-envs 2 --steps 3
```

Then launch the clients from `HeadlessMC` in another terminal.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Instances.ps1 -Count 2
```

Once the smoke test finishes, use the trainer to start a run with two environments.

```powershell
python .\python\train.py --task aim_break_tnt --num-envs 2 --total-timesteps 100000
```

PyTorch and BLAS default to one CPU thread per process. This limits competition
between the small MLP optimizer and the Minecraft clients. Override with
`--torch-threads` when benchmarking other configurations.

The port convention is `rl-0` at 8765, `rl-1` at 8766, and so on. To run
`rl-2` and `rl-3` on ports 8767 and 8768, use this command.

```powershell
python .\python\train.py --task aim_break_tnt --num-envs 2 --first-instance-index 2
```

`--base-port` sets the port corresponding to index zero. `--vec-env shared`
selects the reference backend and `--vec-env subproc` selects the standard SB3
subprocess backend. Checkpoint intervals count total SB3 timesteps across
all workers. Use `--visualized` with one environment.

Close `minecraft_bridge.py` before starting the trainer on the same port.
SB3 prints its rollout metrics and summarizes episodes every 20 episodes by
default. The default `aim_break_tnt` output directory, `models/aim_break_tnt/`,
contains `monitor.csv`, checkpoints every 10,000 steps, and `final_model.zip`.
Ctrl+C saves `latest_model.zip`. For the structured-state MLP policies,
`--device auto` resolves to CPU. The selected device is recorded in
`run_metadata.json`.

Evaluate the final policy or pass a checkpoint explicitly.

```powershell
python .\python\evaluate.py --task aim_break_tnt --episodes 10 --verbose-steps
```

```powershell
python .\python\evaluate.py --task aim_break_tnt --model .\models\aim_break_tnt\checkpoints\ppo_aim_break_tnt_10000_steps.zip --episodes 5
```

Use `--fast` for evaluation without rendering or world sounds. Rendered
evaluation keeps both and runs at 20 TPS. Task classes define the Gymnasium
contract, reward, and episode completion. `python/task_registry.py` sets the
algorithm and default hyperparameters for the shared runners.

### Aim, select a tool, and break a block

`aim_tool_break` places stone, oak planks, or sand at the same reachable
positions as the TNT task. Each reset equips a diamond pickaxe in slot 0,
a diamond axe in slot 1, and an iron shovel in slot 2. A support block keeps
sand in place before the first observation.

The `MultiDiscrete` action selects yaw and pitch bins, an absolute hotbar
slot, and the held attack state. The policy observes the selected slot,
break progress, and the per-tick breaking speed of each tool. Those speeds
come from Minecraft's tool efficiency, harvestability, and block hardness.
They account for different tool tiers and work for a new block such as
cobblestone without a hand-written tool mapping.

Initial yaw spans 360°, pitch spans ±45°, and the starting slot is sampled
from the three tools. Camera bins combine movements up to ±90° with fine
adjustments. A per-step cost and a progress reward favor efficient breaking.

Use the scripted controller to check aiming, tool selection, held attack,
and rewards.

```powershell
python .\python\examples\example_aim_tool_break_loop.py --episodes 12 --visualized
```

Look for `aim_tool_break validation passed` and all three materials in
`blocks={...}`. The visualized test also shows dropped items disappearing at
reset. The server removes and untracks them immediately, which sends the
client the entity-destroy packet.

The PPO preset trains for 300,000 steps by default.

```powershell
python .\python\train.py --task aim_tool_break
```

Evaluate with rendering or collect success statistics without rendering.

```powershell
python .\python\evaluate.py --task aim_tool_break --episodes 12 --verbose-steps
python .\python\evaluate.py --task aim_tool_break --episodes 100 --fast
```

### Multi-block tool selection

`multi_tool_break` places three to five reachable targets at 45° intervals
around the player, including behind them. The fixed observation has room
for five targets and includes each target's angles, distance, presence, and
three tool speeds. The agent can choose the order of destruction.

Before returning the first observation, Java checks that each client's block
ID matches the server's reset. Up to eight extra client ticks can process
pending packets without advancing the world. If the match still fails, Java
checks server state, resends the block packets, and gives the network thread
up to 250 ms. Python retries the same seeded reset once before failing a worker.

The total block-break reward is 10 regardless of the initial number of
targets. Destroying all targets earns another 5. A per-step cost and reversible
changes in aiming, tool choice, and breaking progress give dense feedback.
Held attacks earn a small reward only when aimed at a target with the
correct tool. Leaving a partially broken block gives back its progress reward,
which prevents repeated interruption from earning that reward again.

Check the task with the scripted controller, then train and evaluate with the
shared runners.

```powershell
python .\python\examples\example_multi_tool_break_loop.py --episodes 6 --visualized
```

```powershell
python .\python\train.py --task multi_tool_break --output-dir .\models\multi_tool_break_v2
python .\python\evaluate.py --task multi_tool_break --model .\models\multi_tool_break_v2\final_model.zip --episodes 10 --verbose-steps
```

The curriculum progresses through one target, two targets, three targets,
three to five guided targets, and finally three to five targets in free order.
The four stage transitions require 100 recent episodes and success rates of
at least 70%, 65%, 60%, and 55%, respectively. The Gymnasium spaces stay
fixed throughout. During guided stages, Python puts the nine values for the
`focus` target in the first observation slot while retaining the target's
Java index. This lets the policy reuse the same weights on successive targets.
Evaluation uses the full free-order reset.

The trainer scales `n_steps` to keep vectorized rollouts close to the preset
single-environment size. With ten clients it uses 256 steps per client, or
2,560 transitions per PPO update. `--n-steps` overrides that calculation.
Episode summaries report 100-episode rolling success, blocks broken,
completion, reward, length, and the curriculum stage. Use checkpoints with
the current `focus` observation field when resuming.

### Bed defense V1

`bed_defense_break` trains a single PPO policy to aim, choose between shears,
axe, and pickaxe, break a route, and destroy either half of the bed. Tool
choice is rewarded for wool, wood, end stone, and obsidian. Minecraft 1.8.9
gives the axe no advantage against the bed itself. Gravity remains active,
so players starting in a supported top position can fall when the support breaks.

The observation has two keys with a fixed shape for the entire curriculum.

- `state` contains 50 values describing the two bed halves, player orientation
  and state, the block under the crosshair, best progress on that block, and
  the optional hint.
- `grid` contains eight channels over a 7 × 5 × 7 volume. A linear layer encodes
  the flattened volume. Existing Conv3D checkpoints can still be loaded for
  evaluation with their original extractor. Workers transfer the binary grid
  as `uint8` and the extractor converts it to floats, cutting grid IPC traffic
  to roughly one quarter of the float representation.

The bed's position is given to the policy as map knowledge. Materials behind
opaque blocks stay unknown to the policy. The channels distinguish known
space, visible solids, wool, wood, end stone, obsidian, bed position, and the
guided target. The complete grid is available on the Python side for reward
calculation, temporary hints, and metrics.

Yaw, pitch, tool, and held attack are available at every stage. Early stages
use shorter horizons. The curriculum has 13 stages, indexed from 0 to 12.

| Index | Stage | Reset |
| ---: | --- | --- |
| 0 | `exposed_near` | Exposed bed with the crosshair on target |
| 1 | `exposed_random` | Exposed bed with randomized camera |
| 2 | `single_mixed_aimed` | One wool, wood, or end-stone block, balanced by seed, with the crosshair on target |
| 3 | `single_mixed` | Same geometry and materials with randomized camera |
| 4 | `tunnel_guided` | One- to three-block tunnel |
| 5 | `layer_one` | One defensive layer |
| 6 | `layer_two` | Two defensive layers |
| 7 | `varied_pose` | Start from the side, corner, or above |
| 8 | `obsidian_guided` | Obsidian inner layer |
| 9 | `layered_hint_75` | 75% hint probability |
| 10 | `layered_hint_50` | 50% hint probability |
| 11 | `layered_hint_25` | 25% hint probability |
| 12 | `layered_free` | Full defense without a supplied target block |

Advancement waits for 100 eligible episodes at the current stage and its
success threshold. The window resets at each stage change. For stages 9–11,
only episodes without hints count. The `[mastery]` log shows window size,
measured success, and the required threshold.

The reward charges `-0.01` per tick plus an aim error cost. Switching from
the wrong tool to the right tool earns `0.5`. Every switch costs `0.03`.
An attack without break progress incurs only the time cost so the agent can
learn camera and attack heads together early in training. Signed break
progress contributes up to 4 points per block. Abandoning progress reverses
the reward. Irreversible improvement to the best route can earn up to 3 points
in total. Breaking either bed half earns 20 and terminates the episode.
The PPO entropy bonus is disabled for this task. Sampling actions still
provides exploration. The `[episodes]` log reports `material_success_100`
by material so the overall success rate cannot obscure single-tool behavior.

Java caches the visibility mask while the world, grid origin, eye position,
and all 245 block states match. Rotating in place can reuse it, while falling
or breaking a block causes a recalculation. Each immutable Python observation
caches the NumPy grid and visibility arrays for reward, observation, and
metrics. Training workers skip returning the full raw Java observation in
`info`. Evaluation and scripted controllers retain it for diagnostics.

The 49-value state contract and the current 50-value contract require
different checkpoints. Recent 50-value checkpoints, such as
`ppo_bed_defense_break_100000_steps.zip`, remain compatible across curriculum
and reward changes. The loader supplies the current Gymnasium metadata, so
it can also reload a checkpoint created with NumPy 2 from NumPy 1.26.

To validate a stage on `rl-0`, start a client and run the scripted controller.

```powershell
python .\python\examples\example_bed_defense_break_loop.py --episodes 9 --curriculum-stage 0
```

Use a stage index from 0 to 12. Omitting it tests the final free-order reset.
Add `--visualized` to render and pace the client at 20 TPS.

```powershell
python .\python\examples\example_bed_defense_break_loop.py --episodes 3 --curriculum-stage 12 --visualized
```

Start ten existing clients from `HeadlessMC`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Instances.ps1 -Count 10
```

Run PPO training from `RLCraft`.

```powershell
python .\python\train.py --task bed_defense_break --num-envs 10 --total-timesteps 2000000 --output-dir .\models\bed_defense_break_v3
```

The preset uses the flat extractor, `learning_rate=1e-4`, `n_epochs=5`,
`clip_range=0.15`, and `target_kl=0.02`. Override the rate with
`--learning-rate`. Train a fresh policy for the flat extractor when migrating
from a Conv3D checkpoint.

To profile ten SB3 rollouts (25,600 transitions), send the model and phase
measurements to a separate directory.

```powershell
python .\python\train.py --task bed_defense_break --num-envs 10 --total-timesteps 25600 --output-dir .\models\bed_defense_break_profile --checkpoint-freq 100000000 --profile --profile-output .\models\bed_defense_break_profile\profile.jsonl
```

Skip the first rollout when comparing phase timings because Torch can
initialize lazily. Each `[profile]` group reports collection and
`optimize+between` separately. During collection, `worker_critical` is the
slowest worker's time per vector step, `worker_spread` measures worker
imbalance, `bridge` covers Forge/tick/TCP waits, `task` covers Python reward
and observation computation, and `parent_policy_ipc` estimates inference
and remaining IPC. The critical path includes automatic resets. Reset metrics
include `mean_reset`, `p95_reset`, and `reset_bridge`.

The default `lean-shared` backend uses lightweight workers and shared buffers.
Select `--vec-env shared` for the reference or choose `batched` or `subproc`.
After schema negotiation, training uses the faster validation path.
`--strict-validation` checks the full contract on each transition.

The task saves a periodic checkpoint every 50,000 total transitions.
It also saves a mastery checkpoint before each stage change, with the
mastered stage index in the metadata.

```text
checkpoints\ppo_bed_defense_break_stage_02_single_mixed_aimed_mastered_345600_steps.zip
```

`--checkpoint-freq` overrides the periodic cadence. Mastery checkpoints have
their own trigger.

Use a graphical Forge client to watch the final policy.

```powershell
python .\python\evaluate.py --task bed_defense_break --model .\models\bed_defense_break_v3\final_model.zip --episodes 10 --verbose-steps
```

Use `--curriculum-stage` to replay a specific stage. These examples use a
checkpoint from one million transitions.

```powershell
python .\python\evaluate.py --task bed_defense_break --model .\models\bed_defense_break\checkpoints\ppo_bed_defense_break_1000000_steps.zip --curriculum-stage 0 --episodes 5 --verbose-steps
python .\python\evaluate.py --task bed_defense_break --model .\models\bed_defense_break\checkpoints\ppo_bed_defense_break_1000000_steps.zip --curriculum-stage 3 --episodes 5 --verbose-steps
python .\python\evaluate.py --task bed_defense_break --model .\models\bed_defense_break\checkpoints\ppo_bed_defense_break_1000000_steps.zip --curriculum-stage 12 --episodes 5 --verbose-steps
```

When the stage is omitted, evaluation uses the final free-order reset.
Rendering and deterministic prediction are enabled by default.
`--fast` disables rendering. `--stochastic` samples actions from the policy,
and `--seed` makes the reset sequence reproducible. `--guidance off` disables
hints for the stage, `--guidance on` forces them, and `--guidance stage`
preserves the curriculum probability. To compare deterministic and stochastic
behavior on the same reset sequence, run both with the same seed and stage.

```powershell
python .\python\evaluate.py --task bed_defense_break --model .\models\bed_defense_break_v3\checkpoints\ppo_bed_defense_break_1000000_steps.zip --curriculum-stage 6 --guidance off --seed 42 --episodes 20
python .\python\evaluate.py --task bed_defense_break --model .\models\bed_defense_break_v3\checkpoints\ppo_bed_defense_break_1000000_steps.zip --curriculum-stage 6 --guidance off --seed 42 --episodes 20 --stochastic
```

### BedWars bed-breaking V2

`bed_defense_break_v2` is a separate task and does not change the V1 contract or
its checkpoints. It removes route-oracle guidance and trains the policy only from
the requested player state, current raycast, and visibility-limited block volume.

The bed-relative coordinate frame is stable across world rotations:

- longitudinal points from the foot half to the head half; the two bed centers
  are `-0.5` and `+0.5`;
- lateral is the clockwise horizontal perpendicular `(-longitudinal_z,
  longitudinal_x)`;
- vertical zero is the bed block level. Ground is outside the grid.

The action is `MultiDiscrete([11, 11, 3, 2])`: relative yaw, relative pitch,
tool, and held attack. Camera increments retain the established values in
`AimToolBreakTask`. V2 tool indices are `0=diamond_pickaxe`, `1=diamond_axe`,
`2=shears`; they are translated to the existing Java hotbar layout.

The observation is a stable `Dict`:

- `state` has 24 values: selected-tool one-hot (3), player longitudinal/lateral/
  vertical coordinates (3), vertical motion, on-ground, pitch delta to the bed
  center, sine/cosine of yaw delta to the bed center, looked-at category one-hot
  (`nothing`, `bed`, `wool`, `wood`, `end_stone`, `obsidian`, `other_solid`),
  break progress, looked-at longitudinal/lateral/vertical coordinates, and yaw/
  pitch correction to the safe face region;
- `grid` is `uint8` with shape `(8, 4, 7, 8)` ordered as category, vertical,
  lateral, longitudinal. Categories are `air`, `bed`, `wool`, `wood`,
  `end_stone`, `obsidian`, `other_solid`, `unknown`. Every cell is exactly
  one-hot. Bed cells are always known; every other material is exposed only if
  Forge can raycast a visible point on its actual selection hitbox.

The longitudinal grid centers are `-3.5` through `+3.5`, the lateral centers are
`-3` through `+3`, and vertical levels are `0` through `3`. The safe face region
is the centered 80% of the face of the exact selected bounding box, including
the shorter bed hitbox. Corrections are zero while the cursor is inside it and
otherwise give degree deltas to the closest safe point.

V2 reward constants are deliberately centralized in
`python/tasks/bed_defense_break_v2.py`:

- `-0.01` each step;
- `+20` when either bed half is destroyed, followed by termination;
- `-0.04` for each held-attack tick aimed at an existing block with the wrong
  tool (`bed` accepts every task tool; non-task solids never do);
- `5 * (current_progress - previous_progress)` only under the correct-tool
  condition. A completed break is treated as progress `1.0`, avoiding the
  vanilla reset-to-zero penalty. Per-block accounting prevents tool or target
  oscillation from claiming the same progress twice, while wrong-tool progress
  is remembered without being rewarded or negatively charged when corrected;
- `-0.5` for every non-bed block broken beyond the initial layer count. The
  per-shell progress high-water mark already prevents extra blocks from
  earning more progress reward; this smaller cost discourages inefficient
  routes without making recovery from a mistaken first target irrational.

The seven mastery stages are:

| Index | Reset | Step limit |
|---:|---|---:|
| 0 | exposed, fixed relative position, varied aim point still on the bed | 30 |
| 1 | exposed, continuous random reachable position and random yaw/pitch | 60 |
| 2 | one balanced wool/wood/end-stone layer, fixed position, aimed through to the bed | 60 |
| 3 | same one-layer material balance, random pose | 100 |
| 4 | two independent wool/wood/end-stone layers, random pose | 140 |
| 5 | one or two layers, normally obsidian inner and never doubled, random pose; 20% no-obsidian rehearsal excluded from mastery | 320 |
| 6 | one or two independently selected layers including obsidian, never two obsidian, random pose | 320 |

A layer is a complete Manhattan shell around either bed half (8 blocks for the
first shell) and has one material. Seed schedules balance layer depths and valid
materials/configurations rather than relying on accidental random frequency.
Random ground poses are continuous (not half-block aligned), collision-free, and
keep an exposed bed within 4.5-block survival reach. Layered stages also include
supported, slightly jittered top poses that may fall after their support is
broken.

Stages 3 through 6 use a 10% partial-defense rehearsal. Forge first places the
player, then chooses a random reachable point on either bed half and removes
only defense blocks intersecting that exact eye-to-bed ray. This produces
diagonal and off-axis openings rather than a fixed tunnel pattern. The player
still starts at random yaw/pitch, so the policy must discover the already
reachable bed from its observations. These 80-step episodes disable remaining
defense-shell progress shaping, give no free non-bed block allowance, and are
excluded from mastery; breaking the exposed bed is therefore the only useful
route.

The final two limits retain enough time for the roughly 188 correct-tool attack
ticks required by obsidian; earlier stages no longer leave hundreds of idle ticks.
Stage transitions 0–4 require at least 90 successes in the latest 100 eligible
episodes from that same stage. The obsidian stage requires 80 successes before
advancing to the final distribution. Stage-five no-obsidian and partial-defense
rehearsal episodes train the policy but are excluded entirely from this mastery
window. Episode logs include a rehearsal fraction, per-stage success windows,
plus layer
materials, initial tool, excess blocks, correct/wrong-tool attack fractions, and
maximum progress. They also report progress reward, wrong-tool cost, and excess-
block cost separately so reward inflation is immediately visible. Mastery
checkpoints are saved before each stage switch.

Train ten existing HeadlessMC clients from `RLCraft`:

```powershell
python .\python\train.py --task bed_defense_break_v2 --num-envs 10 --total-timesteps 2000000 --output-dir .\models\bed_defense_break_v2
```

Resume an interrupted model for 500,000 additional timesteps while preserving
its curriculum stage:

```powershell
python .\python\train.py --task bed_defense_break_v2 --num-envs 10 --total-timesteps 500000 --output-dir .\models\bed_defense_break_v2 --resume .\models\bed_defense_break_v2\latest_model.zip
```

Evaluate the final reset distribution or one curriculum stage:

```powershell
python .\python\evaluate.py --task bed_defense_break_v2 --model .\models\bed_defense_break_v2\final_model.zip --episodes 20 --verbose-steps
python .\python\evaluate.py --task bed_defense_break_v2 --model .\models\bed_defense_break_v2\final_model.zip --curriculum-stage 2 --episodes 20 --verbose-steps
```

### Void bridging

`bridging` is a ten-stage mastery curriculum. It starts overhanging the
target-side edge at local `z=1.29`, with absolute yaw `180` and absolute pitch
`82`, aimed near the center of the south/target-facing side of the bedrock
start block. It then randomizes pose and increases
the straight void gap through 1, 5, 8, 11, 15, 22, and 30 blocks; the last
stage samples a gap uniformly from 16 through 64. The intermediate gaps avoid
forcing PPO to jump directly from a memorized five-block behavior to a
fifteen-block bridge. The target is another bedrock block at the
same height and every reset gives hotbar slot zero exactly 64 wool blocks.
The two aimed stages fail reset immediately unless the initial raycast is a
block face on which the held wool can actually be placed.

The action is a `MultiDiscrete` tuple containing use/place, lateral movement,
longitudinal movement, sprint, sneak, yaw delta, and pitch delta. Jumping is
not available. The observation is a `Dict` with a 32-value state vector plus a
binary occupancy grid. The grid has fixed model shape `1x1x1x66`: shorter stages are padded on
the start side, while the last longitudinal cell is always the target block.
This fixed maximum shape lets one SB3 policy continue across every stage.

Success gives a large base and remaining-time reward. Falling below start
height and exhausting the gap-scaled tick limit truncate the episode with
large penalties. Shaping rewards only new three-tick-grounded longitudinal
progress and new bridge extension; invalid placements and lateral blocks are
penalized. Placing any block above the default bridge level applies a large
penalty and truncates immediately. Total per-tick cost remains below the fall
penalty at every horizon. The model receives 98 values in total (32 scalar
state values and 66 occupancy bits); block inventory count is task-internal.

Bridging uses retention-aware mastery rather than advancing from a short noisy
window. Rehearsal is balanced by PPO transitions, not merely episode count:
the trainer measures active and rehearsal episode lengths and adjusts the
rehearsal reset probability between 20% and 90% to keep about 30% of collected
transitions on mastered stages. Normally half of rehearsal resets target the
immediately previous stage and half sample all prior stages. If a retained
stage falls more than ten percentage points below its mastery requirement over
at least 30 rehearsal episodes, a retention watchdog focuses rehearsal on that
stage and targets 50% of PPO transitions until its latest 50 episodes recover
to the required rate. Rehearsal episodes update PPO but never count as evidence
for mastering the active stage. The active stage needs its configured
success rate over the latest 500 eligible stochastic-training episodes. It
must then pass an isolated deterministic validation of every stage from stage
zero through the active stage: 20 episodes for each fixed aimed stage and 100
episodes for each randomized stage. A failure blocks advancement and reopens
exploration before another attempt. Thus reaching a new stage also proves that
all earlier stages still meet their own mastery thresholds.
The fifteen-gap stage uses a 75% threshold; this avoids indefinitely blocking
the curriculum when the longer randomized bridge remains stable in the
70–80% range while still requiring all-stage deterministic validation.

Exploration entropy is adjusted automatically for bridging. `ent_coef` starts
at `0.005`, falls to `0.001` as the active-stage success rate approaches
mastery, and reaches zero only when the stochastic mastery window is ready. It
is raised again after deterministic validation failure, so a
premature deterministic but suboptimal policy is not permanently frozen. The
last stage is handled too: after 500 eligible episodes at 90% success, all
ten stages are deterministically revalidated and a checkpoint ending in
`_mastered_<steps>_steps.zip` is saved as the final certified policy.

Bridging PPO uses conservative continual-learning updates:
`learning_rate=1e-4`, `n_epochs=4`, `clip_range=0.15`, and `target_kl=0.02`.
These values are also applied when resuming an older compatible bridging
checkpoint, so a checkpoint does not silently retain the earlier aggressive
optimizer settings.

Evaluation uses dispersed deterministic reset seeds by default rather than
adjacent integer seeds. Use at least 100 episodes for a useful randomized-stage
estimate; ten episodes are too noisy for comparing nearby checkpoints. Add
`--stochastic` to measure the sampled policy used during PPO collection, or
`--seed-schedule sequential` only when reproducing the older adjacent-seed
behavior. Deterministic evaluation remains the deployment/reliability gate.

```powershell
python .\python\train.py --task bridging --num-envs 10 --total-timesteps 5000000 --output-dir .\models\bridging_v3
python .\python\evaluate.py --task bridging --model .\models\bridging_v3\final_model.zip --episodes 100 --verbose-steps
python .\python\evaluate.py --task bridging --model .\models\bridging_v3\final_model.zip --curriculum-stage 3 --episodes 100 --verbose-steps
```

V1 and V2 observation/action contracts are incompatible, so V1 model files must
not be passed to a V2 `--resume` or evaluation command.

The bridging reset clears and rebuilds a bounded arena around the player.
Use a disposable single-player world for the scripted controller, training,
and evaluation.

## Manual tick modes

For a manual FAST benchmark, run `/rl mode fast`, wait a few seconds, run
`/rl stats`, and return to `/rl mode vanilla`. FAST removes the 50 ms sleep
and advances the logical server clock by 50 ms on each loop iteration.
This can saturate one CPU core and accelerates world activity, including
entity AI and the day cycle. Client throughput still depends on the render
loop, so server ticks per second can be substantially higher. Use
`/rl syncbench` to measure the RL transition loop.

### Step through server ticks

```text
/rl mode step
/rl step 1
/rl step 10
/rl stats
/rl mode vanilla
```

`/rl step 1` permits one integrated-server tick. The client continues at its
normal cadence to process input, menus, chat, and network traffic. In the
synchronized benchmark, the client follows the action, server tick, and
observation sequence described above. If an older build leaves the
interface frozen in step mode, send `/mode vanilla` from the Python bridge
to restore normal ticking.

## Source layout

| Path | Role |
| --- | --- |
| `src/main/java/com/minecraftrl/mixin/` | Hooks for the client timer and vanilla server loop |
| `src/main/java/com/minecraftrl/tick/` | Tick modes, permits, and statistics |
| `src/main/java/com/minecraftrl/network/` | Reconnecting TCP client and NDJSON codec |
| `src/main/java/com/minecraftrl/command/` | In-game `/rl` commands |
| `python/minecraft_bridge.py` | Interactive TCP bridge for one or more clients |
| `python/tasks/` | Task action spaces, observations, rewards, and curricula |

## Build

```powershell
.\gradlew.bat build
```

The mod JAR is written to `build/libs/`. The [optimization
report](benchmarks/optimization_summary.md) describes the measured effect of
`lean-shared` and `-IdleWaitMode event` and the limits of long-run validation.
