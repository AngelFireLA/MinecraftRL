# MinecraftRL implementation and measured results

Updated 2026-09-20. The measured combined configuration is now the default:
`lean-shared`, CPU, and HeadlessMC idle wait `event`. The planned long stability
check did not finish; a later three-minute unprofiled run completed successfully. No PPO algorithm,
reward, action frequency, or task difficulty was changed for this optimization.

## What changed

- **Java idle waiting:** renderless SYNC clients wait for bridge work between
  actions. A generation counter prevents a missed notification, with a 10 ms
  maintenance wake. Active action ticks and reset phases continue normally.
  This is the default; pass `-IdleWaitMode off` to select the reference behavior.
- **Lightweight Python workers:** `--vec-env lean-shared` uses standalone
  environment processes, shared observation buffers, and authenticated Windows
  named pipes. Workers avoid importing Torch/SB3, keep their numerical-library
  thread count at one, and retain parallel environment/task processing.
  Ordinary worker RSS was about 38 MiB versus roughly 578 MiB after the old
  training-registry import; aggregate measured process RSS is reported below.
- **Useful profiling:** policy inference, dispatch/receive, callbacks, rollout
  buffers, minibatch conversion, PPO, socket reads/JSON decoding, and Java phases
  can be timed separately. Resource samples distinguish trainer, workers, and
  clients. `throughput.json` also works without detailed profiling. Checkpoint
  save/load remains supported when profiling is enabled.
- **Benchmark supervision:** manifests record configuration, dependency versions,
  source/mod hashes; owned process trees are cleaned up, and the 92% system-memory
  guard remains enabled. Fixture and sustained failures now propagate explicitly.

## Confirmed paired measurements

Existing `bed_defense_break_v2`, 10 clients, CPU policy, 1 GiB maximum heap/client,
25,600 transitions per run, two warmup rollouts omitted, seeds 301/302/303.
The user's continuing CPU-intensive research workload was left running during
this campaign. Both configurations used the same profiling setup. These are
warmed rollout + PPO rates, excluding client startup and final checkpoint save.

| Metric | Reference: shared / off | Candidate: lean-shared / event |
| --- | ---: | ---: |
| Median transitions/s | 631.87 | 1,164.03 |
| Median vector-step p95 | 20.27 ms | 13.00 ms |
| Median PPO + between-rollout time | 1,127.05 ms | 468.52 ms |
| Median mean reset time | 4.02 ms | 2.62 ms |
| Maximum owned-process RSS across paired runs | 16.67 GiB | 11.26 GiB |

The median warmed throughput gain is **84.2%**. Maximum measured owned RSS fell
by **5.41 GiB (32.5%)**. Summed process RSS includes shared pages and launcher
processes; it is not identical to physical machine RAM or Java heap usage.
Paired final last-100 success differences were **+11, +2, +4 percentage points**.
This passed the short learning guard; it does not establish long-run sample
efficiency or learning equivalence.

The four-way seed-301 screen isolated the causes:

| Backend | Idle mode | Transitions/s | Peak owned RSS |
| --- | --- | ---: | ---: |
| shared | off | 631.9 | 16.67 GiB |
| lean-shared | off | 637.4 | 11.35 GiB |
| shared | event | 1,175.2 | 16.57 GiB |
| lean-shared | event | 1,168.6 | 11.07 GiB |

Idle waiting provides most of the speed gain; lightweight workers provide most
of the memory reduction. Earlier action/reset timers missed CPU consumed while
clients waited for Python. A representative baseline PPO interval consumed
about 16.8 aggregate Java CPU-seconds in roughly 1.1 wall-seconds. Event waiting
reduced that Java work to about 0.03 CPU-seconds, while PPO took roughly 0.45 s.
Freeing CPU for Python is the useful change here. Reset code was not rewritten;
its lower measured time is consistent with reduced contention.

### Repeat the paired comparison

Finish [setup](../../docs/setup.md) and close any MinecraftRL clients and trainers
using ports 8765–8774. Run from the repository root with `JAVA_HOME` set. The
campaign creates any missing instances, alternates configurations, and writes
the results to `runs/bed-defense-paired/results.json`.

```powershell
$comparison = @'
[
  ["shared", "off", 10, 301, 25600],
  ["lean-shared", "event", 10, 301, 25600],
  ["lean-shared", "event", 10, 302, 25600],
  ["shared", "off", 10, 302, 25600],
  ["lean-shared", "event", 10, 303, 25600],
  ["shared", "off", 10, 303, 25600]
]
'@
New-Item -ItemType Directory -Path .\runs -Force | Out-Null
Set-Content -Path .\runs\bed-defense-paired.json -Value $comparison -Encoding Ascii
.\.venv\Scripts\python.exe .\RLCraft\python\optimization_campaign.py `
  --runs .\runs\bed-defense-paired.json --output .\runs\bed-defense-paired
```

## How many clients

Single-seed screen with lean-shared/event:

| Clients | Transitions/s | Vector-step p95 | Peak owned RSS |
| ---: | ---: | ---: | ---: |
| 8 | 1,102.7 | 12.51 ms | 9.01 GiB |
| 10 | 1,168.6 | 13.00 ms | 11.07 GiB |
| 12 | 1,137.9 | 17.46 ms | 13.35 GiB |
| 14 | 1,114.5 | 20.75 ms | 15.58 GiB |

Ten remained the best tested count. Eight sacrifices about 6% in this screen
and saves about 2 GiB, making it a useful setting when other applications need
memory. Twelve/fourteen were not selected or repeated because they did not pass
the throughput/tail-latency criteria. More free RAM does not imply that more
instances will improve throughput.

## Validation limits and failures retained

- The first screen (`optimization_screen`) failed at checkpoint save because of
  profiler method restoration. That was corrected and covered by a save/load/
  resume test. Its results are excluded; `optimization_screen_v2` is the valid screen.
- The original profiling-off run completed at **620.3 warmed transitions/s**.
  System memory rose to 82.3% during that later run. This is inconsistent with
  the preceding short paired results and does not establish profiler overhead.
- The planned 15-minute run was **aborted by the 92% system-memory guard after
  230.7 seconds including startup**, with 11.48 GiB peak owned RSS. At the middle
  and end of its resource log, owned RSS stayed around 11.3 GiB while total
  system memory rose from 84.8% to 92%. This shows broader memory pressure;
  it does not prove the source of that pressure or certify long-run stability.
  The user subsequently reported that Chrome and Discord were running, which
  could have contributed; per-application attribution was not recorded.
  The run reached the `exposed_random` curriculum stage. Its results must not
  be presented as a completed 15-minute pass or a fixed-stage A/B comparison.
- Five strict stage-6 reset checks passed. The live PvP fixture retained a live
  20-HP opponent without attack, then observed attack damage, opponent death,
  and a fresh live opponent after reset. Player death was not exercised.
  These are correctness checks, not a new PvP training throughput benchmark.
- Focused Python tests passed for worker isolation, Box/Dict observations,
  terminal/truncation/reset handling, error cleanup, benchmark summaries,
  trainer/environment behavior, and profiled checkpoint save/load/resume.
  The mod build and Java self-tests passed. No full suite was rerun on resume.

## Resume check: normal training without profiling

On 2026-09-20, one bounded run used lean-shared/event, 10 clients, CPU, 1 GiB
heap, seed 305, with both Python and Java profiling disabled. It completed
**250,890 transitions in 180.216 warmed seconds: 1,392.2 transitions/s**.
Peak owned-process RSS was **11.12 GiB**, peak total system-memory utilization
was **88.7%**, and the final model was saved successfully. The supervisor
completed cleanup. The final recorded curriculum stage was `one_layer_fixed_aimed`.

This confirms operation without profiling and gives a short sustained pass.
It is not paired with a baseline under the same current load and stage mix,
so it does not replace the 84.2% paired comparison or quantify profiler overhead.
It also does not complete the original 15-minute validation. No background
applications were stopped by the benchmark supervisor.

[Resume results](optimization_resume_unprofiled/results.json).

## Decisions for the next PvP task

Use the explicit configuration below to try the measured improvements. Keep
CPU policy execution for the current small structured MLP; the headless game
simulation and environment logic are CPU work, so the mobile RTX 4070 does not
address the measured bottleneck. Earlier CUDA tests lost on the old workload;
revisit only for a materially different policy or batch/device arrangement.

Keep the existing 1 GiB heap. Earlier 768 MiB/1.5 GiB experiments did not justify
changing it. Do not fork SB3 yet: measured minibatch materialization was below
1% of total rollout/PPO wall time, well below the 5% investigation threshold.

The new PvP task should use the lightweight task factory as well as the training
preset registry. Measure its actual combat/reset workload before transferring
these absolute FPS figures. To select the reference behavior, use `--vec-env shared` and
`-IdleWaitMode off`. The lean backend currently requires Windows.

Python terminal (start first):

```powershell
Set-Location .\RLCraft
python .\python\train.py --task bed_defense_break_v2 --num-envs 10
```

Client terminal:

```powershell
Set-Location .\HeadlessMC
.\Start-Instances.ps1 -Count 10 -HeapMb 1024
```

For reduced memory use, set both counts to 8. Add `--profile` only when collecting
phase diagnostics; normal runs still write warmed `throughput.json` in the model
output directory. `--vec-env shared`, `--device cuda`, and
`-IdleWaitMode off` remain available as explicit alternatives.

## Artifacts

- [Paired aggregate results](optimization_final.json) (inspect the sustained status).
- [Four-way screen](optimization_screen_v2/results.json).
- [Additional paired seeds](optimization_confirm/results.json).
- [Instance scaling](optimization_scale/results.json).
- [Original profiling-off run](optimization_profile_off/results.json).
- [Aborted sustained run](optimization_sustained/results.json).
- [Live reset/combat fixture](optimization_fixture/results.json).

The supervisor writes per-run logs, manifests, and resource samples locally.
