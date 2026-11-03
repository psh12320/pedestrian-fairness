# Local signal-timing experiment

This study runs on one CPU worker while the separate guided/RND MAPPO study runs on the cluster. It uses no Torch and changes no cluster source, archives, checkpoints or evaluation seeds.

Engineering checks on 3 November 2026: all seven new unit tests and all nine short SUMO simulations passed. The smoke exercised every duration/controller combination, phase timing, event accounting and final report generation. The full study was then launched in a hidden below-normal-priority process. The initial Python and SUMO processes used about 68 MiB combined; peak use may be higher. Read `status.json` for live progress rather than treating this launch note as a completion claim.

Hypothesis: increasing the pedestrian admission period may let more arrivals join the same protected crossing service, reducing repeated clearance overhead; it may instead delay vehicles. Either outcome is informative. This is a timing ablation, not a learned algorithm or a reproduction of Gini-based MARL.

## Design

- Admission periods: 12, 18 and 24 seconds. This is the period during which people may START crossing; protected clearance and occupancy checks are separate and unchanged.
- Controllers: fixed-time, pedestrian-actuated and pressure-inspired, all at all three durations. The physical phases, minimum vehicle green, yellow/all-red and slow-walker clearance rules are shared.
- Settings: medium demand, pedestrian burst, slow walkers and short spacing/high demand. Synthetic geometry/demand, three junctions and 12 crossings. Short spacing changes demand as well as geometry.
- Four new development demand seeds: 910001–910004. No training and no reserved final test seeds.
- **144 full-horizon simulations**, 2,400 seconds each, measurement cohort 300–2,100 seconds with followup through 2,400.
- Main comparison: each controller at 18/24 seconds versus itself at 12 seconds, under identical demand and geometry. Secondary comparison: controllers versus actuated control at the SAME admission period. A timing gain must not be presented as an RL gain or used to give only RL a more favourable phase plan.

Report vehicle time loss with scheduled/completed/unfinished/uninserted trips; whole-person trip counts; every crossing's pooled mean/p95, entries, pending-at-cutoff and unresolved requests. Preserve event timestamps, walking-speed groups and phase histories in compressed records. Pool waits across demand seeds, not per-run p95s. Paired bootstrap resamples whole demand seeds; only four seeds makes intervals exploratory. Crossing estimates with fewer than 100 entries are inconclusive. Report every duration/controller, including failures.

## Resources and output

One worker, below-normal process priority when launched by the assistant. No new dependencies. Temporary simulation files are removed after each run; compressed records remain. Stop rather than consume the last 512 MiB of free disk or exceed 256 MiB of saved study outputs. Runtime depends on the laptop and concurrent apps; use observed completed-run times, not a promised deadline. Keep the computer awake for progress; interrupted runs resume from the last completed simulation.

The runner locks its output against concurrent writers, fingerprints source/configuration, checks matched demand/network hashes, audits vehicle/person conservation and crossing tails, and applies the existing per-tick safety assertions. An extra check verifies actual completed pedestrian admission and minimum phase durations. A failure stops the study and retains available simulator failure details.

Run/resume from the project folder:

```powershell
python scripts/run_local_timing_study.py
```

Progress:

```powershell
python scripts/run_local_timing_study.py --status
Get-Content runs/local-timing-study.log -Tail 12
```

Stop after the current simulation:

```powershell
New-Item -ItemType File -Path runs/local-timing-study/STOP -Force
```

Before resuming, remove that specific STOP file. Do not edit the simulation code/configuration during a run; a mismatch stops resumption. Results appear at `runs/local-timing-study/REPORT.md` and `summary.json` only after all 144 runs complete. `status.json` distinguishes running, stopped, failed and complete; merely seeing a process exit does not prove success.

Engineering smoke and unit tests:

```powershell
python -m unittest tests.test_local_timing_study -v
python scripts/run_local_timing_study.py --quick --output runs/local-timing-smoke
```

The smoke is nine 180-second simulations on one seed and cannot support scientific conclusions.
