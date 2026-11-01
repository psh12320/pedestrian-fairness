# Cluster checkpoint review: episode 64

Snapshot collected 1 November 2026, 00:48:20 Singapore time.
Source: `runs/fairness-cluster-status.zip`. This is an intermediate validation
review, not the final held-out evaluation or a completed research result.

## Execution checks

- Preparation and all 30 training tasks have Slurm state `COMPLETED`, exit `0:0`.
- All 30 model histories end at episode 64 and report a saved latest checkpoint.
  Checkpoint binaries are intentionally not included in the status bundle.
- Full-budget completion markers are absent as expected: the budget is 512 episodes.
- All 20 calibration runs and 10 validation baselines are present.
- Calibration has zero unknown crossing waits; the least-sampled crossing has
  930 observed entries across calibration runs.
- No recorded episode failure files are listed. Latest logged loss, entropy and
  gradient statistics are finite for all models.
- The saved source/configuration hashes exactly match the current local files.
- Runtime: Python 3.12.3, CPU PyTorch 2.3.0, SUMO 1.27.1.
- No active project jobs were listed at collection time. Evaluation is 0/420
  matched-demand groups, and no final report exists yet.
- Initial missing-package checks in the preparation log are installation probes;
  preparation subsequently completed successfully.

## Early validation outcomes

Actuated control averaged 50.519 seconds of completed-vehicle time loss across
the ten validation seeds. Each learned row below averages its five replicas;
each replica's stored selection score already averages the same validation seeds.

| Configuration | Vehicle time loss (s) | Service/measurement penalty | Positive worst-crossing p95 penalty (s) |
|---|---:|---:|---:|
| IPPO vehicle | 19.849 | 1991.10 | Not used for this objective |
| IPPO pedestrian | 53.027 | 1.72 | 2.78 |
| IPPO constrained | 53.483 | 1.12 | 2.78 |
| MAPPO vehicle | 19.496 | 1991.10 | Not used for this objective |
| MAPPO pedestrian | 53.180 | 1.24 | 2.78 |
| MAPPO constrained | 53.792 | 1.52 | 2.78 |

The service/measurement penalty is a composite selection diagnostic, not a count
of unique unserved people. The crossing column averages per-run nonnegative maxima;
it is not the final pooled crossing-p95 estimand or a confidence interval. A zero
crossing term for vehicle objectives is disabled scoring, not evidence of fairness.

None of these episode-64 models satisfies the stored validation feasibility checks.
The vehicle-only delay gains accompany severe service violations. The pedestrian
and constrained variants have much smaller service penalties, but at this stage
their vehicle loss is higher than actuated and their crossing-tail penalty is
positive. More training may or may not improve these outcomes; do not claim success.

## Continuation

With the snapshot's empty queue and successful first stage, the next dispatcher
should start at `train 128 0` using `scripts/queue_fairness_next.sh`. Confirm there
is no subsequently submitted active chain before launching it once. This resumes
saved checkpoints, advances all remaining training stages and then evaluation.
Use a fresh dispatcher submission rather than an `afterok` dependency on an old
completed job ID that may no longer be retained by the scheduler.
