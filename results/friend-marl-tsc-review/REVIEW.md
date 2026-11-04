# Independent review of the supplied marl_tsc archive

Reviewed 4 November 2026. The original ZIP and this project's active experiments
were not modified. Source and tabular results were copied to
`runs/friend-marl-tsc-review/marl_tsc`; model pickles were not loaded. The archive
hash is in `audit.json`. This is a code/data audit, not a fresh SUMO replication
of the friend's trained models. Five small code probes use synthetic inputs.

## Assessment

This is a useful independent implementation and a promising shielded-policy
experiment. Its raw results support substantial vehicle-delay gains in several
conditions. However, it does not yet demonstrate this project's specified
arrival-to-entry pedestrian criterion or an unconditional service guarantee.
The main blockers are a different pedestrian measurement, incomplete safety
diagnostics/occupancy protection, simulator recovery assumptions, and lack of
independent training replication. There is also a concrete summary-rounding
error: per-seed comparisons are not all passing.

## What was verified

- One synthetic corridor with three signalised junctions and 12 crossings.
- Two walking-population scenarios, each with a 3 by 3 vehicle/pedestrian demand
  grid: 18 settings on one geometry, not 216 different simulated networks.
- Six controllers, each with 180 evaluation rows (18 settings times 10 seeds),
  for 1,080 evaluation simulations. Every expected scenario/level/seed tuple is
  present exactly once. Seeds are 10–19.
- Vehicle/person demand counts match the reference for every paired evaluation.
  Route hashes are not archived, so identical route bytes are not independently
  established by the result files alone.
- Calibration metadata uses seeds 1000–1009, disjoint from evaluation. No budget
  was clamped to its feasibility minimum. Source defines separate training seeds.
  This verifies the configured splits, not the historical claim that results
  were never inspected before design choices were frozen.
- All reported pooled crossing p95 values and vehicle condition means were
  independently recomputed from raw CSV/GZIP files, without running analyze.py.
- Each learned controller passes all 216 pooled comparisons and all 108
  slow-walker pooled comparisons **for the recorded stationary-wait metric**.
  The minimum pooled all-walker crossing count is 230–236 across learned models.
- Every learned controller passes 2,154 of 2,160 per-seed crossing comparisons:
  **99.7222%, with six failures**, not an exact 100% pass rate.
- The three recorded individual budget exceedances all belong to IPPO-vehicle,
  slow-population vhigh-phigh, seed 12. They are 290 and 301 s at J1-W against
  a 49 s budget, and 289 s at J1-N against a 47 s budget. All three people are
  marked as normal-speed walkers within the mixed slow-population scenario.
  These rare but substantial exceedances remain compatible with a pooled p95
  pass; the archive does not contain the physical replay needed to independently
  verify the reported obstruction explanation for those exact events.
- Training files contain 150 PPO iterations, each collecting three independent
  environment episodes: **450 training episodes per controller**, each 1,800 s.
  The 180 figure is the number of evaluation episodes per controller. Each
  learned configuration has only one initialization/training run, seed 0.

## Positive vehicle results are broader than the chat summary

These percentages compare completed-vehicle `timeLoss + departDelay` against
the friend's pedestrian-actuated controller. Negative is better. They are
point estimates for one trained IPPO vehicle-reward model, not evidence about
variation between training runs.

Normal-walker population:

| Vehicle demand | Low pedestrian demand | Medium pedestrian demand | High pedestrian demand |
|---|---:|---:|---:|
| Low | -29.93% | -27.94% | -21.60% |
| Medium | -23.44% | -11.13% | +11.57% |
| High | -23.68% | +9.23% | +37.11% |

Slow-walker population:

| Vehicle demand | Low pedestrian demand | Medium pedestrian demand | High pedestrian demand |
|---|---:|---:|---:|
| Low | -28.53% | -22.25% | -14.49% |
| Medium | -19.87% | +0.49% | +44.90% |
| High | -15.34% | +17.83% | +46.95% |

Both IPPO variants improve this vehicle metric in 11 of 18 settings; both MAPPO
variants do so in 9 of 18. Each IPPO variant beats its corresponding MAPPO
variant's vehicle metric in all 18 settings. Thus "vehicle improvements only
when pedestrian demand is low" is too broad: low vehicle demand also permits
improvements at high pedestrian demand. This suggests studying joint vehicle
and pedestrian load, rather than blaming pedestrian demand alone.

The clearest positive case is IPPO vehicle-reward, normal walkers, low/low
demand: 76.129 to 53.347 s (-29.93%). All scheduled vehicles complete under both
controllers, with zero recorded teleport events over the ten matched runs. The
paired mean difference is -22.782 s; an exploratory paired demand-seed bootstrap
interval is [-24.793, -21.262] s. This interval conditions on one trained model.
It is not a training-replica interval, simultaneous selection-adjusted inference,
or confirmation of the arrival-to-entry pedestrian target.

At normal low-vehicle/high-pedestrian demand, the same model improves 86.652 to
67.938 s (-21.60%) with all vehicles completed, but one teleport event occurs
under IPPO and none under the reference. Keep that distinction visible.

## Findings requiring correction or additional validation

The README is more qualified than the chat message: it explicitly distinguishes
signal service from physical crossing entry and acknowledges statistical baseline
uncertainty. Preserve those limitations. Some top-level guarantee wording and
the claim of per-seed perfection still exceed what the implementation/results
establish.

### 1. Per-seed "100%" is a rounding error

`results/report/constraint_summary.csv` contains 99.72222222222223 for every
learned controller. `analyze.py:334` formats this with zero decimal places,
displaying 100%. The README/chat then interpret it as every comparison passing.

Example: normal, vmed-plow, seed 12, J2-E has reference p95 23.0 s. IPPO-vehicle
has 29.0 s, and IPPO-pedestrian has 30.75 s. The full list of 24 learned-controller
failure rows is in `verified_per_seed_failures.csv`. The pooled 216/216 result
still holds under its stated metric; pooled and per-seed criteria are distinct.

Fix the display to show exact numerator/denominator and sufficient decimals.
Do not remove quiet-crossing failures. A calibrated lower confidence bound does
not guarantee domination of every future finite-sample reference percentile.

### 2. Pedestrian waiting is not the specified event-based measure

`tsc/env.py:35–79` counts one second only when speed is below 0.1 m/s on a
walkingarea whose next edge is a crossing. It does not timestamp reaching a
geometric crossing entrance and then entering that crossing. Moving forward in
a waiting queue pauses its clock; stopping elsewhere on the walkingarea may
also be included. The bias relative to entrance-to-entry time is not necessarily
identical across controllers.

The tracker records an uncensored sample whenever a person/crossing pair
disappears from the walkingarea tracking set, without verifying that the person
actually entered that crossing. At the horizon it retains only active samples
with a positive stationary counter. There are no crossing-exit timestamps.

Reproduced code probes:

- A specified 10-second entrance-to-entry interval containing stop/go motion is
  recorded as five stationary seconds.
- A person removed without entering is recorded as uncensored.
- A still-present moving request with counter zero is omitted by finish().

These are deterministic counterexamples to metric equivalence, not estimates
of their prevalence in the archived SUMO runs. The archive lacks positions and
arrival/entry timestamps, so correct event waits cannot be reconstructed from
these CSVs alone. Add the event tracker, hand-check it, and recalibrate budgets.

`analyze.py:121–139` also includes censored waits as if completed observations.
Only three censored rows appear in the supplied fixed-time results and none in
the learned/reference rows, so this issue does not explain the learned pooled
pass rates. Nonetheless retain pending requests separately in the corrected
analysis rather than treating their lower bounds as completed waits.

### 3. Simulator recovery materially differs from our study

`tsc/env.py:156–164` explicitly enables gridlock teleporting after 300 s. Across
180 evaluations per controller, the raw files record:

| Controller | Teleport events | Runs with a teleport | Total unfinished vehicle trips |
|---|---:|---:|---:|
| Fixed-time | 0 | 0 | 0 |
| Pedestrian-actuated | 67 | 23 | 723 |
| IPPO vehicle | 289 | 75 | 2,634 |
| IPPO pedestrian | 518 | 87 | 1,640 |
| MAPPO vehicle | 439 | 126 | 4,518 |
| MAPPO pedestrian | 707 | 154 | 4,448 |

These are teleport events, not necessarily distinct vehicles. SUMO can remove
a vehicle from the physical network, advance it along its route and reinsert it.
Thus these runs include automatic recovery from blockage; their congested
throughput/delay cannot be equated with runs that disable such recovery.
This does not show that every improvement is caused by teleporting: the normal
low/low IPPO-vehicle example above has zero teleports.
[Official SUMO teleport documentation](https://sumo.dlr.de/docs/Simulation/Why_Vehicles_are_teleporting.html).

The command leaves pedestrian jam-recovery settings at defaults. Current SUMO
documentation describes obstacle-ignoring movement after a pedestrian jam,
with a 10-second threshold on crossings. Its activation cannot be quantified
from the archived CSVs; no jam traces are supplied and warnings are suppressed.
Run a shared, explicitly configured recovery sensitivity test for all controllers.
[Official pedestrian-model documentation](https://sumo.dlr.de/docs/Simulation/Pedestrians.html#jamming).

### 4. Identical nominal phase rules do not establish occupied-crossing protection

The same signal machine is used by baselines and RL, which is good. However:

- `tsc/signals.py:138–148` stops extending all-red after six extra seconds even
  if the vehicle box remains occupied. A probe with a constantly true occupancy
  callback reaches WALK after 11 seconds anyway.
- `tsc/signals.py:126–149` ends pedestrian clearance after a fixed duration and
  has no pedestrian-occupancy hold. A phase probe returns vehicle green after
  15 seconds regardless of whether someone would still be crossing.
- Design walking speeds are not enforced minimum realized speeds; distributions,
  dawdling and obstruction can produce slower traversal.
- The SUMO command does not enable junction collision checking, hides warnings,
  and does not save a collision audit. No collisions are established by this
  review; nor does the archive establish their absence.

Keep minimum timings and a common occupancy-aware safety guard. Log entry,
exit, residual vehicles and collision events. A finite wait guarantee requires
explicit feasibility assumptions: an indefinitely obstructed crossing cannot
always be served physically by a finite deadline while retaining safety.
[Official SUMO safety documentation](https://sumo.dlr.de/docs/Simulation/Safety.html).

### 5. RL attribution and algorithm comparison remain incomplete

The shield masks actions before policy sampling and stores that same mask/action
for PPO. This is a sound basic approach; I found no requested-versus-executed
action mismatch in that path. But the archive includes no matched shielded
queue/pressure baseline. The random-shield option exists; its claimed stress
test outputs are not present in this archive. Random actions are not a formal
worst-case adversary.

`shield_rate` counts observations where the shield removes at least one legal
action. It is not the fraction of policy choices that would have been overridden,
nor the fraction of free decisions eliminated. Recorded rates are approximately
13.65–14.32% overall. Add actual free-choice counts, proposed-action exclusions
and phase-time allocation before saying RL had little discretion.

The actor is shared across three junctions, with local observations and agent ID.
`tsc/ppo.py:58–62` changes reward scope as well as critic inputs: IPPO gets local
rewards, MAPPO gets the mean team reward. Therefore their difference cannot be
attributed solely to the centralized critic. Use a reward-scope ablation to
separate those explanations, and repeat training with independent seeds.

### 6. Reference controller and metrics need precise labels

`tsc/baselines.py:45–84` implements a legitimate push-button-style reference:
it finishes a fixed 30-second EW or 20-second NS green before serving a call.
It does not terminate that vehicle green early in response to queues or a
pedestrian request. This differs from our responsive baseline. Add a stronger
queue-responsive reference with the same safety machine before broad claims.

`veh_mean_time_loss` actually includes `timeLoss + departDelay` for completed
vehicles (`tsc/env.py:329–343`). Including insertion delay can be useful, but it
must be named and reported separately from pure SUMO timeLoss. Unfinished and
uninserted vehicles are merged into one count; full person-trip/crossing-exit
completion is absent. The all-demand delay measure is an additional useful
diagnostic, not a replacement for separate service counts.
[Official TripInfo field definitions](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html).

The archived network was generated with SUMO netconvert 1.28.0; our current
portable installation is 1.27.1. Runtime versions are not pinned by the supplied
requirements. Evaluation caches are accepted solely because an episodes CSV
exists (`evaluate.py:67–68`), without checking configuration/source/checkpoint
identity or coverage. Current v2 CSV coverage is complete, but future reruns
need experiment manifests to prevent accidental mixing of cached results.

## Recommended next experiment

Preserve this archive as an independent exploratory study. First correct and
validate event-based measurement, explicit recovery settings, occupied-crossing
protection and report formatting. Then calibrate budgets on separate data under
those same rules. Compare shielded IPPO, shielded MAPPO, shielded queue-responsive
control and pedestrian-actuated control on the same environment, initially
using a small development set. Retain all poor outcomes, teleports/recovery
counts and unfinished requests. Repeat training before method-level claims and
use a fresh locked evaluation set after implementation decisions are frozen.

The existing IPPO low/low checkpoint is a useful candidate for diagnostic
reevaluation, not a replacement for our four-controller study or proof that
the revised environment will retain its gains. Do not combine raw numbers from
the two implementations as if their topology, baseline, reward, safety,
inference and measurement definitions matched.

## Reproduction of this review

With the selected files extracted to the path described above:

```powershell
python -u scripts/review_friend_marl_tsc.py
```

Outputs: `audit.json`, `verified_vehicle_cells.csv`,
`verified_per_seed_failures.csv`. No model loading, new training, package
installation or new SUMO run is performed by that audit script.
