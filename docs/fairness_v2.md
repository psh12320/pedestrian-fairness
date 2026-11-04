# Pedestrian fairness v2: implementation record

**Completion update, 4 November 2026:** this neural study completed on the cluster (30 models × 512 episodes; 13,860 evaluations). [Shared evidence and artifact limitations](../results/fairness-v2-cluster-review/README.md). This document retains its original implementation history; the current sharing branch is `fairness-service-priority`.

Branch: `pedestrian-fairness-v2`. This is the next research iteration, not a replacement
of the completed study. Existing source, models and published results retain their
original measurement and control definitions. The working tree already contained
uncommitted previous-study work when this branch was created; branch creation did
not commit, discard or independently snapshot those files.

## Implemented system

- `src/traffic_control/fairness/` is a separate versioned package.
- Explicit EW/NS/PED requests share one safety machine and the original inspected
  SUMO movement conflict matrix. A request can proceed from EW directly to PED
  through yellow and all-red. Fixed-time, actuated and pressure-inspired baselines
  share minimums, maximums, transitions and occupied-crossing clearance.
- 0.2-second observations and one-second decisions. Clearance measures elapsed
  seconds. Entrance detection spans the full 4 m crossing width and +/-1.2 m along
  its axis. The previous centre-point disk missed outer walking strips.
- Per-crossing queue count, oldest and average pending age, crossing occupancy,
  per-approach vehicle features, downstream queues, phase and legal-action mask.
- Entrance-connected stopped queues; arbitrary distant stopped people do not
  automatically become requests. Repeated crossing visits, exits and lost requests
  are recorded. Missed approaches have unknown point waits and conservative bounds;
  they are never silently filled with zero.
- Vehicle service at measurement cutoff and simulation end; pedestrian completion,
  insertion and unfinished counts; crossing traversal and pending requests.
- 21 reproducible synthetic scenario definitions, Poisson arrivals, supported turns,
  slower/mixed walkers, bursts, ramps, platoons and heterogeneous driver sensitivity.
- Neural 64x64 local actors, local IPPO/global MAPPO critics, PPO/GAE, action masks,
  clipped updates, gradient clipping and scaled value losses. Per-crossing cost
  critics and projected multipliers implement constrained variants of both methods.
- Six configurations: vehicle, pedestrian weight 3 and constrained for each method.
  Constrained variants retain pedestrian reward weight 3 and add crossing costs.
  Atomic optimizer/model/RNG checkpoints, validation selection and resumable cached
  evaluation. Training demand/initialisation, calibration, validation and test seeds
  are disjoint. Paired comparisons and hierarchical bootstrap reporting are included.
- Executable load, warm-up-cohort and same-trajectory sampling audits. Failures,
  unknown waits and worsening service are retained and block success claims.

See [cluster/local training commands](fairness_execution.md) and the
[assumption register](fairness_assumptions.md). Full v2 experiments are not complete.
Observed checks and run locations are recorded in [verification](fairness_validation.md).

## Run the development baseline

From the repository root (portable Windows SUMO):

```powershell
$env:PYTHONPATH = (Resolve-Path .local/sumo-1.27.1/tools).Path
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue
python -m src.traffic_control.fairness.environment --controller actuated --smoke
python -m src.traffic_control.fairness.environment --controller fixed --smoke
python -m src.traffic_control.fairness.environment --controller pressure --smoke
python -m unittest discover -s tests -p 'test_fairness*.py' -v
```

Without `--smoke`, the development runner uses 300 s warm-up, a measurement cutoff
at 2100 s and an end at 2400 s. Logs and compact JSON outputs go to
`runs/fairness-v2/`; temporary network, route and TripInfo files are removed after
metrics extraction. An explicit `--output` keeps different experiments separate.
Neural tests use an existing PyTorch installation; no local GPU packages are
automatically installed. SUMO/TraCI require local socket access.

On 2026-10-31 the first ten new tests passed in 50.1 seconds, including three live
baseline episodes and an identical-seed actuated repeat (360 simulated seconds
each). Cars and pedestrians completed trips; all twelve crossings existed and
every junction served a pedestrian phase. Test log: `runs/fairness-v2-tests.log`.
The local Torch installation emitted a NumPy compatibility warning; tensor-only
interface tests passed, but the training environment must use compatible package
versions before adding NumPy-based rollout conversion. This is not a full learning
or physical-safety validation.

## Assumptions and limits

The 960/1920/2880 vehicles/hour levels are **provisional synthetic rates**, not
capacity-calibrated labels. The overload rate is a candidate, not yet a verified
5% service-gap condition. Pedestrian rates are network-wide person departures,
not per-crossing request counts. Sparse demand currently targets one walking OD
route: it must be audited before being called a sparse-crossing experiment.

Demand generation, geometry and car parameters are experimental controls rather
than observed Singapore data. Standard walkers use 1.3 m/s and slow walkers 0.8 m/s;
the clearance design speed is 0.8 m/s. No conclusions about age or disability can
be inferred from these speed groups alone. The driver sensitivity uses speed
factors 0.85/1/1.15 and sigma=0.3; these are assumptions, not a fitted population.

Jamming recovery thresholds exceed the run horizon so stalled pedestrians remain
stalled and visible in service metrics instead of moving through obstacles.
This is a changed simulator configuration shared by all new controllers. It does
not establish physical collision safety. The runner checks vehicle collision
notifications, signal conflicts and occupied crossings at each step.

All-run diagnostics include warm-up. The analysis module instead selects requests
reaching crossings during the measurement interval and follows them through the
300-second follow-up. Vehicle service is frozen at the cutoff for scheduled-departure
cohorts. Unknown waits, diverted requests and post-entry jams remain visible.

Entrance membership can be non-monotone. A same-trajectory audit found 0.5-second
differences with the earlier disk detector: the step size is not a universal error
bound. Conservative arrival bounds now begin with the first observed approach.
The full-width detector has a separate sampling audit. Geometric-definition error
is not covered by these timestamp bounds.

## Original gates and final status

1. Review the saved sampling/load audit findings and queue geometry before freezing
   confirmatory claims. The sparse case is explicitly a sparse OD route, not yet
   a validated single-sparse-crossing intervention.
2. Finalise demand/baseline timing assumptions and inspect validation learning curves.
3. Cluster execution completed using CPU jobs; SUMO is the dominant workload.
5. Five training seeds for six settings and twenty evaluation seeds completed:
   21 x 20 x (30 models + 3 baselines) = **13,860 evaluations**. See the linked
   archived results and the limitations of the compact downloaded bundle.
6. Ablations, report/presentation refresh, verified team contributions and resolution
   of the Team 28/38 discrepancy. Keep negative results and all failed-run counts.

The primary target remains a medium-demand vehicle gain with no worse p95 at any
crossing and no worse service. A pass is not guaranteed, and a pass in this setting
does not establish a deployed Singapore improvement.

References: [SUMO pedestrians](https://sumo.dlr.de/docs/Simulation/Pedestrians.html),
[TraCI crossing tutorial](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html),
[TripInfo](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html),
[MAPPO reference implementation](https://github.com/marlbenchmark/on-policy).
