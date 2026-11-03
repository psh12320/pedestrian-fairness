# Local SUMO experiment after the implementation audit

This is a new, controlled development experiment. It tests whether two identified implementation changes improve actual SUMO outcomes, rather than assuming that better code or a lower value-prediction error implies better traffic control. It preserves earlier negative results and does not use the reserved final test seeds.

## Frozen comparison

Both IPPO and MAPPO use normalized-clock value estimators, pedestrian reward weight 3, separate local actors, and the same physical signal phases, minimum greens, yellow/all-red transitions, admission duration and pedestrian clearance.

| Arm | Observation | Actor/critic gradient clipping |
|---|---|---|
| `legacy` | Previous 42-feature local observation | One shared norm limit |
| `memory` | Controller memory added, 47 local features | One shared norm limit |
| `separate` | Previous 42-feature local observation | Independent actor and critic limits |
| `memory-separate` | Controller memory added, 47 local features | Independent actor and critic limits |

The memory extension exposes the already-committed next phase and the clearance-empty timer. These fields primarily improve the critic's description of forced transitions; they are zero at ordinary eligible actor decisions. Added input weights start at zero while matching original feature columns and network functions, so the comparison does not obtain a different initial policy merely by adding dimensions. MAPPO still uses global information only in its critic.

This is a two-factor design. Predeclared paired contrasts are memory versus legacy, separate versus legacy, both versus memory, both versus separate, and both versus legacy. Both standalone corrections and their combination remain in the report. No successful arm, initialisation or demand seed is selected after viewing evaluation results.

All arms also use the audit's common collision-detection and platform-stable demand changes. These are common verification improvements, not learning interventions. New insertion-delay measures supplement, rather than replace, the original completed-vehicle time loss. The original vehicle-focused versus pedestrian-aware proposal comparisons remain separate; this diagnostic fixes reward weight at 3.

## Budget, scenarios and seeds

- 2 methods × 4 arms × 2 independent initialisations = **16 models**.
- **64 training episodes per model**, each 600 seconds, for **1,024 training simulations**. Every arm uses the same scenario/demand sequence within a replica and the same number of PPO update opportunities.
- Training cycles medium vehicle/medium pedestrian demand, medium vehicle/low pedestrian demand, a pedestrian burst, and short-spacing/high-demand traffic. Each setting appears 16 times per model.
- Evaluation uses medium demand, pedestrian burst, short spacing and slow walkers. Each model runs all four settings on four fresh demand seeds, for **256 evaluations** of the final checkpoint.
- Fixed-time, pedestrian-actuated, pressure-inspired and uniform legal-action controls run on all 16 evaluation-demand groups and the one validation group, for **68 baseline simulations**.
- Validation at episodes 16, 32 and 64 uses one medium-demand seed, for **48 validation simulations**. Validation records learning behavior; it does not select the final checkpoint.
- Full planned coverage: **1,396 simulations**, excluding explicitly labelled engineering smoke tests.

Initialisation seeds are 94001 and 94002. Training demand is `930001 + 1000 * replica + episode_index`, with episode index starting at zero. Validation demand is 950001. Evaluation demands are 960001–960004. Reserved final test seeds 58001–58020 remain untouched. Shared seeded action streams preserve categorical inference across training and evaluation; action randomness is not separately replicated.

Evaluation retains the 2,400-second horizon, a 300–2,100-second measurement cohort and followup until 2,400 seconds. Geometry and demand are synthetic. Short spacing also raises vehicle demand; it remains a combined stress case, not an isolated geometric explanation. Slow-walker evaluation tests transfer outside this study's training mixture.

The full design is intentionally bounded for local execution. Two training replicas cannot establish reliable method-level superiority. A future larger study must be frozen separately if these development results justify it. The current budget does not claim convergence, and collecting many total simulations across arms must not be confused with the experience of each actor.

## Measurements and interpretation

Record scheduled, completed, unfinished and uninserted vehicles beside completed-trip time loss. Record insertion waiting (`departDelay`) separately, and the sum of time loss plus insertion delay. These completed-trip measures still exclude the eventual delay of traffic not served by the horizon; the service counts must remain visible.

For every crossing, retain event waits, mean wait, pooled p95, entries by replica, requests pending at the cutoff, requests unresolved after followup and walking-speed groups. Crossing p95 pools events over demand seeds **within each training replica**, then averages replica tails. It is neither an average of per-run p95s nor an indiscriminate pool over trained replicas.

The unchanged descriptive proposal criterion requires lower vehicle time loss, no higher p95 at any of the 12 crossings, and zero strict service deterioration. Report individual service components and cutoff-only failures so that a pending request later served is not mistaken for a permanently unserved pedestrian. No success criterion is loosened after inspecting outcomes.

The report includes paired exploratory 95% bootstrap intervals over whole training replicas and demand seeds; paired comparisons use the same draws on both sides. Baseline-only comparisons resample demand seeds only. Two replicas and four demand seeds provide a weak uncertainty estimate, not a confirmatory claim. Any crossing with fewer than 100 pooled entries in a replica or reference is labelled sparse and inconclusive. Mean vehicle delay is a mean of run means, with completion counts shown beside it.

Track effective free decisions, action entropy, KL, both gradient norms and value-fit statistics. A correction may improve prediction without improving behavior. A vehicle gain may still worsen pedestrian tails. A change can help relative to legacy PPO and still lose against pedestrian-actuated or fixed-time control; these are distinct findings.

## Remaining hypotheses

This study does not change the reward, expose insertion backlog to the critic, train a continuing traffic objective, increase the corridor-through route share, or reproduce a published external traffic algorithm. Those were separate audit concerns. Keeping them unchanged allows the memory and clipping mechanisms to be tested without attributing a bundle of unrelated changes to one correction.

The existing guided/RND cluster study remains separate and unmodified. No report here should imply that it has finished unless its own evidence confirms completion.

## Outputs and verification

Observed engineering verification on 3 November: all 21 targeted learner,
reporting and independent integration/runner tests passed. The complete quick
pipeline finished 56 short SUMO runs. Replaying its saved MAPPO combined-arm
policy exactly reproduced outcomes and diagnostics without updating parameters;
resuming the completed pipeline preserved all 62 scientific artifacts byte for
byte. These are engineering checks, not evidence that the corrections improve
traffic. The full run was launched at 13:07 UTC (21:07 Singapore time), with two
below-normal-priority CPU workers; this note is not a completion claim.

Read compact progress without importing Torch:

```powershell
python scripts/collect_audit_trial_status.py
```

Live logs are `runs/audit-repair-trial.log` and
`runs/audit-repair-trial-errors.log`. A status file marked `running` can be stale
after a crash; compare its update time and check the recorded process. The
runner writes `complete` only after full coverage checks and final reporting.
Keep the computer awake. It checkpoints after each completed training episode
and validation; interrupted work resumes from the last atomic checkpoint.

To request a stop at the next simulation boundary, create
`runs/audit-repair-trial/STOP`. Remove that specific marker only when intentionally
resuming. An unexpected worker failure also creates it to stop the other worker;
inspect the error before resuming. Do not edit fingerprinted source/configuration
while the study runs. A compact `source.zip` in the run folder preserves the
exact implementation for later replay.

The runner records a frozen source/protocol fingerprint, atomic checkpoints, matched demand/network hashes, final-model completion markers and individual evaluation files. `REPORT.md` and `summary.json` are final outputs only after all prescribed models and evaluations pass coverage and integrity checks. A disappearing process, a partial learning curve or a passing smoke test is not a completed scientific result.

The report includes all controllers, contrasts and crossing results. Saved raw per-run waits make the crossing summaries independently reproducible. Compact records/checkpoints and temporary simulator files avoid large trajectory dumps on the laptop.

Reporting regression checks are runnable from the project folder:

```powershell
python -m unittest tests.test_audit_trial_reporting -v
```

The seven reporting tests passed during implementation. They check two-replica paired intervals, insertion-delay transparency, unchanged cutoff-service rejection, exact weighted versus explicit crossing-tail resampling, missing-measurement handling, distinct whole-horizon counts and final-report/validation coverage gates. Report generation also validates raw event-file hashes when present. These are engineering checks, not evidence of a traffic improvement.

## Run and monitor

Run from the project root using the existing CPU Python/SUMO installation. The runner locates this workspace's portable SUMO tools; no new packages or downloads are required.

```powershell
# Engineering smoke: eight models, four short episodes each; not research evidence.
python -m src.traffic_control.audit_trial.pilot --quick --workers 2 --output runs/audit-repair-smoke

# Frozen full local experiment. Reuse this command to resume unchanged source/output.
python -m src.traffic_control.audit_trial.pilot --workers 2 --output runs/audit-repair-trial

# Read progress without starting another worker pool.
python -m src.traffic_control.audit_trial.pilot --status --output runs/audit-repair-trial

# Rebuild the final report only after every required output is present.
python -m src.traffic_control.audit_trial.pilot --stage report --output runs/audit-repair-trial
```

Two workers keep CPU and memory use bounded. The initial planning estimate was roughly two to three hours, but actual laptop runtime depends on congestion and competing applications; it is not a completion promise. Use completed-run counts and observed timings. Keep the computer awake. The runner enforces a 512 MiB output cap and stops before disk free space falls below 512 MiB.

The source fingerprint includes runtime Python files. Do not edit those files during a run; use its saved source snapshot or a new output/protocol for subsequent changes. Existing cluster source and historical results remain separate. `REPORT.md` is generated only after successful coverage checks; all intermediate states must be described as running or incomplete.
