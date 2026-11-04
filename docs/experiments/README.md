# Experiment catalogue

Updated **5 November 2026**. Start here to understand what was built, what ran,
what failed, and which evidence a teammate can inspect from a fresh clone.
The machine-readable [registry](registry.json) connects these studies to code,
reports and locally retained artifacts. The [artifact guide](ARTIFACTS.md)
explains what Git does and does not contain. The [engineering inventory](ENGINEERING.md)
covers smoke tests, retries and diagnostic folders.

## The question and the answer so far

Can a learned signal controller reduce vehicle time loss **without increasing
the pedestrian p95 wait at any crossing or worsening service completion**, relative
to pedestrian-actuated control? A secondary result is how many of the twelve
crossings improve; 6/12 is useful evidence of a trade-off, not success on the
original every-crossing target. Lower vehicle time loss is better, but completing
only the easy trips can misleadingly lower that average.

We built a single-junction milestone and a three-junction SUMO corridor with
twelve protected crossings. Fixed-time, pedestrian-actuated, pressure-inspired,
IPPO and MAPPO controllers share the applicable study's safety mechanism. IPPO
learns local value predictions; MAPPO can use network information for value
prediction during training. Both choose actions from local junction information.
The newer actions request EW, NS or pedestrian service. The common safety layer
handles minimum greens, yellow/all-red, pedestrian admission and clearance;
the actor does not choose an arbitrary pedestrian green duration.

**We have not demonstrated the original joint target.** Some learned policies
reduce vehicle delay, but their crossing tails or service counts deteriorate.
Several targeted interventions improve pedestrian outcomes relative to their
starting model, at a vehicle cost. Neither a negative result nor a successful
software test establishes that MARL cannot work in another setting.

## Read the latest evidence first

| Study | Completed work | What it established | Evidence |
|---|---|---|---|
| E15: demand and early service | **720 evaluations**, no training; four low/medium demand combinations; ten new demand seeds; all three saved training replicas | Lower demand did not produce a joint win. Earlier service improved all twelve crossing p95 point estimates relative to continued MAPPO, but vehicle time loss exceeded actuated in all four settings. | [Review and all-crossing plots](../../results/demand-service-review/REVIEW.md), [complete tables](../../results/demand-service-review/archived_REPORT.md) |
| E13: pedestrian fine-tuning | **876 simulations**: 576 training, 54 validation, 54 baseline, 192 evaluation; nine continuations | Teacher-guided continuation improves all twelve tails relative to its parent in all four settings, yet all twelve remain worse than actuated. | [Review](../../results/pedestrian-finetune-cluster-review/REVIEW.md) |
| E12: implementation repair trial | **1,396 simulations**: 1,024 training, 48 validation, 68 baseline, 256 evaluation; sixteen models | Controller memory and separate gradient clipping do not establish a reliable joint improvement. Fixing a real implementation issue need not solve the research problem. | [Review](../../results/audit-repair-review/REVIEW.md), [full tables](../../results/audit-repair-review/archived_REPORT.md) |
| E14: congested failure replay | **8 exact replays**, no training | Crossing occupancy can hold vehicle clearance for hundreds of seconds. Measuring only time before crossing entry misses a pedestrian stuck during traversal. | [Replay evidence](../../results/failure-mechanism-review/REVIEW.md) |

E15's vehicle results below are mean completed-vehicle time loss, in seconds.
The crossing count compares early service with **actuated**, not with its parent.
Use the linked review for paired uncertainty, all twelve tails and service counts.

| Vehicle / pedestrian demand | Actuated | Continued MAPPO | Early-service hybrid | Hybrid crossings no worse / 12 |
|---|---:|---:|---:|---:|
| Low / low | 35.60 | 33.83 | 41.47 | 5 |
| Low / medium | 45.04 | 38.74 | 48.24 | 8 |
| Medium / low | 40.20 | 34.06 | 45.02 | 4 |
| Medium / medium | 51.17 | 38.43 | 54.97 | 6 |

All 720 E15 runs completed all whole-person trips; twelve vehicle instances
remained unfinished across ten runs at the final horizon. A request pending at
the measurement cutoff but served during follow-up is a cutoff service deficit,
not permanent starvation. Such distinctions remain in the review and summaries.

## Full study history

The IDs below index evidence, not a total count of independent experiments.
Training, validation, evaluation, demonstrations, diagnostics and exact replays
must not be added together and called independent test runs. Follow-up seeds
are development evidence after inspecting earlier results. Do not pool old and
new measurement or safety protocols.

### E01 — Milestones and initial exploratory comparison

- Moving vehicles/pedestrians, deterministic demand, common signal safety,
  immediate crossing entry, queued entry and a request unserved at the horizon.
- Initial four learned controllers: 36 episodes each, one initialization,
  five evaluation seeds. The earlier stopped-approach wait definition is
  superseded. [Initial report](../../results/REPORT.md) is historical evidence.
- A separate measurement-resolution fixture observes the same 0.1-second SUMO
  trajectory at 0.1 and 1 second. Thirty entered events differ by 0.49 seconds
  on average (maximum 1.30), with two requests retained pending. This is a fixture
  result, not a universal error bound. [Audit](../../docs/measurement_safety_audit.md)
  and [trace data](../../results/measurement-resolution/).

### E02 — Repeated linear-policy comparison

- IPPO/MAPPO × pedestrian reward weights 0/3/10 × five training replicas:
  **30 models, 150 episodes each**. Validation selected the common budget.
- **4,290 locked evaluations**: thirteen scenarios × ten seeds 9001–9010 ×
  thirty learned instances plus three baselines. Scenarios include the 3×3
  vehicle/pedestrian demand grid, slow walkers, burst, overload and short spacing.
- Medium-demand IPPO weight 3: 33.3 s versus actuated 34.7 s;
  paired difference −1.4 s [95% CI −4.2, +1.5]; four crossings worse.
  None of the six learned configurations meets the original target.
- Fixed-time overload leaves 17.1% of vehicles unfinished/uninserted on average.
  Twenty-eight pedestrian-jam warnings in short-spacing runs limit physical
  safety interpretation; this older protocol allowed SUMO jam recovery.
- [Report](../../results/final-study/REPORT.md),
  [saved-output audit](../../results/reproducibility/final_output_audit.json),
  [excluded safety attempts](../safety_revision_v3.md).

### E03 — Separate Singapore-derived geometry sensitivity

- **330 evaluations** of the E02 controllers, ten seeds 9101–9110.
- Three approximate NUS–NUH junction centroids, straight connectors, synthetic
  side arms, left-hand traffic. Small public map/hardware extracts exist, but
  observed vehicle counts, pedestrian arrivals and LTA signal plans do not.
- The original OSM import lacked protected signal-controlled crossings; the
  evaluated network is a simplified safe reconstruction. It is **not a calibrated
  Singapore corridor or a comparison with the deployed LTA controller**.
- No learned setting meets the crossing target.
  [Report](../../results/singapore-transfer/REPORT.md), [limitations](../singapore.md).

### E04 — Large neural fairness-v2 study

- IPPO/MAPPO × vehicle/pedestrian/constrained objectives × five replicas:
  **30 models × 512 episodes of 1,200 seconds**.
- **13,860 evaluations**: 21 settings × 20 seeds 34001–34020 × 33 instances.
  Evaluation includes 300-second warm-up, measurement to 2,100 seconds and
  follow-up to 2,400 seconds. [Scenario rationale](../fairness_assumptions.md).
- Vehicle-only policies report very low completed-vehicle delay while leaving
  many pedestrians unserved; this is not a success. Penalties and per-crossing
  constraints do not deliver the joint target either.
- [Archived report and limitations](../../results/fairness-v2-cluster-review/README.md).
  The downloaded status collector truncated `comparisons.json`; that file is
  unusable. All-scenario absolute summaries were subsequently recovered in the
  service-priority review's summary. Full raw E04 outputs/models require the
  original cluster folder; the compact Git export is not a complete replay bundle.

### E05 — Service-priority pilot

- **Twelve models × 32 episodes** (384 training), four scenarios and two demand
  seeds, **128 evaluations**. Plain/priority IPPO and MAPPO plus conventional
  variants test earlier service to long-waiting crossings.
- Some crossing gains occur, but substantial vehicle under-service remains.
  Several crossing samples are sparse. This prompted the policy-choice audit.
- [Results](../../results/service-priority-cluster/REVIEW.md),
  [design](../service_priority_followup.md). Local smoke and medium-demand pilot
  outputs are supporting checks, not additional independent training replicas.

### E06 — Sampled versus greedy action audit

- **110 evaluations**, including uniform legal choices, plus 26 deterministic
  replay checks. No additional actor training.
- Near-uniform action probabilities can become repetitive starving decisions
  under greedy selection. Sampling restores much service but does not meet the
  joint target. Random control beating an arm diagnoses behavior, not PPO math.
- [Review](../../results/service-policy-audit/REVIEW.md),
  [protocol](../service_policy_audit.md). Actor weights are absent from that audit ZIP.

### E07 — Critic fitting and horizon diagnostics

- Local actuated-data checks use four training and two check trajectories, with
  separate horizon checks. Cluster checks use sixteen short simulations under
  two frozen learned behaviors, producing 24 fit files / 144 fitted critics.
- Normalization and time-to-horizon improve prediction; global inputs do not
  automatically help. Lower prediction MSE is not lower traffic delay: these
  diagnostics update the value estimator, not the controlling actor.
- [Local review](../../results/critic-checks/REVIEW.md),
  [cluster review](../../results/critic-checks-cluster/REVIEW.md),
  [design](../critic_checks.md). Raw trajectories/checkpoints are not in these ZIPs.

### E08 — Critic data volume

- **80 trajectories**, two frozen behaviors, four versus 32 training trajectories,
  eight check trajectories, three critic initializations, local/global inputs,
  128/512 fitting updates. Twelve fit files contain 72 fitted critics.
- All twelve reported critic gates pass at 512 updates. More varied experience
  improves value prediction; no actor or vehicle performance claim follows.
- [Review](../../results/critic-data-volume-cluster/REVIEW.md),
  [design](../critic_data_volume.md). Raw trajectory tensors are absent from the ZIP.

### E09 — PPO repair pilot

- IPPO/MAPPO × original / normalized-clock / normalized-clock-batch4 × three
  replicas: **18 models × 64 episodes**. **1,796 simulations** comprise
  1,152 training, 108 validation, 104 baseline and 432 evaluations.
- Four settings, six development demand seeds. No joint win. Better critic
  calibration does not establish better traffic control. Four-rollout batching
  has sixteen updates versus sixty-four at the same episode budget, a confound.
- [Review](../../results/ppo-repair-cluster/REVIEW.md),
  [design](../ppo_repair_pilot.md). The downloaded ZIP lacks actor weights and raw traces.

### E10 — Guided exploration, imitation and curiosity

- MAPPO control / age reward / teacher guidance / RND curiosity × three replicas,
  **128 episodes each**; three additional imitation-only models.
- **2,090 planned simulations**: 1,536 training, twelve demonstrations, six teacher
  validation, 72 learned validation, 104 baseline and 360 evaluation runs.
  Four settings; evaluation seeds 890001–890006.
- Control improves vehicles while worsening all crossing tails; guidance and
  imitation recover some pedestrian performance at vehicle cost. No joint win.
- [Review](../../results/guided-exploration-cluster-review/REVIEW.md) and
  [full tables](../../results/metrics-guide/guided_all_settings.md).
  Later review treats 1e-9-second arithmetic differences as ties: guided crossing
  counts are 4/12 medium, 5/12 burst, 9/12 slow, superseding archived 3/4/7 counts.
  This correction is not a change to the substantive non-inferiority target.
- Fifteen checkpoints are in the downloaded archive. Congested failures are
  retained and replayed in E14, not dropped from the analysis.

### E11 — Pedestrian admission timing sensitivity

- **144 conventional-controller evaluations**: 12/18/24-second admission × three
  baselines × four settings × four seeds. No RL training.
- Longer admission increases vehicle delay in all tested cells and worsens at
  least one crossing tail. This does not prove twelve seconds globally optimal.
- [Report](../../results/local-timing-study/REPORT.md), [design](../local_timing_study.md).

### E12 — Deep audit and implementation repair

The audit identified missing transition memory, a junction-collision checking
gap, platform-dependent demand bytes, missing insertion-delay reporting and
coupling of actor/critic gradient clipping. The completed factorial trial uses
IPPO/MAPPO × legacy/memory/separate-clipping/both × two replicas, 64 episodes
of 600 seconds, with evaluation seeds 960001–960004. Common verification repairs
apply to all arms; they are not an algorithm advantage. Medium MAPPO memory
versus legacy improves by 3.21 seconds, with paired CI [−17.73, +6.97] for the
difference, so it does not establish a reliable gain.
[Audit findings](../deep_audit_2026_11_03.md),
[completed review](../../results/audit-repair-review/REVIEW.md).

### E13 — Pedestrian-focused continuation

All three frozen 128-episode MAPPO-control parents already used pedestrian
reward weight 3. Each receives 64 further episodes under continue/tail/guided
arms; these are not vehicle-only parents. Four settings and evaluation seeds
980001–980004 compare all nine continuations with parents and baselines.
The cluster study is complete. The local duplicate stopped at **42/876** after
transfer and must not be pooled as another replication.
[Review](../../results/pedestrian-finetune-cluster-review/REVIEW.md),
[design](../pedestrian_finetune.md), [cluster instructions](../pedestrian_finetune_cluster.md).

### E14 — Failure mechanism replay

Eight exact replays compare three learned congested failures with actuated and
imitation controls. Long stationary crossing occupancy explains why clearance
remains active in those traces. A nearby stopped turning vehicle is a candidate
obstruction; no removal intervention proves it is the unique cause.
[Review](../../results/failure-mechanism-review/REVIEW.md).

### E15 — Low demand and early pedestrian service

Vehicle rates 960/1,920 per hour and pedestrian rates 400/1,200 per hour form four
settings. Evaluation seeds 1010001–1010010 are new development seeds. Three
replicas of original/continued/tail/guided MAPPO and an early-service wrapper,
plus fixed/actuated/pressure baselines, give eighteen instances × forty cases.
The wrapper uses the continued model but requests pedestrian service whenever
a request is pending and the action is legal. Empty pedestrian requests remain
possible. It changes service timing, not safety or learned weights.
[Review](../../results/demand-service-review/REVIEW.md), [design](../demand_service_test.md).

### E16 — Independent friend's implementation review

[The review](../../results/friend-marl-tsc-review/REVIEW.md) audits an external
`marl_tsc.zip`. Its stopped-person wait definition, reference controller, safety
timings and teleport settings differ from ours; its IPPO/MAPPO ranking cannot
be combined with ours. Four models each saw 450 episodes, one training seed;
216/216 pooled cells passing is not a hard individual guarantee (2,154/2,160
per-seed cells pass). Only our review is included here, not the friend's source
or checkpoints. Their work must be attributed separately if used in the report.

## Excluded, incomplete and proposed work

- E02 safety-failed/phase-mismatched attempts at seeds 7001–7010 and 8001–8010
  are excluded, preserved and described in the safety revision; 9001–9010 is
  the completed revised study.
- E04's episode-64 status is a historical progress snapshot, not a second study.
  Slurm submission-limit failures interrupted scheduling; completion requires
  output coverage, not merely an empty `squeue`.
- E13's local 42-run partial copy is stopped. The 876-run cluster copy is the
  evaluated evidence. Smoke, preflight, platform-retry and resumed jobs are not
  additional independent replicas.
- E15 preflights include a blocked local connection attempt and a replay
  serialization check repair before the successful frozen run. All 720 final
  evaluations are retained. The local audit trial likewise preserves its I/O
  recovery record.
- **Not run:** the proposed empty-PED suppression × early-service factorial,
  new horizon-matched actor training, a junction-coupling intervention, SIGN-MAPPO
  or a graph-policy replacement, and calibrated Singapore experiments using
  observed pedestrian counts/timing plans. Research discussion is not an
  implemented algorithm or a result. See [hypothesis register](../failure_explanation_protocol.md).
- Reserved final seeds 58001–58020 remain unused in the supplementary studies.
  Do not describe iterative development results as an untouched final test.

## How to interpret and write this up

Read the [metric dictionary](../metrics_and_results_explained.md) before comparing
tables: completed-trip delay excludes uninserted vehicles; crossing-entry wait
excludes traversal time; missing entries are not zero wait. Always report service
counts, per-crossing sample sizes, all twelve tails and uncertainty alongside
vehicle gain. Older linear and newer neural protocols are different studies.

The evidence supports a story about **vehicle–pedestrian trade-offs, service
timing, finite-horizon failures and limits of the tested learning procedures**.
It does not prove a unique reason all MARL methods fail, a deployment benefit,
or a hard fairness guarantee. Synthetic demand spans light load through overload
and slow walkers/bursts/short spacing, but is not calibrated observed traffic.
Some follow-ups have only two or three training seeds and four demand seeds.
Preserve those limitations, negative settings and failed interventions.

The original role assignments are not verified individual contributions. Team
members should record their actual work before submission; do not invent it.
Resolve Team 38 versus the proposal's Team 28 and confirm the course deadline.
Disclose LLM-assisted implementation, debugging, experiment orchestration,
analysis and writing, with team verification. The earlier ten-page PDF/slides
describe E02/E03 and are **historical drafts**, not a final account of these studies.
