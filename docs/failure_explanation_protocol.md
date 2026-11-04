# Explaining the vehicle–pedestrian trade-off

**Status, 4 November 2026:** the eight congested replays, repair trial, fine-tuning and 720-run demand/service test are complete; see [catalogue E12–E15](experiments/README.md). Proposed horizon-matched actor training, junction-coupling interventions and new architecture tests below remain unrun. Hypotheses are not established unique causes.

Prepared 3 November 2026. This is an evidence map and a staged development
protocol, not a claim that the original proposal target has been met. Existing
studies and unsuccessful runs remain part of the record. Reserved final seeds
58001–58020 are not used for diagnosis or tuning.

## The question and the result to explain

The proposal asks whether local IPPO/MAPPO signal controllers can reduce
vehicle time loss while keeping the pedestrian 95th-percentile wait no higher
at **every crossing**, relative to pedestrian-actuated control, without reducing
service. Lower mean pedestrian wait alone does not answer that question.
Counting how many crossings meet the target is a useful secondary outcome;
it does not replace the original all-crossing criterion.

The earlier guided-exploration study provides evidence of learning, not universal
failure. At medium demand, its pedestrian-penalised MAPPO control reduced mean
completed-vehicle time loss from 50.55 to 40.61 seconds (approximately 19.7%),
while all 12 crossing p95 point estimates worsened. Guided MAPPO reached 49.57
seconds, with four crossings no worse after arithmetic-roundoff correction
(the archived report said three) and a worst increase of 1.53 seconds;
its vehicle difference interval includes zero. Imitation-only reached 51.22
seconds, seven crossings no worse, and a worst increase of 1.07 seconds.
All retain failures under the original joint criterion.

These are three training replicas and six paired development-demand seeds.
P95s pool events over demand seeds within each replica, then average over
replicas. They are point estimates, not simultaneous guarantees. This study
contains MAPPO only. Its results must not be attributed to IPPO or conflated
with the earlier four-controller proposal comparison.

Evidence: [independently checked guided results](../results/guided-exploration-cluster-review/REVIEW.md),
[repair-study results](../results/ppo-repair-cluster/REVIEW.md),
[implementation audit](deep_audit_2026_11_03.md),
[timing experiment](../results/local-timing-study/REPORT.md).

## Evidence strength: what can already be said

| Explanation | Existing evidence | What it does not establish |
|---|---|---|
| Vehicle and pedestrian service compete for signal time | In the separate local timing study, actuated spends about 50.2% of medium-demand junction time in pedestrian admission/clearance, versus 27.5% for fixed time. Increasing admission from 12 to 18 or 24 seconds increased vehicle loss in every tested setting/controller combination, with at least one worse crossing tail. | No proof that simultaneous improvement is impossible, or that 12 seconds is globally optimal. |
| The trained reward is a proxy for the proposal target | The base reward penalises local vehicle queue, external uninserted backlog and pending pedestrian count; it does not directly constrain all crossing p95s. Adding age cost improves medium-demand tails at 11/12 crossings versus MAPPO control but costs 2.84 seconds vehicle loss. | The age intervention is evidence about this reward and budget; it does not prove a particular constrained algorithm will solve the target. |
| Teacher guidance changes the trade-off rather than automatically dominating the teacher | Imitation has about 96.8% sampled agreement with the teacher at medium evaluation; guided PPO about 94.9%. Guided versus imitation reduces vehicle loss by 1.65 seconds, while worsening p95 point estimates at 8/12 crossings. | No wholesale imitation failure. Pretraining and continuing guidance were bundled; their individual effects are not identifiable. |
| Training may be insufficient for robust traffic learning | Each actor has only 3,969–4,851 free decisions over 128 training episodes. On fixed validation demands, unguided control vehicle loss changes from 60.19 at episode 64 to 41.46 at 128. | Not convergence, nor proof that more training improves the joint criterion. Different historical studies have different seeds. |
| Critic information and optimisation have identifiable weaknesses | Missing committed-phase/clearance memory is confirmed by indistinguishable-observation counterexamples. Joint clipping couples actor and critic gradient norms. Normalisation previously improved prediction error without solving traffic control. | Counterexamples and software tests alone do not establish the size of any traffic effect. The current factorial trial tests that. |
| The corridor gives limited opportunity for coordination | Only 22.25% of vehicles in six audited medium cohorts traverse all three junctions; the rest use one. MAPPO's critic sees the network but predicts local returns. | Not a causal explanation for IPPO/MAPPO parity until route share and reward are varied independently. |
| Averages conceal unserved demand | In guided-study congestion, actuated completes about 33% of scheduled vehicles, MAPPO control 71%, fixed time 95%. Some learned runs have zero entries at a crossing and many unserved requests. | Low completed-trip time loss is not low total scheduled-demand delay. Finite pooled p95 does not show that every run provided service. |
| Rare clearance failures need simulator-level diagnosis | Three archived learned runs have 76, 105 and 130 unresolved measured crossing requests and prolonged total clearance time. | Aggregate phase durations cannot identify a continuous stall, an obstructing person/vehicle, or the cause of obstruction. |

For the software checks already performed, reward alignment in time, legal
action masking, gradient sign, episode boundaries and checkpoint restoration
passed. Small positive-control tasks learn. Therefore the statement “the
learning algorithm is completely broken” is unsupported. These checks do not
prove the entire simulator or learner error-free.

## Stage 1: identify the congested failure mechanism

Replay these *already observed* failures, without modifying their checkpoints,
demand, signal settings, simulator version or categorical action streams:

- MAPPO age replica 2 / demand 890004;
- MAPPO guided replica 1 / demand 890004;
- MAPPO guided replica 2 / demand 890005.

Use the matched actuated controllers on both demands and imitation-only
replicas 1/2 on their corresponding demands: eight distinct simulations.
`scripts/replay_guided_failures.py` extracts hash-verified historical source
into a separate directory under `runs`, loads only the relevant checkpoints,
and observes positions and signal state without changing decisions. It verifies
the demand hash after CRLF normalisation, network identity, all measured
outcomes and diagnostics against the archived run. If reproduction fails, stop
causal interpretation and investigate the difference first.

Record continuous clearance intervals, individual stationary crossing
occupancy, pending-request events, internal vehicle positions during prolonged
stalls, and whether the actor has any legal choice. Distinguish:

1. A request waits while the actor has opportunities to select pedestrian
   service: investigate the learned scheduling decision.
2. The actor has no available action because a crossing remains occupied:
   investigate pedestrian/vehicle interaction and clearance release.
3. The underlying pedestrian motion is obstructed: identify the obstruction
   from trajectories before naming it as the root cause.

Exact replay establishes a mechanism in those runs, not its population
frequency. Keep the original settings in the replay, including historical
collision checking; it is not a new safety certification. If a geometry or
simulator-setting correction is justified, make a separately labelled
all-controller sensitivity experiment. Do not drop failed seeds or release
vehicles onto occupied crossings to improve scores.

SUMO documents pedestrian jamming and an automatic recovery mode that permits
movement through obstacles. Our experiments set recovery time beyond the run
horizon. That preserves obstruction behaviour but may expose persistent jams;
turning obstacle-ignoring recovery on is not a physical safety fix.
[SUMO pedestrian interactions and jamming](https://sumo.dlr.de/docs/Simulation/Pedestrians.html#jamming).

## Stage 2: finish the existing controlled learning experiment

**Initial replay result:** guided replica 2 / demand 890005 exactly reproduces
the archived traffic measures. Person p000076 remains stationary on crossing
:C0_c2 from 315.8 to 2400 seconds. C0 stays continuously in protected clearance
from 320 to 2400 seconds, with no legal actor choice. A stationary northbound
vehicle occupies internal edge :C0_5 in the prolonged-stall snapshots; it is a
candidate obstruction, not yet a fully isolated root cause. Four verification
tests ensure floating diagnostic roundoff cannot excuse altered traffic
metrics. Seven matched replays were launched sequentially on one low-priority
worker. See the [replay evidence](../results/failure-mechanism-review/REVIEW.md).

This reveals a necessary distinction for the safety audit: conflict-free
signal indications do not by themselves establish that vehicles from the
previous phase have physically cleared the junction before pedestrian entry.
Inspect residual vehicle occupancy at admission and why a vehicle remains
stopped. Preserve protective clearance during diagnosis. Any physical-entry
guard or geometry correction must be common to all controllers and evaluated
in a separate matched study, not silently introduced into a replay.

The stationary person's arrival-to-entry wait is only 1.4 seconds, but the
person does not finish crossing. Report entered-but-not-exited visits and
person-trip completion beside queue waiting, especially for stalled cases.
This does not change the proposal's wait definition; it supplements it with
evidence of successful crossing service. The rare lockout does not explain the
ordinary medium-demand trade-off where all measured requests were served.

The [currently running factorial trial](audit_trial.md) tests both IPPO and
MAPPO with legacy/memory observations crossed with joint/separate gradient
clipping. Sixteen models, two initialisations per arm and 1,396 total simulations
are planned. Every arm has the same 64-episode training budget and matched
demands. The final checkpoint is fixed in advance. No runtime source is edited
while this fingerprinted trial is running.

For each factor report value fit, free decisions, actor/critic gradient norms,
action probabilities, vehicle service and loss, and all crossing tails. A
prediction improvement without a traffic improvement is itself informative.
Require matched outcome improvement to attribute a traffic benefit to a repair.
Retain unchanged and worsened outcomes. Two initialisations make the effect
estimates exploratory; absence of a clear effect cannot rule the mechanism out.

Do not claim that a repaired arm has converged at episode 64: the separate
128-episode guided study has visible late learning. If a longer comparison is
necessary, extend **all four arms of both methods** under the same predeclared
budget, rather than training only the apparent winner longer.

## Stage 3: one matched training-distribution experiment, if needed

Only after the active trial and failure diagnosis, freeze a new protocol for
the strongest remaining training-design question: empty-start training versus
longer traffic episodes. Fix one implementation based on engineering validity,
not favourable held-out scores; apply it to both methods and both conditions.

- Condition A: 128 independent 600-second episodes.
- Condition B: 32 independent 2,400-second episodes, collected in four
  600-second rollout chunks each, with value bootstrapping across nonterminal
  chunk boundaries and no bootstrap at the declared episode termination.
- Both: 76,800 simulated training seconds per model, 128 update opportunities,
  identical optimisation settings and reward, three independent initialisations,
  and the same mixture of four training scenarios. Chunked updates must not
  repeatedly train on overlapping transitions. Log actual free decisions too.
- Compare final checkpoints on a new, frozen development seed set; keep the
  reserved final test untouched. Pair evaluation demand and action streams.

This isolates the *package of reset frequency and terminal horizon* while
holding simulated time and update opportunities fixed; it cannot separately
identify their individual effects. Demand from steady traffic differs from
frequent startup, which is precisely the tested distribution change. If this
helps both methods at the joint outcome, the paper gains a supported training
design explanation. If it only helps vehicle loss or value prediction, report
that narrower effect. A new decision-boundary learner is a separate experiment:
it must accumulate intermediate rewards and use elapsed-time discounts, not
simply delete forced-action rows.

This is a proposed follow-up, not implemented or running as of this document.
Its cost is 12 models and 1,536 training chunks plus evaluation. Do not launch
another campaign merely because a new algorithm name is available.

## Optional extension: directly test coordination

If the report needs a causal claim specifically about MAPPO versus IPPO, use
a separate route-share by reward experiment. Keep three junctions, spacing,
vehicle origin arrival counts, pedestrian arrivals, speeds and safety fixed;
change the share routed through adjacent signals and compare local versus
explicitly shared/hybrid reward under both methods. All four cells and both
methods must be retained. Increased route share also increases downstream
traffic exposure even with fixed origin counts: measure link-level flows and
queues and describe that consequence instead of claiming identical loads.

Do not use the current short-spacing case alone as this test: it changes both
spacing and demand. A clean geometry sensitivity keeps demand fixed while
varying spacing. Weak coordination remains a hypothesis if this extension is
not run. This large optional experiment is lower priority than diagnosing the
observed failures and completing the current controlled test.

## Statistics, stopping and a defensible paper

Use paired demand seeds, all training replicas, and uncertainty over whole
replicas/demand runs. Never treat thousands of crossing events from one model
as thousands of independent trained controllers. Report each crossing's sample
count and zero-entry runs, not just pooled p95s. Preserve pending-at-cutoff and
unresolved-after-followup separately. Do not interpret a one-second sampling
system's small point differences as precise physical differences; report the
measurement audit and uncertainty rather than retroactively adding a tolerance.

Report vehicle time loss with completions, unfinished and uninserted vehicles.
New runs also report insertion delay. SUMO defines `departDelay` separately
from driving `timeLoss`; neither completed-trip average accounts for all
unserved demand. [Official definitions](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html).

The primary question remains unchanged. Maintain the zero-margin every-crossing
criterion. Use the maximum of the 12 crossing deltas as a network fairness
summary and simultaneous paired uncertainty for a stronger claim. Show the
component service failures; do not equate a cutoff-only failure with permanent
starvation. Any new non-inferiority margin belongs to a separately justified,
prospectively frozen analysis, not a rescue of previous failures.

Stop development after the declared diagnostic stages and budget; do not run
until a successful seed appears. For a final method-level claim, use the planned
five independent training seeds per final configuration and the reserved test
once decisions are frozen. If time prevents that, disclose the smaller study
and limit claims. Repeatedly inspected development results cannot become a
fresh final test by renaming them.

The paper can be organised around:

1. A joint target that is stricter than average pedestrian performance.
2. A controlled synthetic corridor, justified stress scenarios and common safety.
3. The original IPPO/MAPPO comparison, including failures and uncertainty.
4. The measured trade-off and targeted reward/guidance evidence.
5. Replayed failure mechanisms and controlled implementation ablations.
6. What improves, what remains unresolved, and limits of transfer to real roads.

Use “these tested controllers did not meet the joint target under the studied
conditions and budgets”, not “MARL cannot work”. One synthetic three-junction
corridor is one geometry family; four demand/stress settings are not four
independent real sites. Uncalibrated arrivals, modelled pedestrian interaction,
restricted phase choices, finite horizons and few training seeds limit external
conclusions. The [original MAPPO paper](https://arxiv.org/abs/2103.01955) studies
cooperative benchmark environments and implementation factors; it supplies no
guarantee of domination over this pedestrian-responsive traffic baseline.

Enough evidence means answering a few explicit mechanisms with controlled
tests, not collecting a large number of loosely related unsuccessful trials.
Where evidence is absent, leave a hypothesis open rather than filling the paper
with a confident but untested explanation.
