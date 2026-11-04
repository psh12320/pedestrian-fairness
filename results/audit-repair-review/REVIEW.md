# Completed laptop audit: what the learning corrections changed

Reviewed 4 November 2026. The run finished at 02:25 Singapore time with all
1,396 planned simulations: 1,024 training, 48 validation, 68 baseline and 256
learned evaluations. Sixteen models each completed 64 training episodes.

The independent standard-library review checked 47 frozen source hashes,
16 final-checkpoint hashes, all training schedules and matched training demand,
324 baseline/evaluation records and compressed crossing-event files, 3,888
crossing records, and 48 controller-by-setting aggregate rows. Vehicle and
person accounting, event-derived arrival-to-entry waits, pooled crossing means
and p95s, model identities and evaluation demand/network pairing passed.
There were no unknown/lost or unresolved measured-cohort crossing requests in
those 324 records. This is a saved-data audit, not a fresh physical SUMO replay.
The reported bootstrap intervals were inspected in the implementation but
were not independently regenerated.

Evidence: [audit output](audit.json), [complete original report](archived_REPORT.md),
[original summary](archived_summary.json.gz).

## Result

None of the eight learned configurations meets the joint target in any of the
four settings. At medium demand every learned configuration has higher vehicle
time loss and higher p95 at all 12 crossings than actuated control. The smallest
pooled crossing sample is 171 events, so this is not the earlier five-entry-tail
problem. Two training replicas and four demand seeds still give limited evidence
about training variation and rare failures.

All arms use pedestrian weight 3 and the same current collision checks,
clearance, signals and event measurement. This is a correction experiment,
not the original vehicle-reward versus pedestrian-reward comparison.

## Medium-demand comparison

Vehicle time loss is the mean of completed-vehicle run means, for trips scheduled
in [300,2100) seconds and completed by 2100 seconds. Lower is better only when
read with service counts. Pedestrian followup continues to 2400 seconds.

| Learning configuration | IPPO vehicle loss s | MAPPO vehicle loss s | Crossings no worse than actuated, IPPO / MAPPO |
|---|---:|---:|---:|
| Previous normalized-clock learning | 71.80 | 62.91 | 0/12 / 0/12 |
| Known signal-state information added | 71.31 | 59.70 | 0/12 / 0/12 |
| Actor/critic update sizes limited separately | 75.34 | 67.29 | 0/12 / 0/12 |
| Both changes | 70.08 | 65.05 | 0/12 / 0/12 |

Actuated: **51.42 s**. Fixed time: **33.37 s**, but 0/12 crossings no worse.
Pressure-inspired: **39.91 s**, with 1/12 crossings no worse and a largest
p95 increase of 10.60 s. Uniform legal random choices: **56.05 s**, also 0/12.
No baseline other than the reference itself provides every-crossing protection.
Random choices are not a joint winner either.

The lowest-delay learned arm, MAPPO with memory, is 8.28 s worse than actuated,
with an exploratory paired interval of [4.48, 13.78] s. Its worst crossing p95
increase is 24.10 s. It completes 919.00 versus 923.50 reference vehicles by
the cutoff, from the same mean 963.50 scheduled trips.

## What the corrections taught us

Adding signal-state information to MAPPO reduces medium-demand vehicle loss by
3.21 s against its matched previous-observation control, but the paired interval
[-17.73, 6.97] includes both improvement and worsening. Eight crossing p95 point
estimates improve or tie against that learned control, while four worsen; all
12 still lose to actuated. This does not establish memory as a successful fix.

Separate gradient limits have context-dependent effects. Under short spacing,
IPPO drops from 84.55 to 72.65 s versus its matched control: -11.90 s with an
exploratory interval [-19.22, -6.52]. But nine crossing p95s worsen against that
control, with a worst increase of 14.80 s. MAPPO's equivalent correction raises
short-spacing vehicle loss by 8.83 s, interval [3.29, 13.74]. These selected
contrasts are exploratory and not adjusted for examining many settings.

Together, the tests show that the confirmed information/update limitations are
not sufficient explanations for the whole performance gap at this budget.
They can change behaviour without satisfying the joint objective. The result
does not prove that the fixes have no possible value after longer training.

## Congestion requires reading throughput alongside delay

The short-spacing case also raises demand; it does not isolate spacing alone.
The measurement cohort averages 2,353.50 scheduled vehicles.

| Controller | Vehicle time loss s | Completed by cutoff | Unfinished | Uninserted |
|---|---:|---:|---:|---:|
| Actuated | 46.64 | 709.50 | 64.50 | 1,579.50 |
| MAPPO with signal memory | 77.57 | 1,429.38 | 98.62 | 825.50 |
| Pressure-inspired | 53.38 | 1,903.50 | 86.25 | 363.75 |
| Fixed time | 37.70 | 2,251.00 | 86.50 | 16.00 |

MAPPO with memory serves roughly twice as many vehicles as actuated here, but
both conventional alternatives serve more. The reference's low completed-trip
delay describes only its served minority. None meets the joint proposal target.
New insertion-delay fields reinforce the need for service accounting; they
still do not measure the eventual complete delays of trips never inserted.

## Scheduling and learning diagnostics

At medium demand, actuated allocates about 35.38% of junction time to vehicle
green and 49.99% to pedestrian admission plus clearance. IPPO's previous-learning
arm allocates 33.56% and 52.68%, respectively. MAPPO's previous-learning arm
allocates 37.03% and 47.89%, yet has worse vehicle and pedestrian outcomes.
Total green allocation alone therefore cannot explain performance; how service
is scheduled in response to demand also matters. This is descriptive trace
evidence, not an isolated causal intervention on phase allocation.

The largest recorded medium-demand interval with a queued vehicle direction
receiving no green is 141 s for actuated and 322 s for MAPPO with both changes.
These are maxima across the observed runs, not means or population bounds.
They motivate looking at localized service gaps instead of only mean delay.

Each trained junction actor receives roughly 2,206-2,442 free decisions over
64 episodes. The remaining recorded time steps do not all permit an action.
The single validation-demand seed shows changing, often nonmonotonic behaviour.
Average MAPPO memory vehicle loss at episodes 16/32/64 is 57.86/58.40/56.96 s;
its previous-observation control gives 59.13/60.37/65.06 s. These curves do not
establish convergence, and validation must not be used to erase later failures.

Training from empty 600-second episodes versus warm 2400-second evaluation,
reward alignment with individual crossing tails, missing external-backlog inputs,
and strength of junction coupling remain separate hypotheses. This audit does
not prove any one of them is the unique cause. MAPPO versus IPPO rankings here
are conditional on this short-budget formulation; they do not overturn the
different rankings in previous studies or the friend's distinct implementation.

## Next evidence

The cluster fine-tuning study starts from all three already vehicle-efficient
128-episode MAPPO parents. It compares ordinary continuation, a long-wait cost,
and that cost plus urgent actuated advice, on new matched development seeds.
That is a different hypothesis from retraining these 64-episode audit arms.
No cluster result has been reviewed yet. It must demonstrate its own vehicle,
crossing-tail and service outcomes; neither teacher guidance nor more episodes
is a promise of success.

The local fallback completed 42/876 simulations and then obeyed its STOP marker.
Its experiment state is `stopped`. The launcher labels the nonzero exit `failed`,
but the recorded cause is the requested stop, not an algorithm exception. Its
partial output must not be combined with the cluster's complete study as extra
independent replicas.
