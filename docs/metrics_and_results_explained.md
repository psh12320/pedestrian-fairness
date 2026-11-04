# Understanding our traffic-signal experiments

Checked against saved source, protocols and results on 4 November 2026. This is an explanation of existing evidence, not a new training result. Completed studies, the friend's implementation and the current unfinished study must remain separate.

## The question in ordinary language

Can our learned traffic lights reduce vehicle delay while keeping long pedestrian waits at each crossing no worse than a pedestrian-responsive reference controller, without leaving more trips unserved?

The original target is all 12 crossings. Improving 6/12 is useful partial evidence, not success on that original target. We can also investigate why a trade-off occurs. A negative result does not establish that MARL can never work.

There are three junctions, with four crossings at each junction. C0, C1 and C2 are junction names; c0 through c3 identify their crossings. Twelve crossings are not twelve junctions. Different demand settings on one network are not different real-world sites.

## A metric is just a measurement

For example, a stopwatch measures a person's wait. A counter measures how many vehicles finish. We need both: a light that serves cars continuously could give very low car delay by never serving pedestrians. One good number cannot establish that the whole system works well.

Example, using invented timestamps solely to explain the definition:

| Event | Simulation time |
|---|---:|
| Person reaches the crossing entrance or its connected queue | 100 s |
| WALK begins | 112 s |
| Person actually enters the crossing | 115 s |
| Person leaves the crossing | 124 s |

The crossing wait is 115 - 100 = 15 seconds. It ends on actual entry, not merely when WALK appears. Crossing traversal takes another 9 seconds. A vehicle blocking entry can therefore cause waiting even while WALK is displayed.

Our current operational entrance definition uses a 1.2 m entrance zone and connected stopped queues, sampled every 0.2 s. It is a documented simulation measurement, not exact knowledge of a real person's intention. Sampling bounds do not eliminate uncertainty in the geometric definition.

The friend's tracker instead accumulates seconds below 0.1 m/s on an approach walking area. A synthetic check with 10 seconds between approach arrival and entry, including 5 seconds moving slowly, recorded only 5 seconds. That metric is not interchangeable with our arrival-to-entry target. Our earliest exploratory report also used a superseded stopped-approach definition; do not combine it with later measurements.

## Vehicle measurements

| Name | Meaning and preferred direction | Example or availability |
|---|---|---|
| Scheduled vehicles | Trips that demand generation intends to start in the measured cohort. Same schedule for matched controllers. | Guided-study medium demand: mean 960 per run. |
| Inserted/departed vehicles | Scheduled vehicles that successfully entered the network by the cutoff. | Reference: 959.67 = 919.33 completed + 40.33 unfinished. |
| Completed vehicles | Inserted vehicles that reached their destinations by the cutoff. More is usually better at matched demand. | Reference 919.33; MAPPO control 924.50. |
| Throughput | Completed trips per observation interval; can be expressed as a count or rate. Compare the same cohort/window. | Reference completes 95.76% of its 960 scheduled cohort by the cutoff; this is a completion fraction, not a measured lane saturation flow. |
| Unfinished vehicles | Inserted vehicles still travelling at the cutoff. Lower is better, but trips scheduled near the cutoff can naturally remain. | Reference 40.33; MAPPO control 35.17. |
| Uninserted vehicles | Scheduled vehicles still outside the network, usually because entry is blocked. Lower is better. | Medium reference/control: 0.33 each. Short-spacing reference: 1,538.00. |
| Completed mean time loss | Average seconds lost travelling below each completed vehicle's desired speed. Lower is better for that served subset. | Medium reference 50.55 s; MAPPO control 40.61 s. It is not the entire journey duration. |
| Completed total time loss | Sum of those losses, not their average. Depends on how many trips complete. | Recorded in run-level vehicle summaries; do not multiply rounded means from separate runs to reconstruct an exact total. |
| Departure/insertion delay (`departDelay`) | Actual network-entry time minus scheduled departure time. Lower is better. | Added to current run summaries; unavailable retrospectively in the archived guided summary. Do not assume zero. |
| Time loss plus departure delay | Combines driving delay and insertion delay for completed trips. Still excludes uncompleted trips' full eventual outcomes. | This is what the friend's `veh_mean_time_loss` actually reports. Our older 50.55 s reference does not include departure delay. |
| Travel duration | Time from actual departure to arrival, including ordinary movement. Different from time loss. | Not a primary aggregate in the reviewed guided result. |
| Vehicle stopped waiting time | Time stopped or nearly stopped. Does not include every slowdown. | Different from time loss; no separate primary aggregate in the reviewed guided result. |
| Teleport events | Simulator recovery events moving blocked vehicles out of their actual positions. Not successful physical traffic service. | Friend: 67 reference events and 289 IPPO-vehicle events over 180 runs per controller. |
| Collisions | Physical conflict diagnostics, distinct from whether the commanded green combination is legal. | Current runner enables junction collision checks and stops on detected collisions. Old archives without the new checks are not retroactive safety certificates. |

At a consistent cutoff: scheduled = completed + unfinished + uninserted. Counts with decimals are averages over whole-number runs, not fractions of a real car.

The recent studies measure vehicles scheduled in [300, 2100) seconds at 2100 seconds. They continue the simulation to 2400 seconds to follow pedestrian requests. Vehicles unfinished at 2100 should not be described as necessarily unfinished at 2400.

The short-spacing guided-study reference completes about 781/2,392 vehicles with 43.13 s completed-vehicle loss; MAPPO control completes about 1,701 with 63.99 s loss. Ranking only 43.13 against 63.99 hides the extra traffic served by MAPPO. Neither number by itself establishes a joint solution.

## Pedestrian measurements

| Name | Meaning and preferred direction | Actual example or interpretation |
|---|---|---|
| Crossing request / arrival | A person reaches the operational crossing entrance or its connected queue. Starts the wait clock. | One person may make more than one crossing request during their trip. |
| Crossing entry | Person starts using the crossing edge. Ends the wait clock. | A green light alone is not an entry event. |
| Crossing exit | Person leaves that crossing edge. | Needed to detect a person stranded on the crossing after a short initial wait. |
| Mean wait | Sum of observed completed crossing waits divided by their number. Lower is better. | Medium C0_c0: reference 10.72 s; MAPPO control 16.65 s; guided MAPPO 11.31 s. |
| p95 wait | Sort waits shortest to longest; approximately the 95th out of 100. Describes the long-wait end, not the average or maximum. Lower is better. | Same crossing: reference 25.60 s; control 43.47 s; guided 27.13 s. |
| Maximum wait | Longest observed wait. Can reveal extreme events hidden by p95. Sensitive to sample size. | Friend recorded waits of 289, 290 and 301 s against individual budgets of 47 or 49 s, despite pooled p95 passes. |
| Per-crossing statistics | Separate mean/p95 for each of 12 crossings. | A network-wide average could hide one neglected crossing. |
| Crossing p95 difference | Controller p95 minus reference p95 at the same crossing. Negative is better; zero is a tie. | C0_c0 control: +17.87 s; guided: +1.53 s. |
| Worst crossing difference | Largest of those 12 differences. Must be no more than zero for the every-crossing point-estimate target. | Medium control +17.87 s; guided +1.53 s. This is not the worst individual's wait. |
| Crossings no worse | Number of crossing p95 differences at or below zero. | Medium guided: 4/12 after treating roundoff as ties; original report says 3/12. Explanation below. |
| Entry count / sample size | Number of observed waits supporting a crossing estimate. | Medium reference C0_c0: 305 across six seeds. Each learned replica also has 305 here. |
| Pending at cutoff | Requests in the measurement cohort that have not entered by 2100 s. | Medium reference: mean 3.33 requests per run; control 6.11; guided 3.44. |
| Unresolved after followup | Those requests still without entry at 2400 s. Lower is better. | Zero for every medium-demand controller in the guided study. Not zero for some congested runs. |
| Pending age / censored wait | Time already waited when observation ends; eventual wait is unknown. | Someone waiting 40 s at the end has waited at least 40 s, not exactly 40 s. Do not record this as a completed 40-second wait or as zero. |
| Traversal time / entered without exit | Time spent physically on the crossing; if there is no exit, that crossing movement is incomplete. | Replayed person p000076 waited 1.4 s before entry but never exited before the 2400 s horizon. A short waiting metric alone missed that failure. |
| Person-trip completed/unfinished/uninserted | Status of the whole walking trip, distinct from one crossing request. | Medium guided learned models complete all scheduled person trips: mean 703.17 per run. |
| Slow-walker statistics | Repeat the crossing calculations for the slower group. | A mixed population's good overall mean can conceal poor service to slower walkers. Report the group's sample counts. |
| Unknown arrival / lost request | Tracking failures: entry without an observed arrival, or a tracked person disappearing/diverting without a resolved event. | Current analysis retains these diagnostics and rejects them in its service/measurement check. Missing is not zero waiting. |
| Wait lower/upper bounds | Bounds from sampled event times, separate from the point estimate. | Current sampling interval is 0.2 s; geometric ambiguity can exceed a single sampling interval. |

Current fairness summaries use a nearest-rank p95. The friend uses an interpolated percentile. With many observations they may be similar; with few observations the convention can matter. Pooled p95, average of per-run p95, and maximum of per-run p95 are different quantities.

In the guided study, waits are pooled across six demand seeds separately for each trained replica; the reported p95 then averages the three replica p95s. We do not treat thousands of pedestrians as thousands of independent training runs.

## Phase timing, safety and teleporting

The pedestrian admission period is when people may begin crossing. Clearance follows it: new entries stop, while people already crossing retain protection from conflicting vehicles. The agent chooses when to request pedestrian service; it cannot shorten these safety stages.

Current defaults:

| Parameter | Value / rule |
|---|---|
| Minimum vehicle green | 12 s |
| Fixed controller's vehicle green | 25 s per vehicle axis |
| Responsive maximum vehicle green | 30 s when competing demand requires a change |
| Yellow | 3 s |
| All-red transition | 2 s |
| Pedestrian admission | 12 s |
| Planned slow walking speed | 0.8 m/s |
| Minimum pedestrian clearance | Ceiling of longest crossing length / 0.8 + 3 s buffer |
| Occupancy protection | Clearance cannot finish until crossings are empty for 3 s and the minimum clearance has elapsed |
| Simulation step | 0.2 s |
| Decision interval | 1 s; many such steps do not allow a free decision |

For illustration only, an 8 m crossing with 0.8 m/s planning speed needs 10 seconds of traversal plus the buffer. Use the actual network lengths in experiments. Clearance is not a reward or a performance score.

The friend's code uses timer-based pedestrian clearance without our continuing pedestrian-occupancy check. Their experiment also permits gridlock recovery teleports after 300 s. Our current runner uses `--time-to-teleport -1` to disable that gridlock recovery and disables pedestrian obstacle-ignoring jam recovery during the run. Collision-related simulator behavior is separate: detected collisions stop the current experiment.

Removing recovery teleports can expose longer queues, more incomplete trips and worse physical congestion. A fixed 2400-second run still stops at 2400 simulated seconds; it does not wait indefinitely for every trip. Computer running time may change in either direction and needs measurement. A drain-until-empty run can instead take much longer or fail to empty under gridlock.

The friend's results need a matched no-recovery reevaluation of every controller before making claims about that environment. Training may then need repeating under the changed environment. Zero teleports alone does not establish realistic demand or validated safety.

## What our reference controller actually does

Our pedestrian-actuated reference is a deterministic rule-based program written for this simulation. It is not a learned average, not an average of controllers, and not a claim to reproduce deployed Singapore signals.

In the current environment it:

1. Requests pedestrian service when a person is waiting and the common phase rules permit it.
2. Otherwise keeps the current vehicle direction while it has a queue, or selects the other vehicle direction.
3. After pedestrian clearance, selects the larger vehicle queue, resolving ties consistently.
4. Always obeys the same minimum green, yellow, all-red and pedestrian-clearance rules as every other controller.

Fixed time instead cycles vehicle directions and pedestrian service using its schedule. The pressure-inspired comparator scores each vehicle direction by upstream queue minus downstream queue, and pedestrian service by twice the pending pedestrian count. It is a deliberately labelled approximation, not a verified reproduction of a published max-pressure algorithm. Uniform control randomly selects an available legal action. Imitation-only control learns to copy the actuated teacher.

We run the reference separately on each matched demand schedule and average its measurements. Thus 50.78 s in the PPO-repair study and 50.55 s in the guided study are different seed averages, not different definitions of an ideal score. Neither is a universal target to reuse across settings.

The friend's actuated controller follows different rules: it finishes 30 s EW and 20 s NS greens before a requested pedestrian phase. Comparing against a different reference changes the difficulty of beating it.

## Training budgets and what 'more data' would mean

An episode is one complete simulated traffic run used to collect experience. A training replica is an independently initialized model. A PPO update is a learning step; it is not necessarily one episode. Evaluation episodes test a frozen model and do not train it. Most of our experience is generated by our own synthetic SUMO demand, not measured Singapore traffic.

| Study | Episodes per trained model | Duration per episode | Independent training replicas per configuration | Status |
|---|---:|---:|---:|---|
| First exploratory linear-policy study | 36 | See archived study configuration | 1 | Completed; superseded measurement |
| Revised linear-policy study | 150 | See archived study configuration | 5 | Completed; separate environment/formulation |
| Large fairness-v2 cluster study | 512 | 1200 s | 5 | Completed, six learned configurations / 30 models |
| Service-priority pilot | 32 | 600 s | 3 | Completed |
| PPO repair pilot | 64 | 600 s | 3 | Completed |
| Guided exploration | 128 | 600 s | 3 | Completed, four MAPPO arms / 12 RL models |
| Observation/clipping audit | 64 | 600 s | 2 | Completed: 1,396 simulations; [review](../results/audit-repair-review/REVIEW.md) |
| Pedestrian fine-tuning | Additional 64 after a 128-episode parent | 600 s | 3 | Cluster complete: 876 simulations; local duplicate stopped; [review](../results/pedestrian-finetune-cluster-review/REVIEW.md) |
| Demand / early service | No training | 2400 s evaluation | 3 saved replicas | Completed: 720 evaluations; [review](../results/demand-service-review/REVIEW.md) |
| Friend's study | 450 = 150 iterations x 3 environments | 1800 s | 1 | Completed, four learned models |

One fairness-v2 model saw 614,400 simulated seconds (170.67 simulated hours). One friend's model saw 810,000 simulated seconds (225 hours). One guided model saw 76,800 simulated seconds (21.33 hours). These are simulated traffic durations, not computer hours, unique real observations or equal counts of informative decisions. Different actor architectures, horizons and update budgets prevent a causal comparison based on these totals alone.

The guided actors had only 3,969-4,851 free decisions per actor over 128 episodes, despite 76,672 recorded training states. Safety transitions and minimum greens force many steps. Independent replicas increase evidence about repeatability; their experience does not combine into one more-trained model.

More varied, correctly measured experience can help. In our frozen-policy critic diagnostic, increasing 4 to 32 independent trajectories reduced prediction MSE by 41.6% for local inputs and 55.9% for global inputs under one behavior policy. That experiment did not update the signal-controlling actor and therefore did not demonstrate traffic improvement. More repetitions of an incorrect reward, impoverished observations or unrepresentative startup conditions can instead reinforce the same behavior.

The useful quality improvements are accurate event measurements, representative congestion and slow walking, enough free decisions, correct learning updates, and experts generated in the same environment. Keep matched budgets, separate validation/test demand, multiple training replicas and the failures when testing them.

## Learning diagnostics: not seconds saved

| Term | Meaning | Value or example |
|---|---|---|
| Actor / policy | Maps the local observed junction state to legal action probabilities. | Chooses EW, NS or pedestrian service when allowed in the current categorical environment. |
| Critic / value estimate | Predicts future accumulated reward, to help training judge actions. | IPPO uses local inputs; MAPPO can use network-wide inputs. Does not see the future. |
| Reward | The numeric incentive during training, not the final evaluation outcome. | Queue and pedestrian penalties can encourage service but do not automatically enforce p95 at every crossing. |
| Reward weight | Relative strength of an incentive, not seconds or a percentage improvement. | Early studies tried pedestrian weights 0, 3, 10. Guided control already has weight 3; it is not vehicle-only. |
| Return | Accumulated, usually discounted future reward. | Negative returns reflect delay penalties; units depend on reward scaling. |
| Advantage | Estimated benefit of an action over expected reward in that state. | A training signal, not a reported vehicle-delay difference. |
| Critic MSE | Mean squared error between value predictions and reward-return targets. Lower is better on matched held-out targets. | 41.6% MSE reduction above is not 41.6% less traffic delay. |
| Critic RMSE | Square root of MSE, in reward units. | Repair pilot IPPO: 113.53 to 41.65; MAPPO: 112.19 to 42.77. |
| Critic bias | Average prediction minus target. Zero is desirable. | Repair IPPO +104.96 to +6.77; MAPPO +103.54 to +5.21. |
| Explained variance | How much variation in returns the critic explains; 1 is ideal, near 0 explains little, negative is possible. | Repaired IPPO 0.067 and MAPPO 0.032: calibration improved, state-dependent prediction remained weak. |
| Error relative to a constant predictor | Critic error divided by error from always predicting a fixed mean. Below 1 is better. | More-data diagnostic: local 0.614, global 0.631 for the plain behavior, 32 trajectories / 512 fitting updates. |
| Policy entropy | How spread out action probabilities are. Higher means more random choices, not better traffic. | Medium guided study control 0.626; guided 0.040; imitation 0.033. Values depend on legal action masks. |
| Teacher agreement | Fraction of sampled free decisions matching the teacher's choice. | Guided 94.93%; imitation 96.84% at medium demand. Similarity is not superiority. |
| Teacher-action probability | Probability the actor gives the teacher's choice; differs from sampled agreement. | Guided 94.87%; imitation 96.91%. |
| KL divergence | How far the new action-probability distribution moved from a reference policy. | Queued fine-tuning uses parent-KL weight 0.02; that is a coefficient, not a measured KL result. |
| PPO clipping / clip fraction | Limits the learning influence of large probability changes; fraction measures how often this limit is active. | A diagnostic, not a crossing-safety rule. No single overall archive value is reported here. |
| Gradient norm / gradient clipping | Size and limiting of a learning update. | Completed audit tests separate actor/critic clipping; no reliable joint improvement was established. |
| Learning curve / validation | Performance as training progresses on separate development seeds. | Guided control vehicle loss at 32/64/128 episodes: 56.23/60.19/41.46 s on the same validation seeds. Not a monotonic guarantee. |
| RND bonus | Rewards novelty in observed states to encourage exploration. | Medium RND vehicle loss 43.30 s, still worse pedestrian p95 at all 12 crossings than actuated. |
| Shield restriction rate | Fraction of decisions at which a service constraint restricts some actions. | Friend approximately 13.65-14.32% across learned models; not the fraction of actual actor choices overridden. |
| Phase-time share | Fraction of simulation time spent in each signal stage. Helps explain lost capacity. | Older audited medium traces: actuated vehicle-green share 35.27% versus fixed 60.42%; descriptive phase allocation, not a causal isolated effect. |

MSE/entropy/bias measured on different policies' own trajectories need not be directly comparable: the targets can differ in difficulty. Use matched-target tests when attributing a difference to a critic design.

## Statistics and pass criteria

| Term | Plain meaning | How to read it |
|---|---|---|
| Demand seed | Reproducible choice of arrivals/routes/speeds. | Matched runs use the same demand; randomness is not secretly fresh for each controller. |
| Training seed / replica | Independent network initialization and training randomness. | Three guided replicas, five large fairness-v2 replicas, one friend replica per configuration. |
| Paired difference | Model outcome minus reference on matched demand. | Negative vehicle or p95 difference is better. |
| Percentage change | 100 x (model - reference) / reference. | 40.61 vs 50.55 s is about -19.7%. It says nothing by itself about pedestrians. |
| 95% bootstrap interval | Uncertainty estimated by resampling whole demand seeds and, where available, trained replicas. | Guided vehicle difference -0.99 s, interval [-2.19, +0.29]: current evidence includes both improvement and worsening. |
| Sparse estimate | Too few events to resolve a tail reliably. | Development crossing-count threshold often 100; a screening rule, not a mathematical guarantee. |
| Descriptive pass | Point estimates satisfy vehicle, every-crossing and service requirements together. | No guided arm achieves the joint pass in the four evaluated settings. |
| Statistical support | Paired uncertainty also supports the declared joint claim. | Stronger than a point pass; testing 12 crossings requires simultaneous consideration. |
| Service failure | Some required vehicle/person/pending/measurement comparison worsens. | Can be one extra request at the cutoff that is served during followup; inspect the component before calling it starvation. |
| Budget exceedance | Individual observed wait exceeds the shield's chosen threshold. | Rare events can exceed a budget even when p95 passes. A p95 is not a maximum-wait guarantee. |
| Pooled vs per-seed pass | One aggregates seeds before computing p95; the other compares each seed separately. | Friend: 216/216 pooled cells per learned controller, but 2,154/2,160 per-seed cells, not exact perfection. |

The friend's 216 cells = 18 demand/population settings x 12 crossings. Their 2,160 per-seed cells multiply by 10 seeds. These counts are not independent networks or trained models.

## The IPPO/MAPPO question

There is no established universal winner in our evidence. Any earlier statement that MAPPO was simply better was too broad.

Large fairness-v2 medium-demand results (five training replicas; completed-vehicle time loss):

| Controller | Vehicle loss s | Whole person trips unfinished at simulation end |
|---|---:|---:|
| Actuated | 51.68 | 0 |
| IPPO vehicle reward | 14.92 | 698.6 |
| MAPPO vehicle reward | 15.19 | 698.6 |
| IPPO pedestrian reward | 53.21 | 0 |
| MAPPO pedestrian reward | 53.03 | 0 |
| IPPO constrained | 53.63 | 0 |
| MAPPO constrained | 53.33 | 0 |

The tiny MAPPO advantage for pedestrian reward is 0.18 s in this condition. Vehicle-only IPPO is slightly lower than MAPPO, but both leave extensive walking demand unfinished. Those are not successful joint policies.

In the later matched 64-episode repair study, normalized-clock IPPO gives 52.19 s and MAPPO 55.14 s against a 50.78 s reference. Neither meets the joint target. The 128-episode guided study contains MAPPO arms only, so its 40.61 s result cannot establish superiority to a comparably trained IPPO.

The friend finds lower vehicle delay for IPPO than corresponding MAPPO in all 18 settings of their implementation, with one training seed. That is useful evidence, subject to their different wait metric and recovery behavior. Their IPPO uses local rewards while MAPPO uses a team-mean reward; the comparison changes both critic information and reward allocation. Geometry, traffic load, phase timing, teacher strength and sampled versus deterministic evaluation actions also differ from ours.

A larger-information critic can help when neighboring junctions' conditions explain consequences, but also needs enough experience to learn those relationships. These results motivate hypotheses; they do not identify one proven cause or establish that either algorithm must dominate.

## Numerical tie correction found during this explanation

The archived guided report counts a positive difference of about 3.9e-14 seconds as a worsening. This is floating-point noise, far below the 0.2-second sampling resolution. Using a 1e-9-second arithmetic tolerance (not a substantive acceptable-worsening margin) gives:

| Setting / controller | Originally reported crossings no worse | Arithmetic ties corrected |
|---|---:|---:|
| Medium / guided MAPPO | 3/12 | 4/12 |
| Burst / guided MAPPO | 4/12 | 5/12 |
| Slow walkers / guided MAPPO | 7/12 | 9/12 |
| Slow walkers / imitation | 8/12 | 9/12 |

No every-crossing pass or study conclusion changes. The original archive and active experiment code remain unchanged. The companion tables display both counts to make the correction traceable.

## Sources and full tables

- [All four guided-study settings, vehicle/service counts and all crossing statistics](../results/metrics-guide/guided_all_settings.md).
- [Original large cluster protocol](../runs/cluster-final-review/protocol.json) and [results](../runs/cluster-final-review/REPORT.md).
- [Original large-study absolute values and service-priority pilot](../results/service-priority-cluster/REVIEW.md).
- [PPO repair results and learning diagnostics](../results/ppo-repair-cluster/REVIEW.md).
- [Guided study review](../results/guided-exploration-cluster-review/REVIEW.md) and [saved summary](../results/guided-exploration-cluster-review/archived_summary.json).
- [Critic data-volume experiment](../results/critic-data-volume-cluster/REVIEW.md).
- [Pedestrian admission 12/18/24-second experiment](../results/local-timing-study/REPORT.md).
- [Failure replays](../results/failure-mechanism-review/REVIEW.md).
- [Friend's independent review](../results/friend-marl-tsc-review/REVIEW.md).
- [Original revised linear-policy study](../results/final-study/REPORT.md); this precedes the neural fairness-v2 cluster study despite the historical folder name.
- [Current signal rules](../src/traffic_control/fairness/signals.py), [event tracker](../src/traffic_control/fairness/events.py), [vehicle accounting](../src/traffic_control/metrics.py), [configuration](../config/fairness_v2.json).
- [Official SUMO TripInfo definitions](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html), [teleport recovery](https://sumo.dlr.de/docs/Simulation/Why_Vehicles_are_teleporting.html), [pedestrian simulation](https://sumo.dlr.de/docs/Simulation/Pedestrians.html).

Do not infer missing archived metrics as zero, pool incomparable studies, choose favorable demand settings after inspecting results as if they were the only planned tests, or describe synthetic outcomes as deployed improvements.
