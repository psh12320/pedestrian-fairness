# Independent RL implementation audit — 3 November 2026

For the integrated repair status and final regression checks, see the
[consolidated audit](deep_audit_2026_11_03.md).

This audit separates reproducible implementation deficiencies from plausible explanations and legitimate modelling choices. It does not assume that PPO must outperform a conventional controller. None of the numerical software probes below is a traffic performance result. Historical source archives and checkpoints were not edited.

Reproduce the lightweight software probes with `python scripts/deep_audit_rl.py --clip-ablation`. Evidence is in `runs/deep-audit-rl.json`. The script performs no SUMO simulation, downloads nothing, and trains only tiny artificial positive controls.

Verification: all six 64-update positive controls completed; the targeted `tests.test_ppo_repair.RepairTests` and `tests.test_guided_exploration.GuidedTests` suites passed all 18 tests. The diagnostic script also compiled successfully.

## 1. Confirmed omission: known signal memory was hidden from both critics

The legacy observations in `fairness/environment.py` contain current phase, phase age and legal actions. They do not expose `SafeSignal.target`. After leaving EW, exactly the same EW_YELLOW or ALL_RED observation can precede NS or PED. `SafeSignal.advance()` follows the stored target. Thus the observation can identify neither the committed next phase nor its consequences. MAPPO concatenates the same incomplete local observations, so centralising them does not recover the missing fact.

The script constructs two signals, requests NS in one and PED in the other at second 12, verifies identical legacy signal observations at seconds 13 and 16, then verifies their next phases differ at second 17. Traffic features can be held equal in this counterexample. If two aliased input states had targets -10 and -30 with equal frequency, any deterministic predictor of those inputs would have minimum MSE 100. This numerical pair illustrates the mechanism; it is not a measured share of our actual critic error.

A second omission is `empty_since` during pedestrian clearance. Identical phase, phase age and currently empty crossings can lead to READY immediately or a continued clearance wait depending on how long crossings have been empty. The script reproduces this difference too.

This affects legacy neural fairness, service-priority, PPO-repair and guided/RND studies. It does not prove how much of their performance it explains. A partly observed environment remains a valid RL problem, but withholding the controller's own known state is an avoidable source of value-estimation error. Actors cannot act during yellow or clearance; the immediate concern is the critic used to propagate credit through those transitions.

The parent audit implemented an opt-in `controller-memory-v1` observation schema with committed-phase one-hot fields and a known/elapsed clearance-empty timer. I reviewed the helper: it adds local controller memory, without future demand or other-junction information. Legacy remains the default. Old 42-input checkpoints cannot simply consume new 47-input observations; a separately versioned training run is required. This removes these aliases, not all partial observability of aggregate traffic.

## 2. Confirmed training coupling: critic gradients also clip actor gradients

`fairness/learning.py:100`, `ppo_repair/learning.py:170`, and `guided_exploration/learning.py:222` apply one norm clip to all actor and critic parameters. Actor and critic networks are separate, but a large critic gradient therefore changes the actor's clipping factor. The official MAPPO trainer clips actor and critic separately, with separate updates. Our code is a custom PPO implementation, not a reproduction of that trainer. [Official MAPPO trainer](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py).

The artificial probes instrumented the exact production updates. In the repaired cases, mean critic gradient norms were 8.59 (IPPO) and 9.29 (MAPPO), against actor norms 0.192 and 0.184. The mean shared clipping factors applied to the actor were 0.112 and 0.121. These factors do **not** imply an equivalent reduction in Adam's effective learning rate: adaptive moment scaling matters.

An isolated separate-clipping ablation, implemented only by the audit script, produced the following optimal-action probabilities after 64 updates:

| Known-solvable software task | Existing shared clip | Separate actor/critic clips |
|---|---:|---:|
| Repaired IPPO | 85.18% | 88.02% |
| Repaired MAPPO | 82.95% | 84.91% |

This is one fixed synthetic seed, not a SUMO experiment or statistical evidence of a traffic improvement. Separate clipping is a well-motivated implementation improvement to test, not an explanation that has been causally established on our traffic results. It must be frozen as a named ablation, not silently substituted into an existing experiment.

## 3. Positive controls: the basic policy update does learn

The audit constructed three local contextual bandits with known best actions. Each actor receives its own context; MAPPO's value estimator receives all contexts. The reward resulting from an action appears in the next row, exactly as in the traffic collector. The experiment uses the existing PPO update, masks, raw-unit GAE, normalisation and optimiser, with 64 updates and 192 decisions per junction per update.

| Learner | Initial best-action probability | After 64 updates |
|---|---:|---:|
| Original IPPO | 33.33% | 93.20% |
| Repaired IPPO | 33.33% | 85.18% |
| Original MAPPO | 33.33% | 93.96% |
| Repaired MAPPO | 33.33% | 82.95% |

These tests contradict a blanket explanation such as “the gradient has the wrong sign” or “the actor never updates.” They do not validate long-delay traffic learning or prove every implementation choice is good. Local Torch emitted its already-known NumPy bridge warning, but all tensor/list-based probes completed successfully.

By source inspection and the existing unit tests, the current repair/guided implementations correctly align action t with reward t+1, preserve legal collection masks, compute old log probabilities before optimisation, stop GAE at each declared finite episode, use raw-unit values with the repaired normaliser, mask forced actor decisions, and leave evaluation parameters unchanged. Actor networks have only local inputs. Separate junction parameter sets make sequential per-junction optimiser updates valid; no shared critic is accidentally changed before another junction's old log probabilities are calculated. No reversed reward, future reward leak, action-index swap, or actor accidentally receiving global observations was found in this inspection.

## 4. A logged “600-state rollout” contains very little actor experience

The archived 18-model repair study contains 64 update batches per original/repaired model and only 16 per batch-four model. Across its final four updates, it averaged roughly 33–40 free decisions per junction per 600-second episode. Reported state counts are mostly forced minimum-green/transition states, which train the critic but not the actor.

Final mean free-choice entropies were approximately 0.77–0.99 nats; final mean update KLs were approximately 0.00005–0.001. A three-way uniform choice has entropy about 1.10; two-way masks have maximum entropy about 0.69, so these numbers must be interpreted together with masks. They indicate substantial stochasticity remained, rather than a decisive learned policy. These are actual archived traffic diagnostics, not the toy-task figures above.

An independent fresh 600-second actuated trace (seed 920102) had 31 free choices at each junction and 328–352 transition-only rows per junction. This is a baseline trace, not evidence of on-policy gradients. It shows why counting all simulation seconds as policy decisions overstates the effective actor sample budget. A “large number of episodes” across many different experimental variants is not a large training budget for each final model.

Measure free choices, policy entropy, action probabilities, KL and both gradient norms in the next pilot. Increase meaningful on-policy experience only under a fixed budget rule. Compare greedy and categorical evaluation on development seeds as an explicit inference-mode ablation; prior deterministic studies also failed, so greedy evaluation is not a promised solution.

## 5. Delayed credit is a specific hypothesis, not a proven arithmetic error

The current GAE coefficients are gamma=0.99 and lambda=0.95 **per one-second row**. A TD residual k seconds away is weighted by 0.9405^k: approximately 0.159 at 30 seconds, 0.086 at 40 seconds and 0.025 at 60 seconds. The fresh actuated trace had approximately 10–11 seconds of pedestrian clearance, making vehicle-to-PED-to-vehicle protection roughly 3 yellow + 2 all-red + 12 admission + 11 clearance = 28 seconds. It would be inaccurate to call clearance alone 30+ seconds here.

An accurate value estimator can propagate long-term effects despite these residual weights. Lambda is not a hard reward horizon, and increasing it can increase variance. However, omitting transition targets while heavily relying on short-range bootstrapping is a particularly unhelpful combination. The current future value enters through ambiguous observations during precisely the forced interval caused by an action. [GAE paper](https://arxiv.org/abs/1506.02438).

Test corrected observations first. Then compare one controlled longer-credit setting or a decision-boundary semi-Markov rollout that accumulates rewards with the appropriate elapsed-time discount. Do not merely delete forced rows: that would discard transition costs and apply the wrong time discount. A continuing-task variant also needs an explicit final-state bootstrap, unlike the currently declared finite-horizon objective.

## 6. Other confirmed mismatches requiring separate tests

- The reward includes uninserted external vehicle backlog, but neither actor nor critic observations include that backlog. Two states with identical visible in-network queues can have very different ongoing costs. This matters most when insertion is blocked. Add privileged backlog to critic inputs under a new schema, or provide a documented local virtual arrival detector if actors should also observe it. Removing the cost would conceal unserved demand and is not a valid fix.
- Repair/guided training resets the network empty every 600 seconds, while evaluation runs 2,400 seconds and measures after warmup. Zero terminal bootstrap is internally correct for the explicitly declared finite 600-second objective, but this is a train/evaluation objective mismatch, not a continuing traffic-control formulation. The existing clock repair gives the remaining time to critics only; it does not make training equivalent to a continuing task. Test long warm starts or correctly bootstrapped contiguous rollout windows.
- Both IPPO and MAPPO optimise local queue/backlog/pedestrian-count rewards. A central value estimator alone does not create a cooperative network objective or direct crossing-tail constraint. Queue sums reward total waiting person-seconds, not the maximum crossing's p95. The new age penalty is closer to long waits but still does not guarantee every crossing is no worse than its baseline. Test local versus shared/hybrid rewards under an explicit ablation if network cooperation is the hypothesis.
- The old linear Bernoulli controller, original neural fairness model, service-priority wrapper, repaired critic and guided/RND implementation are different generations with different action spaces, inference modes and protocols. A failure in one cannot be assigned to the others without checking its source archive. The original raw critic scaled its loss rather than its outputs; the repaired critic already addressed that conditioning issue. It should not be presented as a newly discovered unfixed error.

PPO/MAPPO papers report empirical success under particular tasks and implementation choices, not guaranteed dominance over fixed or responsive traffic controllers. The MAPPO reference itself emphasises implementation and hyperparameter sensitivity. [PPO paper](https://arxiv.org/abs/1707.06347), [MAPPO study](https://arxiv.org/abs/2103.01955).

## Recommended next discriminating experiment

Freeze a new development protocol with the same geometry, safety, demand seeds, budgets, and baseline runs. Compare legacy observations versus controller-memory observations with every other choice held fixed; log both gradient norms and free-action counts. Test independent clipping as a separate factor if resources permit. Require the software positive control and masked-action invariants first. Then compare held-out value prediction, learned action patterns and actual vehicle/crossing/service outcomes. Only scale training when these diagnostics show useful learning. A new algorithm name or more episodes alone would not distinguish these causes.

No new full SUMO training run was completed as part of this RL audit. The confirmed omissions/coupling can be corrected and tested; their effect on the proposal's joint target remains unmeasured.
