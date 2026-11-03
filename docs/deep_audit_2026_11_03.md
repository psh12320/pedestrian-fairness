# Deep implementation and experiment audit — 3 November 2026

Three independent agents audited the learner, simulator/measurement, and experiment/results pipeline. The primary agent checked their evidence, applied bounded repairs, and ran integration/regression checks. We found actionable defects and design weaknesses, but no evidence that every negative result is an arithmetic mistake or that a dominating policy is guaranteed to exist. Existing studies remain valid observations of their documented implementations, with the limitations below.

## Findings and their status

| Finding | Evidence and consequence | Status |
|---|---|---|
| Known signal memory absent from observations | Same yellow/all-red phase, age, mask and traffic can precede either NS or PED; same clearance observation can release immediately or wait, depending on the hidden empty timer. Both critics inherit this ambiguity. | Fixed in opt-in `controller-memory-v1`; old models require separate retraining. No measured traffic improvement yet. |
| Junction collision detection disabled | SUMO defaults do not perform junction vehicle-shape overlap checks without `--collision.check-junctions`. Four fresh 600-second checks found no collisions; this is a verification gap, not an observed cause of poor learning. | New environment source enables it for every controller and includes collision details in failures. |
| Windows/Linux demand hashes differed for identical traffic | CRLF versus LF explains the entire difference for all six reconstructed cluster seeds. Within-study demand was correctly matched. | Fixed binary XML writing; Windows seed830001 now exactly matches the archived Linux byte hash. |
| Vehicle insertion delay absent from delay reporting | SUMO `timeLoss` excludes waiting before entering the network. Counts alone cannot expose large insertion delays among eventually completed trips. | Added separate completed-trip `departDelay` and `timeLoss + departDelay`, with coverage and no missing-value imputation. All-scheduled accrued delay remains a future metric. |
| Critic gradients determine actor clipping | One gradient norm covers separate actor/critic networks. Official MAPPO clips them separately. A one-seed artificial ablation improved correct-action probability modestly with separate clips. | Candidate learner ablation, not silently changed and not a demonstrated SUMO fix. |
| Reward penalizes unseen external backlog | Uninserted-due vehicles affect reward but are absent from critic observations. Equal visible in-network queues can conceal different costs. | Unfixed input-design limitation. Test critic-only backlog features; actor access would need a declared virtual detector assumption. |
| Little effective actor training | Entire 64-episode pilot: 2,230–2,736 free decisions per actor. Four settings receive 16 episodes each; batch-four receives only 16 updates. Policies remain highly stochastic. | Budget/convergence concern. Report choices and updates; do not equate total experiment count with experience per model. |
| Startup training differs from evaluation | Empty-reset 600-second episodes versus 2400-second evaluation, excluding warmup; slow walkers held out. | Declared distribution shift, not a terminal-return arithmetic bug. Test continuing or longer training explicitly. |
| Weak network coupling and local objectives | Only 22.25% of six evaluated medium cohorts traverse all 3 junctions; MAPPO estimates local returns despite seeing global observations. | Design limitation. Central information alone creates neither downstream incentives nor a guaranteed advantage. |
| Service headline is stricter than aggregate service | Every extra cutoff request in any paired run fails the gate even if later served.10/15 repaired-IPPO failed pairs were cutoff-only. | Preserve old criterion; show components. Removing it still does not rescue worse vehicle loss and all 12 worse tails. |

## What was ruled out in the checked evidence

The current PPO reward shift, policy-gradient sign, legal masks, raw-unit GAE, forced-action masking, episode boundaries and checkpoint restoration passed inspection/targeted tests. All four original/repaired IPPO/MAPPO software positive controls learned a known task: optimal-choice probability rose from 33.33% to 82.95–93.96%. These controls contain 192 choices per actor/update, far more than one traffic episode; they establish that basic learning works, not that traffic learning converges. Two additional software controls tested separate clipping.

All 144 local timing records contain 135,627 crossing entries. Every recorded entry occurs during pedestrian admission, with zero unknown approaches, lost requests or completed phase-duration violations in this audit. This checks the saved traces; it is not a universal physical safety proof or a replacement for hand-observing every geometric arrival case. Four new collision-enabled 600-second simulations found no collisions. One-second internal-vehicle occupancy sampling also found no vehicle inside the junction during pedestrian green in those checks; it is not continuous collision detection.

The independent protocol audit reproduced all six medium cluster demand schedules and their normalized hashes, recalculated crossing tails and vehicle accounting, and found no reporting arithmetic error that changes failure into success. Identical fixed-time vehicle outcomes across medium/burst/slow cases are expected: vehicles and fixed phase histories match, while pedestrian records differ. The earlier 50.78s and newer 54.14s actuated references use different demand seeds and must not be treated as an algorithm change.

## The phase-time evidence explains a real trade-off

Medium demand, 12-second admission, averages across four new local seeds and three junctions:

| Controller | Vehicle green | Pedestrian admission | Pedestrian clearance |
|---|---:|---:|---:|
| Fixed |60.42%|14.50%|13.00%|
| Actuated |35.27%|26.17%|23.99%|
| Pressure-inspired |41.85%|22.96%|21.05%|

Actuated control gives pedestrians service as soon as allowed and spends about half of junction time in admission/clearance. Fixed time allocates much more time to vehicles. Beating fixed-time vehicle delay and actuated pedestrian tails simultaneously is therefore a demanding optimization question. This does not establish impossibility. It does explain why a general claim that either PPO method must dominate both is unsupported.

## Repairs made and verified

`fairness/control_observation.py` adds local committed-phase and clearance-empty timer features. Select them explicitly using `overrides={"observation_schema": "controller-memory-v1"}` in the environment. Default remains `legacy-v2` so old 42-input actors are not silently fed 47-input observations. Existing training entrypoints do not automatically opt in; a newly frozen training protocol must forward the option and record it. This is an environment correction ready for an ablation, not a newly trained controller.

The environment now enables junction collision checks and preserves collision participants/type in failure messages. The demand generator writes bytes identically across operating systems. Vehicle summaries now additionally include:

- `completed_depart_delay_observed_count`
- `completed_mean_depart_delay_seconds`
- `completed_total_depart_delay_seconds`
- `completed_mean_time_loss_plus_depart_delay_seconds`

Original time loss, vehicle counts, crossing measurements and strict success rules remain available. Old artifacts without departure-delay data are not retroactively imputed; their additional metrics are unknown. Existing report aggregators are not automatically a new insertion-delay analysis: these new fields are retained in run-level vehicle results for future reporting.

Observed verification:

- 46 distinct targeted regression tests passed across learning, masking, normalization/resume, comparison, common phases/events, controller memory, demand bytes and insertion-delay accounting. The known Torch/NumPy bridge warning remained; tensor-based tests passed.
- Two matched 180-second SUMO runs of the final source produced exactly equal actions, vehicle/person/crossing outcomes, phase histories, and demand/network hashes across legacy and augmented observations. 540 junction rows were compared; 98 exposed a committed target and 44 exposed the empty timer. All completed trips had observed insertion delay.
- All four primary software positive controls and two separate-clipping controls completed.
- The uploaded guided-exploration source archive is unchanged: SHA256 `20a8ea6cca9072cf9a5b15b845dd4d0d34456b29d3da87924f3d6df25cd9f979`.

Source fingerprints have changed locally. Use a new output/protocol for future runs; do not resume an old fingerprinted training directory using this checkout. Already-running cluster copies and saved archives were not modified. No new full SUMO training campaign was run, no old negative result was removed, and final test seeds 58001–58020 remain untouched.

## Next experiments, with a reason for each

1. **Observation ablation first:** same IPPO/MAPPO methods, demand, seed sets, safety, reward and fixed experience budget; compare legacy observations with controller memory. Compare held-out value fit, actual action choices, all vehicle/service metrics and all 12 crossing tails. This directly tests the confirmed missing-information mechanism. Add external backlog to critic information as a separate factor so its effect is identifiable.
2. **Separate actor/critic clipping:** hold observations and everything else fixed. Log both norms, KL and free-choice counts. The toy improvement motivates this test but does not establish traffic benefit.
3. **Delayed-credit and startup check:** measure policy behavior on matched 600- and 2400-second horizons, then test longer/continuing training with correct bootstrap handling. A decision-boundary formulation must accumulate forced-transition rewards with elapsed-time discounts; deleting forced rows is incorrect. Change one credit-assignment setting at a time.
4. **Verify demonstration fidelity from the current cluster study:** the imitation-only actor must reproduce actuated behavior closely in closed-loop simulation before treating it as a reliable starting point for PPO. Classification accuracy alone is insufficient. Check whether later PPO preserves or loses that service behavior.
5. **Cooperation and reward ablations only after learning is demonstrated:** keep the original corridor and add a declared high-corridor-share case while fixing volume/spacing; compare local and shared/hybrid rewards under both methods. Do not replace the original case with a favorable one. Queue rewards optimize total waiting proxies, not the every-crossing p95 constraint.

Use the same budget and declared extension rule for comparison arms, evaluate all replicas and held-out demand seeds, and retain failures. Scaling a bounded pilot should depend on useful learning evidence rather than a new algorithm name. None of these experiments guarantees the proposal target; each distinguishes a specific explanation.

## Evidence and reproduction

- [Independent learner audit](audit_rl_findings.md)
- [Independent simulation audit](audit_simulation_findings.md)
- [Independent protocol/results audit](audit_protocol_findings.md)

From the project folder:

```powershell
python scripts/deep_audit_protocol.py
python scripts/deep_audit_rl.py --clip-ablation
python scripts/deep_audit_simulation.py
python scripts/verify_control_memory.py
python -m unittest tests.test_control_state_observation tests.test_demand_bytes tests.test_vehicle_departure_delay -v
```

Protocol and saved-trace audits read downloaded/local artifacts. The last simulation verification needs working SUMO/TraCI; it creates small temporary runs. Live collision reproduction is available with `python scripts/deep_audit_simulation.py --live`. The independent reports document the pre-repair source they inspected; use this consolidated record for the repair status.

Primary references checked: [official MAPPO implementation](https://github.com/marlbenchmark/on-policy), [actor/critic updates](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py), [SUMO junction collisions](https://sumo.dlr.de/docs/Simulation/Safety.html), [TripInfo delay definitions](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html), [pedestrians](https://sumo.dlr.de/docs/Simulation/Pedestrians.html), and [crossing tutorial](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html).
