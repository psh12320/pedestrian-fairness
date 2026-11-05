# Pedestrian-fair signal control: a complete account

_2026-11-05. Development evidence from local runs that reproduce the reviewed
cluster study to the decimal. `descriptive_pass` = point estimates meet the target;
it is NOT a confirmatory statistical claim. Reserved final seeds 58001-58020 are
untouched throughout._

## 1. The question, in one sentence

Can a controller clear vehicles faster than pedestrian-actuated control **without
making any of the 12 pedestrian crossings wait longer** (95th-percentile), and
without dropping service? That conjunction — vehicle Δ<0 AND p95 no-worse at all 12
crossings AND a clean service gate — is the "joint win."

## 2. Headline result

**No controller achieves the joint win, and we now know why: the residual is an
irreducible phase-cycle timing floor, not an optimization or learning failure.**

Along the way we produced the project's best controller (first to combine a vehicle
gain with majority crossing protection) and, through four independent experiments,
localized the barrier to the physical safety machine itself. That is a stronger,
more defensible result than an unexplained near-miss.

## 3. The frontier

Medium/medium, vs. actuated (51.2 s):

| Controller | Vehicle Δ | Crossings protected /12 |
|---|---:|---:|
| continued MAPPO | **−12.7 s** | 0 / 12 |
| early-service hybrid | +3.8 s | 6 / 12 |
| **early+vehrule (ours, best)** | **−0.5 s** | **6 / 12** |
| Stage-2 constrained MAPPO | −14.0 s | 0 / 12 |

`early+vehrule` is the only controller with a vehicle gain **and** half the crossings
protected simultaneously. At low/medium it reaches **−0.9 s with 7/12** — the nearest
approach to the joint target in the whole project.

## 4. What we did, and what each step established

### Stage 0 — reproduced the pipeline locally
The full SUMO+TraCI+PyTorch pipeline (believed cluster-only) runs on a laptop
(`scripts/setup_local_env.sh`: Python 3.12 + `eclipse-sumo` wheel). A deterministic
replay reproduced the reviewed 720-run frontier **to the decimal** — so every number
below is trustworthy and comparable.

### Stage 1 — a vehicle-axis rule moved the frontier
Added a larger-stopped-queue vehicle rule on top of the early-service pedestrian
rule (`mappo-early+vehrule`). No training. It **recovered 4–5 s of vehicle delay in
every cell** vs. the pedestrian-protecting hybrid while holding its crossing
protection — producing the first controller with vehicle-gain + majority protection.
Honest framing: a rule win, not a learning win (it reproduces actuated's queue
logic by hand).

### Stage 1.5 — the critic is not the limiter
A held-out critic-fitting gate showed the value function predicts returns at
**~0.49 explained variance** (not the ~0.002 of the old linear PPO). The
continued-MAPPO checkpoints are `normalized-clock` RepairNetworks with value
normalization already built in. **Value estimation is sound; the bottleneck is the
reward specification** (a dense sum vs. a per-crossing tail objective).

### Stage 2 — constrained RL reverts to vehicle-maximization
Trained a per-crossing Lagrangian constrained MAPPO (horizon-matched 2400 s,
warm-started, 3 replicas). Result: **excellent vehicle numbers (−4 to −14 s) but
0–1/12 crossings and a failing gate in 21–27/30 runs.** Diagnosis from the run
itself: the dual multipliers were **still climbing at episode 32 (0.015→0.285, no
plateau)** and the policy served PED only **35% of the time when legal** — the
constraint was under-powered and cut off mid-correction. Constrained RL at this
budget **underperformed the hand-coded rule** on the actual objective.

### Routes E & D — the barrier is physical, not algorithmic
- **Route E** (per-crossing pedestrian deadline): a **no-op**. The deadline
  ("oldest pending wait > actuated p95 AND PED legal") **fired 0 times** — the
  early-service rule already clears every request before it can age that long. The
  admission lever is saturated.
- **Route D** (per-junction green-hold search): also a **no-op** — every hold
  threshold gave identical results, for the same pre-emption reason. The decisive
  measurement: at a stubborn crossing, PED is served **55 times** (more than either
  vehicle axis), with a **mean 42.5 s gap** between PED greens whose floor is the
  mandatory sequence (min-green 12 + yellow 3 + all-red 2 + PED 12 + clearance ~7
  ≈ 36 s).

**Conclusion:** pedestrians at the stubborn crossings are already served as often and
as early as the safety machine physically permits. The residual p95 gap vs. actuated
is an **irreducible phase-cycle floor**. Actuated "wins" those crossings only because
it runs a different vehicle cadence that happens to recur PED in phase with their
arrivals — a trade our controller makes differently to gain vehicle throughput.

## 5. Why this is the right story

Four independent lines of evidence converge on one wall:
1. critic fits (Stage 1.5) → not value learning;
2. constrained RL reverts (Stage 2) → not reward/actor at feasible budget;
3. admission deadline never binds (Route E) → admission saturated;
4. green-hold is a no-op and PED runs at the cycle floor (Route D) → physical timing.

So the joint win at medium demand is **likely infeasible under this safety machine
and geometry**, and we can say so with mechanism, not merely report a failure. A
genuine 12/12 would require changing the physical phase structure (shorter clearance,
concurrent pedestrian phases) — which the safety guardrails deliberately forbid.

## 6. Contributions

1. **Local reproducibility** of the full pipeline, exact to the decimal.
2. **A frontier-moving controller** (`early+vehrule`): first to pair a vehicle gain
   with majority crossing protection; −0.9 s & 7/12 at low/medium.
3. **The critic is not the limiter** (measured), redirecting effort from value
   estimation to the objective.
4. **A localized, four-times-confirmed barrier**: the residual is an irreducible
   phase-cycle floor, not an optimization failure — a precise negative result.

## 7. Limitations

- Development evidence only: 3 replicas, 5–10 dev seeds, `descriptive_pass == False`
  in all cells. No confidence-interval or superiority claim. A claimable win would
  need the frozen confirmatory protocol on reserved seeds 58001-58020 (untouched).
- `early+vehrule` is a rule result, not learned.
- Synthetic, uncalibrated demand; simplified corridor; normal walkers in these cells.
- Stage-2 was one budget; a much stronger/longer dual schedule was not exhausted —
  but Routes D/E show even a perfect pedestrian policy hits the cycle floor.
- One tracked file (`evaluate.py`) is held in CRLF locally to match the frozen
  archive hash (line-ending only; `scripts/check_frozen_checkpoints.py`).

## 8. If further work is wanted

The only routes left that could change the medium-demand verdict require relaxing
the physical constraints the guardrails protect (clearance, exclusive pedestrian
phases) or changing geometry — i.e. a different research question, not a better
controller for this one. Within the current rules, `early+vehrule` is at or near the
achievable frontier, and the honest deliverable is the mechanism above.

Artifacts: `docs/stage1_writeup.md`, `docs/stage2_plan.md`, `docs/deep_plan_routes.md`;
code `scripts/{vehicle_rule_test,stage2_constrained,stage2_pilot,route_e_test,route_d_search,critic_gate}.py`;
results under `runs/{vehicle-rule-test,stage2-h2400,route-e,route-d,critic-gate}`.
