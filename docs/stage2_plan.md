# Stage 2 plan: constrained MAPPO for per-crossing pedestrian fairness

_Written against the VERIFIED code surface (2026-11-05). Every dimension and
interface below was checked against the actual loaded checkpoint, not inferred.
The two landmines found during mapping are called out explicitly so they do not
surface mid-run._

## Objective

Reach a **learned** controller that, in at least one demand cell, descriptively
meets the joint target: vehicle Δ < 0 AND all 12 crossings p95 no worse AND clean
service gate. Stage 1 showed a hand-coded rule reaches vehicle-Δ<0 with 6–7/12;
Stage 2 asks whether *training the actor* under a per-crossing constraint can
protect the remaining crossings without giving the vehicle gain back.

Why this is the right lever (from Stage 1.5): the critic already fits held-out
returns (~0.49 EV) — value estimation is NOT the bottleneck. The bottleneck is the
**reward**: a dense sum `−(queue + w·pending)` cannot express "no single crossing's
tail gets worse." Stage 2 changes *what the agent optimises*.

## Verified facts the implementation must respect

**Model.** `RepairNetworks`, `variant='normalized-clock'`, method `mappo`,
`observation_schema='legacy-v2'`. Verified dims (NOT the explorer's 50/150 estimate):
- `local_dim = 42`, `global_dim = 126` (42 × 3 junctions).
- **Main critic input = 127** (126 global + 1 time-to-horizon clock), via
  `critic_inputs()` + `prediction()` (with PopArt `RunningValueScale`).
- **Cost critic input = 126** (raw global_state, NO clock), output = 4
  (one per local crossing), via inherited `JunctionNetworks.values()`.

**LANDMINE 1 — asymmetric critic inputs.** The main critic eats 127-dim; the cost
critic eats 126-dim. The Lagrangian code must feed cost_critics the **raw
global_state (126)**, never the clock-augmented 127 tensor. Mixing them crashes
(`127 vs 126` matmul error). Keep two tensors: `inputs` (127, main) and
`global_state` (126, cost).

**LANDMINE 2 — cost_critics are untrained dead weight.** `RepairLearner` never
touches `cost_critics`; the checkpoint's cost_critics are random init. Stage 2 must
(a) reinitialize them deterministically and (b) add their training + the dual
multiplier loop. Their presence in the checkpoint does NOT mean they carry signal.

**Events.** Rollout `events.json` rows carry `crossing_id`, `wait_seconds`,
`reached_at`, `entered_at`, `pending_age_seconds`, `lost` — everything the
per-crossing violation cost needs. Crossing ids are `:C{j}_c{k}`, 4 per junction.

**Trace rows** carry `observation, global_state, mask, action, time, vehicle_queue,
uninserted_due, pending_count` — the reward and GAE inputs. `prepare_batch`
(ppo_repair) already builds `inputs/returns/old_logp/valid/masks`; it does NOT
build cost tensors.

**Checkpoint contract** (`RepairNetworks.from_snapshot`): `saved['model']` must have
`schema` (=`ppo-repair-pilot-v1`), `method`, `variant`, `features`, `weights`.
`observation_schema` is NOT in the snapshot — re-set to `'legacy-v2'` after load
(as `load_model` does at `demand_service_test.py:160`). Evaluation reuses
`vehicle_rule_test` / `demand_service_report`, so the final checkpoint must load
through that path unchanged.

**Seeds.** Verified-free range for Stage 2 (no overlap with E04 810k, E09 60k,
E13 990k, E15/audit 930k, guided 870k, demand-service 1010k, reserved 58001–58020):
- training `940001 + replica*1000 + episode`
- validation `941001, 941002`
- evaluation (dev) `942001…942010`
- reserved `58001–58020` **untouched**.

**Reward/GAE.** `−(vehicle_queue + uninserted_due + weight·pending_count)/10`,
`gamma=.99`, GAE `lambda=.95` (`advantages()`); weight was 3 in all learned arms.
Keep weight=3 so the only change vs. the continued-MAPPO parent is the constraint.

## Design

Warm-start from the three continued-MAPPO checkpoints (so we measure the *delta*
from the current frontier, not re-train from scratch). Apply three changes, each a
factor so attribution is clean:

1. **Per-crossing Lagrangian constraint (C).** Add cost critics + dual multipliers
   to the RepairLearner path, ported from `fairness/learning.py:59-104`:
   - cost per crossing = violation indicator `wait_seconds > threshold` (threshold
     = that crossing's actuated p95, or a fixed 20 s to start), credited to the
     reward step nearest the event time, minus a 5% slack;
   - undiscounted cost returns; `cost_adv = cost_returns − cost_value`;
   - `adv ← adv − Σ_c λ_c · cost_adv_c`; dual update `λ_c ← clip(λ_c + 0.05·(rate_c − 0.05), 0, 100)`.
   - cost critic trained with MSE on the raw 126-dim global_state (LANDMINE 1).
2. **Horizon match (H).** Train at `end_seconds=2400` (eval horizon), not 600 —
   removes the 600-vs-2400 distribution shift. Keep a 600 s control arm.
3. **Vehicle-axis envelope (V).** Reuse Stage 1's larger-queue rule as a training
   action prior via the `ProtectedPolicy.align_training_trace` pattern: forced rows
   become critic-only (not actor-trained). Keeps the vehicle gain while the actor
   learns pedestrian timing within the envelope.

**Factorial (development, dev seeds only):** at minimum
`{C} × {H=600, H=2400}`, MAPPO, 3 replicas, warm-started, 32 episodes each, 4 demand
cells. Add `{+V}` as a second pass if C alone under-protects. Controls: the
unchanged continued-MAPPO parent and Stage-1 `early+vehrule` are already evaluated,
so they are the reference rows — no need to retrain them.

**Checkpoint selection.** Lexicographic `selection_score` (service, worst-crossing
p95, vehicle loss) on `vmedium-pmedium` + `vmedium-plow` validation only, every 8
episodes — same rule as `service_priority/pilot.py`, so selection is comparable.

## Acceptance / exit criteria

- **Win:** a selected checkpoint shows `descriptive_pass` (vehicle Δ<0, 12/12,
  gate 0) in ≥1 cell on dev seeds → proceed to Stage 3 (reserved-seed confirmation).
- **Partial:** crossings protected increases beyond Stage-1's 6–7/12 with vehicle
  Δ still <0 in a cell → report as progress; the constraint helps but isn't enough.
- **Null:** no improvement over `early+vehrule` → the per-crossing constraint does
  not navigate the trade-off at this budget; this is a clean, strong negative and
  we stop. Record the λ trajectories and cost-critic EV as evidence of *why*.

All outcomes are development evidence (3 replicas, dev seeds). A claimable win still
requires Stage 3 on the reserved seeds.

## Implementation steps (then run)

1. **`ConstrainedRepairLearner`** (new, in `scripts/stage2_constrained.py` or a new
   module): subclass/compose `RepairLearner`; add `thresholds`, `multipliers`, cost
   tensors in a `prepare_cost_batch`, cost-critic MSE on 126-dim state, dual update.
   Reinitialize cost_critics deterministically at construction.
2. **Training pilot** modelled on `ppo_repair/pilot.py` (clean resume, `final.pt`
   via `model.snapshot()`), with `cfg_for` horizon factor and optional ProtectedPolicy
   envelope; validation/selection via `selection_score`.
3. **Smoke test** `--quick` (1 replica, 2 episodes, 180 s) end-to-end: trains,
   selects, saves a `final.pt`, and that `final.pt` **loads through
   `vehicle_rule_test.load_model`** (contract check) and evaluates. Assert cost-critic
   input width is 126 and no 127/126 mismatch fires (LANDMINE 1 guard).
4. **Full run** (background, resumable) once smoke passes; evaluate with the Stage-1
   report path; compare to `early+vehrule` and actuated.

## Risks (pre-identified, mitigated)

| Risk | Mitigation |
|---|---|
| cost-critic 126 vs main 127 crash (L1) | separate tensors; smoke-test asserts widths |
| cost_critics carry stale random signal (L2) | reinitialize at construction; log cost EV |
| warm-start + new cost head → unstable λ | start λ=0, small step 0.05, clip [0,100]; log λ trajectory |
| horizon change confounds critic PopArt | RunningValueScale handles it; keep 600 s control arm |
| final.pt not loadable by eval path | smoke step 3 is an explicit contract check |
| seed collision | verified-free 94xxxx range; keep overlap guard |
