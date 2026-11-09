# Improvement plan: closing the vehicle–pedestrian trade-off

**Audience.** An implementing agent with SUMO + TraCI + PyTorch available (the NUS
Slurm cluster, or a host with the pinned `torch 2.3 / numpy 1.26.4` env and
`sumo 1.27.1`). This plan assumes you can run the full pipeline; the authoring
machine could not (no `traci`, no `sumo`, Python 3.9), so every code reference was
verified by reading the repo, NOT by executing end-to-end. Stage 0 exists to catch
any drift between this plan and the live environment before you trust a number.

**How to read this.** §1 defines the benchmark (already in code). §2 is the blocker
ledger — the five things that actually prevent the joint win, each with its
evidence and its falsifiable test. §3 is the staged plan; every stage names exact
files, function signatures, invariants, and accept/abort criteria. §4 is the global
guardrail set. §5 is the decision tree. §6 is the risk register. Do not reorder
stages; later stages depend on earlier exit artifacts.

---

## 1. The benchmark — "improvement" is already defined in code

Read before writing anything:

- `src/traffic_control/fairness/analysis.py`
  - `selection_score(result, baseline) -> (service, worst_crossing_p95_delta, vehicle_loss)`.
    Element 0, the **service gate**, must be `0`. It independently penalises every
    way of faking the vehicle average: fewer completed, more unfinished, more
    uninserted, unserved persons, and per-crossing requests pending-at-cutoff /
    unresolved-after-followup (`analysis.py:37-51`). Element 2,
    `completed_mean_time_loss_seconds`, is over completed trips only — which is why
    a low value is NOT a win on its own.
  - `primary_inference(groups, bootstrap=1000)` → `descriptive_pass` and
    `statistical_support`. The confirmatory branch (`analysis.py:99`) only activates
    at **≥20 demand seeds, ≥5 replicas, ≥300 minimum crossing entries**.

- **Joint target** (vs. pedestrian-actuated on matched demand), all three at once:
  1. `vehicle_delta < 0`,
  2. `max_{12 crossings}(p95_cand − p95_act) ≤ 0`,
  3. `service_gate == 0`.
  Conjunction = `point_pass` (`service_priority/pilot.py:169-170`) /
  `descriptive_pass` (`analysis.py:96`).

- **Two tiers of claim.** `descriptive_pass` = point estimates satisfy (1)∧(2)∧(3).
  `statistical_support` = paired bootstrap gives a *simultaneous* upper bound
  `< 0` on vehicle delta AND `≤ 0` on the worst crossing, gate clean. Only the
  second is a reportable "joint win". Say which one you have, every time.

**Frontier today** (E15, medium/medium, `results/demand-service-review/REVIEW.md`):

| Controller | Vehicle loss | Crossings p95 no-worse / 12 | Joint pass? |
|---|---:|---:|---|
| Actuated (reference) | 51.17 s | — | — |
| Continued MAPPO | 38.43 s (−24.9%) | 0 / 12 | ✗ tails |
| Early-service hybrid | 54.97 s (+7.4%) | 6 / 12 | ✗ vehicle + tails |

No controller in 16 studies satisfies (1)∧(2)∧(3) together. The goal is to move
**one single controller** across all three, first descriptively then with support.

---

## 2. Blocker ledger — why the joint win has not happened

Five distinct blockers. They differ in kind, which is why no single past change
cleared them. Each entry: the claim, the evidence, and the test that would confirm
or kill it.

### B1 — The objectives conflict at the decision level (structural)
- **Evidence.** Continued MAPPO: −24.9% vehicles, 0/12 crossings. Early-service
  hybrid: +7.4% vehicles, 6/12. Each controller sits on one side of the frontier;
  none in the middle (E15 table above; E02/E10 corroborate).
- **Nature.** Serving PED sooner/more lengthens the cycle (vehicle cost); maximising
  vehicle throughput starves crossings. Partly intrinsic to the corridor + demand.
- **Test.** If a controller can be placed strictly inside both constraints in even
  one demand cell, the conflict is *navigable* here, not absolute. Stages 1–2 test
  this; a persistent failure across both is itself the headline negative result.

### B2 — Vehicle-axis misallocation after clearance (specific, attackable)
- **Evidence.** `results/demand-service-review/vehicle_choice_diagnostic.json`:
  the hybrid selects the *smaller* stopped queue ~13% of post-clearance choices
  (17.27 / 132.30 per medium/medium window); actuated selects the larger queue by
  construction (`fairness/signals.py:101-108`). Empty-PED requests explain the
  low-pedestrian-demand cost but NOT the medium-demand vehicle gap
  (`REVIEW.md` "Empty service cannot by itself explain the whole medium-demand gap").
- **Nature.** A concrete, correctable decision — the single highest-leverage lever.
- **Test.** Override those choices toward the larger queue (frozen model, no
  training) and measure the vehicle-delta recovery — Stage 1.

### B3 — Policy/critic degeneracy (the learning is weak)
- **Evidence (E06, `results/service-policy-audit/REVIEW.md`).** Under argmax,
  actors collapse to near-deterministic starvation: `plain-mappo-r0` C0 EW has
  **2354 consecutive queued/no-green samples**; mean gap between top-two action
  probabilities = **0.00823**. Critic explained variance ≈ **0.002–0.003** — i.e.
  the value function barely predicts returns. Sampling hides the symptom but does
  not fix the estimator.
- **Root cause in code (`fairness/learning.py:82-92`).** The value loss is scaled by
  `returns.std()` (`value_scale`) while the critic output stays in raw reward units
  and stored `old_value` predictions are never renormalised. This is NOT the
  value-normalise-then-denormalise scheme of marlbenchmark/on-policy; it is a known,
  self-flagged implementation gap.
- **Test.** The critic-fitting gate (Stage 1.5): can a correctly-normalised critic
  fit held-out return targets at matched budget, when the current one cannot? If the
  critic can't learn value, retraining (Stage 2) cannot allocate well regardless of
  any safety envelope.

### B4 — Physical occupied-crossing lockout (environment, not policy)
- **Evidence (E14, `results/failure-mechanism-review/REVIEW.md`).** In congested
  replays a stalled vehicle on an internal edge keeps a crossing occupied for
  **986–2084 s**; the safety machine then correctly refuses conflicting releases and
  **the actor has no legal action** (`SafeSignal.advance`/`mask` return all-False in
  clearance). guided-r2/seed-890005: C0 in continuous clearance 320→2400 s.
- **Nature.** An absorbing state no controller can exit once entered. Earlier policy
  choices may lead *into* it, but the lockout itself is geometric/simulator.
- **Non-goal.** The guardrails forbid the only "fixes" (removing travellers,
  shortening clearance, enabling unsafe moves). This blocker is to be *measured and
  documented*, and at most *avoided* by not steering into it — never engineered away.

### B5 — Validity / protocol ceilings (bound the claim, not the behaviour)
- **Evidence.** 600 s training vs 2400 s evaluation (`demand_service_test.py:114`);
  few training/demand seeds in follow-ups; synthetic uncalibrated demand + simplified
  NUS geometry (E03); `descriptive_pass` ≠ `statistical_support`.
- **Nature.** Even a real behavioural win is not *claimable* as a joint win without
  held-out seeds, enough replicas/entries, and uncertainty. Addressed by Stage 3's
  frozen confirmatory protocol and Stage 4's limitations, not by any algorithm.

---

## 3. Staged plan

Order is fixed: **0 → 1 → 1.5 → (2 only if gated) → 3 → 4**; Stage 4 runs in
parallel from the start. Each stage lists: goal, blocker(s) targeted, exact code,
invariants, accept/abort, deliverable.

### Stage 0 — Reproduce and confirm the frontier (no modelling)
**Goal / blockers.** Trust the harness before any new number; baseline for B1–B5.

1. Environment probe: `python -c "import traci"`; `which sumo`; `echo $SUMO_HOME`
   (resolved by `scenario.py::executable`); `python -c "import torch,numpy;print(torch.__version__,numpy.__version__)"`.
2. `python -m unittest tests.test_service_priority tests.test_fairness_analysis
   tests.test_demand_service_test -v` — all pass. (First two already pass under
   Python 3.9 on the authoring machine; the third needs torch + the reviewed archive.)
3. Deterministic replay of E15 against committed hashes:
   `python -m scripts.demand_service_test --verify-replay --workers 1
   --output runs/demand-service-replay`. Reuses the frozen archive
   `runs/pedestrian-finetune-cluster-results.zip` (SHA pinned at
   `demand_service_test.py:27`); event/decision hashes must match exactly.

**Accept.** Replay passes; the frontier table reproduces.
**Abort.** Any hash mismatch → STOP, report environment drift; nothing downstream
is valid until fixed. **Deliverable.** `docs/stage0_repro.md`: SUMO/torch/numpy
versions, replay result, reproduced frontier numbers.

### Stage 1 — Vehicle-axis service rule (frozen model, NO training)
**Goal.** Recover the hybrid's vehicle cost without sacrificing its pedestrian
protection. **Blockers.** B2 directly; probes B1's navigability. **Zero training.**

**New component — a composed frozen wrapper** modelled exactly on
`scripts/demand_service_test.py::EarlyServicePolicy` (which overrides only the
*requested action* and lets the unchanged `SafeSignal` own all transitions). Add a
sibling, e.g. `VehicleAxisPolicy` in a new module `scripts/vehicle_rule_test.py`:

- Start from **continued MAPPO** weights (the exact checkpoints E15 used;
  load via the archive path + hash check in `demand_service_test.py::load_model`).
- Compose two overrides on the base policy's `distribution(junction, obs, masks)`:
  1. **Pedestrian rule (unchanged):** pending local request (`:<tls>_c*/count > 0`)
     ∧ legal PED (`mask[2]`) ⇒ point-mass on action 2. (Copy from `EarlyServicePolicy`.)
  2. **Vehicle-axis rule (new):** when PED is *not* forced and both EW and NS are
     legal (`mask[0] and mask[1]`), compare per-axis stopped queues and put a
     point-mass on the larger-queue axis; tie → hold current stage, matching
     actuated's `(queues[i], -i)` tie-break (`fairness/signals.py:108`). Read queues
     from observation features `<edge>/queue` summed per axis via
     `connected.incoming_groups(3)[tls]` (features laid out at
     `fairness/environment.py:137-142`).
- **Hard invariants (assert in code):** never place mass on an action where
  `mask[a]` is False; never override during min-green/clearance (mask already
  forbids); keep the SAME categorical draw for any non-forced decision so arms are
  paired. Add an audit assertion symmetric to the early-service one
  (`demand_service_test.py:185`): "when the vehicle rule fires, executed action ==
  larger-queue axis", so a wrapper bug fails loudly in `audit_outcome`.

**Arms (all frozen, zero training episodes):** `mappo-continue` (baseline),
`mappo-early` (pedestrian rule only — already exists), `mappo-early+vehrule` (new).
Keep `fixed/actuated/pressure` baselines; actuated is itself a larger-queue selector
and bounds the rule's ceiling.

**Protocol.** Reuse E15 development seeds `1010001–1010010`, all four demand cells,
3 continued-MAPPO replicas, 2400 s horizon. Write to a NEW dir `runs/vehicle-rule-test`
(new source ⇒ new fingerprint in `make_protocol`; the hash guard refuses a mixed
dir — intended). The `used & set(seeds)` guard (`demand_service_test.py:94`) must
keep rejecting any overlap with training/eval/reserved seeds — do not weaken it.
Evaluate through `scripts/demand_service_report.py::report` for per-crossing p95,
the service gate, and paired bootstrap intervals.

**Accept (any of the three is a valid result):**
- *Win:* `mappo-early+vehrule` meets (1)∧(2)∧(3) in ≥1 cell → Stage 3.
- *Partial:* vehicle cost shrinks but ≥1 condition still fails → quantify residual,
  go to Stage 1.5 then possibly 2.
- *Null:* no vehicle gain → the smaller-queue choices were anticipatory, not
  mistakes; the frozen-rule ceiling is reached; retraining (Stage 2, gated) is the
  only remaining lever. All three are clean, mechanism-level findings.
**Caveat to report honestly.** If the rule fully recovers vehicles, you have
rebuilt actuated's logic by hand — a *rule* win, not a MARL win (E06 note:
"reproducing actuated's choices recovers a reference-like policy"). State this.
**Deliverable.** `runs/vehicle-rule-test/REPORT.md` + a short interpretation note.

### Stage 1.5 — Critic-fitting gate (cheap diagnostic, gates Stage 2)
**Goal.** Decide whether retraining can possibly help. **Blocker.** B3.
This is E06's own recommended next check and MUST pass before Stage 2 — otherwise
Stage 2 spends cluster-days training an actor on top of a critic that cannot
predict value.

**Procedure (no new long training).**
1. Collect a small held-out set of development trajectories (frozen-model rollouts
   via `fairness/environment.run(..., trace_sink=...)`; the trace rows already carry
   `observation`, `global_state`, `action`, and the per-row reward inputs
   `vehicle_queue`, `uninserted_due`, `pending_count` used in `learning.py:52-53`).
2. Fit two critics at a *matched update budget* on those targets:
   - **Current scheme:** value in raw reward units, loss scaled by `returns.std()`
     (reproduce `learning.py:82-92` exactly).
   - **Normalised scheme:** maintain a running mean/var of returns, train the critic
     on normalised targets, denormalise before computing advantages (the
     marlbenchmark/on-policy scheme the code comments reference). Separate actor and
     critic optimisers.
3. Report, per scheme: explained variance, raw-value bias/RMSE vs. Monte-Carlo
   returns, and save/resume unit-consistency. Use a separate held-out trajectory set
   for evaluation; do not evaluate on the fit set.

**Accept (gate to Stage 2).** Normalised critic reaches materially higher explained
variance than the current ≈0.002–0.003 on held-out returns AND passes unit
round-trip. **Abort.** If neither critic can fit held-out returns, value learning is
not the bottleneck you can fix by normalisation; do NOT run Stage 2's RL — instead
record that the learned value estimator cannot support this task at the current
budget/horizon, and treat Stages 1 + 4 as the deliverable.
**Deliverable.** `runs/critic-fit-gate/REPORT.md` with the EV/RMSE table.

### Stage 2 — Horizon-matched retraining with safety rules as a prior (gated)
**Run only if Stage 1 is partial/null AND Stage 1.5 passed AND cluster runway exists.**
Two known confounds are fixed together, so attribute via the control arms below.

**Changes vs. all prior training:**
1. **Horizon match (fixes part of B5).** Train at the 2400 s evaluation horizon
   (or a documented curriculum to it), not 600 s. Keep a 600 s control arm to isolate
   the horizon effect.
2. **Normalised critic (fixes B3)** — only the variant that passed Stage 1.5.
   Implement in `fairness/learning.py` as a new `Learner` option; keep the old path
   available behind a flag so the ablation is clean. Preserve the existing
   forced-transition masking (actor loss masked on forced rows, critic trained on
   them) and the reward-at-t+1-belongs-to-action-at-t convention (`learning.py:5`).
3. **Safety rules as a differentiable prior, not post-hoc override (targets B1/B2).**
   Reuse `service_priority/control.py::ProtectedPolicy`, which already handles the
   PPO subtlety (forced-service rows critic-only; actor log-probs use the *effective*
   mask via `align_training_trace`). Extend its `service_mask` with a **vehicle-side
   max-red / larger-queue admissibility rule**, so the actor learns allocation
   *within* an envelope that protects both pedestrians (age deadline) and starved
   vehicle movements (max-red). This directly targets the C0-starvation failure mode
   (`service-priority-cluster/REVIEW.md`: 12 s EW green over a whole run). The
   existing per-crossing Lagrangian machinery (`learning.py:59-104`,
   `cost_critics` in `networks.py:43`) stays; the max-red rule is a *mask* constraint,
   not a new reward term, to avoid reward-shaping confounds.

**Protocol (follow E05/E13 conventions, `service_priority/pilot.py` as template):**
- Factorial: `{pedestrian-rule, pedestrian+vehicle-rule} × {600 s, matched-horizon}
  × {current-critic, normalised-critic}`, IPPO and MAPPO, ≥3 development replicas,
  4 demand cells, development seeds only. Change ONE factor at a time in the write-up.
- Per-replica init seed `46001 + replica` (`pilot.py:91`); matched demand seeds across
  arms; resumable atomic checkpoints; lexicographic `selection_score` checkpoint
  selection on `vmedium-pmedium`/`vmedium-plow` validation only (`pilot.py:110-128`).
- New source ⇒ new fingerprint ⇒ fresh checkout + new output dir on the cluster
  (the `make_protocol` hash guard refuses a mixed dir — intended).
- Submission mirrors `scripts/run_service_priority_cluster.sh`: one allocation,
  source the pinned venv, unit tests first, pilot module, then the packaging script.
  Provide SBATCH options at submit time per account policy.

**Accept.** A single trained controller with `descriptive_pass == True` in ≥1 cell
on development seeds, gate clean → Stage 3. **Abort.** If still none after the
factorial, that is the project's definitive negative result — stop and write up.
**Deliverable.** `runs/<study>/REVIEW.md` with the full factorial and ablation.

### Stage 3 — Confirmatory held-out evaluation (reserved seeds, ONCE)
**Run only if Stage 1 or Stage 2 produced a `descriptive_pass` candidate.**
**Blocker.** B5 (the only path to a claimable win).

- **Freeze first.** Lock the method, budget, and the single chosen controller
  *before* touching seeds `58001–58020`. Record the freeze: commit hash + protocol
  fingerprint in `docs/final_protocol_freeze.md`.
- Evaluate on `58001–58020` at full 2400 s with **≥5 replicas**, so the
  `primary_inference` statistical branch is eligible (`len(groups) ≥ 20`,
  `len(replicas) ≥ 5`, `minimum ≥ 300`; `analysis.py:99`). If you cannot meet these
  counts, report `descriptive_pass` only and do NOT claim statistical support.
- **Reportable win.** `statistical_support == True`: simultaneous bootstrap upper
  bound `< 0` on vehicle delta AND `≤ 0` on worst crossing, gate clean.
- **Irreversibility.** Once these seeds inform any selection they are no longer
  held-out. Use them exactly once, for one pre-registered candidate. No second bite.
**Deliverable.** `results/final-confirmatory/REVIEW.md`.

### Stage 4 — Consolidate the write-up (always, in parallel)
The strongest contribution already exists: a measured, mechanistic account of the
trade-off. Independently of 1–3:
- Consolidate E01–E16 around current evidence; backbone = `docs/experiments/README.md`
  + `docs/experiments/registry.json`. Keep every negative setting, failed
  intervention, slow-walker and overload result.
- Carry all B5 limitations verbatim: uncalibrated synthetic demand, simplified NUS
  geometry, few seeds, 600-vs-2400 horizon, finite-horizon vehicle counts,
  `descriptive_pass` ≠ `statistical_support`, and B4 (occupied-crossing lockout) as
  an environment limit, not a policy failure.
- Commit the owner's uncommitted local code/reviews. Reconcile Team 38/28 naming and
  the deadline. Disclose LLM assistance per the catalogue's closing section. Do NOT
  invent individual contributions.

---

## 4. Global guardrails (every stage)
- Wrappers/masks may only *remove or redirect within* `SafeSignal.mask(...)`; never
  add an action outside it, never shorten min-green / yellow / all-red / clearance,
  never choose pedestrian green duration.
- One controller must clear (1)∧(2)∧(3) **together**; never present a vehicle win and
  a pedestrian win from *different* controllers as progress.
- New/changed source ⇒ new output directory (fingerprint guard enforces it).
- Reserved seeds `58001–58020`: untouched until Stage 3, once, pre-registered.
- Always report service counts, per-crossing sample sizes, all 12 tails, and paired
  uncertainty alongside any vehicle number.
- Never impute zero for missing/unserved waits (`analysis.py` does not; keep it so).
- `squeue` emptiness ≠ success; require full output coverage + passing hash/replay checks.
- B4 is off-limits to "fix": do not remove travellers, shorten clearance, or enable
  unsafe movements to escape a lockout.

---

## 5. Decision tree
```
Stage 0 replay passes? ── no → STOP (fix env)
          │ yes
Stage 1 (frozen vehicle rule)
   ├─ WIN  (1∧2∧3 in a cell) ───────────────→ Stage 3 → Stage 4
   ├─ PARTIAL (vehicle cost ↓, not all 3) ──→ Stage 1.5
   └─ NULL  (no vehicle gain) ──────────────→ Stage 1.5
Stage 1.5 critic gate passes?
   ├─ yes → Stage 2 (gated retrain)
   │          ├─ descriptive_pass in a cell → Stage 3 → Stage 4
   │          └─ still none → STOP, write up (Stage 4) [definitive negative]
   └─ no  → STOP, write up (Stage 4) [value learning is the ceiling]
Stage 3 statistical_support?
   ├─ yes → JOINT WIN, claimable (Stage 4 reports it)
   └─ no  → report descriptive_pass only; do not claim a win (Stage 4)
Stage 4 runs in parallel throughout.
```

## 6. Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| Stage-1 "win" is just actuated-by-hand (B2) | high if it wins | Report as a rule result, not MARL; keep actuated bound visible |
| Critic can't fit even normalised (B3) | moderate | Stage 1.5 gate aborts before wasting Stage-2 cluster time |
| Stage-2 confounds (horizon × critic × rule) | moderate | Factorial with one-factor-at-a-time arms; 600 s + current-critic controls |
| Demand-seed overlap contaminates held-out | low | `used & set(seeds)` guard (`demand_service_test.py:94`); never relax |
| Occupied-crossing lockout dominates a cell (B4) | setting-dependent | Document per E14; exclude from "fixable"; do not engineer around |
| Over-claiming from point estimates (B5) | high human-error | Distinguish `descriptive_pass` vs `statistical_support` in every table |

## 7. Effort / likelihood
| Stage | Training? | Cluster time | Chance of joint win | Value if it fails |
|---|---|---|---|---|
| 0 Confirm frontier | no | hours | n/a | trustworthy baseline + reproduction |
| 1 Vehicle-axis rule | no | hours–1 day | low–moderate | quantifies the exact decision cost (B2) |
| 1.5 Critic gate | no (short fits) | hours | n/a | decides if retraining can help (B3) |
| 2 Retrain w/ rules | yes | days | moderate | isolates horizon + envelope + critic (B1/B3) |
| 3 Held-out confirm | no | hours | — | the only *claimable* win (B5) |
| 4 Write-up | no | — | — | the result you already have |

**Recommended order:** 0 → 1 → 1.5 → (2 if gated) → 3, with 4 in parallel. Stage 1
is the single highest-leverage, lowest-risk experiment: it attacks E15's localized
mechanism (B2) with no training and reuses an audited wrapper pattern. Stage 1.5 is
the cheap insurance that stops Stage 2 from building on a critic (B3) that cannot
learn value. The honest expected outcome, given 16 prior studies, is a sharper and
better-explained negative result — which is itself the project's strongest
contribution — with a bounded, real chance of a per-cell descriptive win from
Stages 1–2 and a claimable win only if Stage 3's held-out support holds.
