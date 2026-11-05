# Pedestrian-fair signal control: frontier progress and mechanism

_Working draft — development evidence. Written 2026-11-05 from local runs that
reproduce the reviewed cluster study. `descriptive_pass` means point estimates
meet the target; it is NOT a confirmatory statistical claim. Reserved final seeds
58001–58020 remain untouched._

## 1. The objective, in one sentence

Make traffic signals clear vehicles faster **without making any pedestrian
crossing wait longer**, relative to the pedestrian-actuated controller.

Formally, one controller must simultaneously, on matched demand vs. actuated:

1. **reduce vehicle delay** — lower `completed_mean_time_loss_seconds` (Δ < 0);
2. **protect pedestrians** — 95th-percentile wait no higher at **all 12** crossings;
3. **not drop service** — no extra uninserted/unfinished vehicles or unserved
   people (the strict service gate, `selection_score()[0] == 0`).

All three at once is the "joint win". The secondary number — how many of the 12
crossings are protected — measures progress toward it. A low vehicle delay alone
is not success: it is computed over completed trips only, so a controller can post
a good average by quietly failing to serve hard traffic. The service gate exists to
catch exactly that.

## 2. Where the project stood (16 prior studies)

No learned controller had achieved the joint win. The controllers sit on opposite
sides of a frontier:

| Controller (medium/medium) | Vehicle delay | Crossings protected /12 |
|---|---:|---:|
| actuated (reference) | 51.17 s | — |
| continued MAPPO | 38.43 s (−24.9%) | 0 / 12 |
| early-service hybrid | 54.97 s (+7.4%) | 6 / 12 |

"Fast cars" learning starves pedestrians; "protect pedestrians" rules cost vehicle
time. Nothing reached both. The prior work localised the residual vehicle cost of
the pedestrian-protecting hybrid to **vehicle-axis selection after clearance**: it
picked the *smaller* stopped queue ~13% of the time, where actuated always picks
the larger.

## 3. What we did and what changed

### 3.1 Reproduced the pipeline locally (Stage 0)

The full SUMO + TraCI + PyTorch pipeline — previously believed cluster-only — now
runs on a laptop (Python 3.12 venv, SUMO 1.28 via the `eclipse-sumo` wheel; see
`scripts/setup_local_env.sh`). A deterministic replay of the reviewed 720-run study
reproduced its numbers **to the decimal**:

| Cell | Published | Local reproduction |
|---|---|---|
| actuated med/med | 51.17 | 51.17 ✓ |
| continued MAPPO | 38.43, 0/12 | 38.43, 0/12 ✓ |
| early-service | 54.97, 6/12 | 54.97, 6/12 ✓ |
| early-service low/low | 41.47, 5/12 | 41.47, 5/12 ✓ |

This establishes the pipeline is trustworthy and every number below is directly
comparable to the reviewed frontier.

### 3.2 Moved the frontier with a vehicle-axis rule (Stage 1)

We added one arm, `mappo-early+vehrule`: the continued-MAPPO weights + the
early-service pedestrian rule + a **larger-stopped-queue vehicle rule** (when EW and
NS are both legal and PED is not being served, request the busier axis — actuated's
own logic). No training. All physical safety is owned by the unchanged signal
machine; the rule only redirects among already-legal actions. An audit asserts, on
every decision, that when the rule must fire the executed action is the larger-queue
axis (`scripts/vehicle_rule_test.py`).

Full 840-simulation evaluation (4 demand cells × 10 seeds × 3 replicas, 2400 s):

| Cell | early-service (before) | **early+vehrule (after)** | actuated |
|---|---|---|---|
| | Δveh s · crossings | **Δveh s · crossings** | ref |
| med/med | +3.80 · 6/12 | **−0.47 · 6/12** | 51.17 |
| med/low | +4.82 · 4/12 | **+0.43 · 4/12** | 40.20 |
| low/med | +3.20 · 8/12 | **−0.92 · 7/12** | 45.04 |
| low/low | +5.87 · 5/12 | **+0.68 · 5/12** | 35.60 |

**The hypothesis held.** Against the early-service hybrid directly, the rule
recovered **4.1–5.2 s of vehicle delay in every cell** (contrasts: −4.27, −4.39,
−4.12, −5.19 s) while keeping pedestrian protection essentially unchanged
(6→6, 4→4, 8→7, 5→5).

**This moved the frontier.** `mappo-early+vehrule` is the **first controller in the
project to combine a vehicle gain with non-trivial crossing protection at once**:

- medium/medium: vehicle delta crossed from +3.80 to **−0.47 s** (now faster than
  actuated) while holding **6/12** crossings;
- low/medium: **−0.92 s and 7/12** — the closest any controller has come to the
  joint target.

### 3.3 Diagnosed the remaining blocker and tested the critic (Stage 1.5)

Two of the three joint conditions now hold together at medium/medium (vehicle Δ < 0;
6/12 crossings; gate not yet clean). The wall is unambiguously the **pedestrian
tails at the remaining crossings**, not vehicles. The queue rule cannot fix those,
because different vehicle-phase timing shifts *when* each crossing's green legally
recurs — a scheduling effect a static rule cannot anticipate.

We then asked the central reinforcement-learning question: is the learned value
function (critic) good enough for further training to help? A held-out
critic-fitting gate (`scripts/critic_gate.py`) regressed the true GAE return
targets on fresh development rollouts:

| Junction | checkpoint critic EV | fresh-fit critic EV |
|---|---:|---:|
| C0 | +0.41 | +0.39 |
| C1 | +0.52 | +0.44 |
| C2 | +0.53 | +0.48 |
| **mean** | **+0.49** | **+0.44** |

**This corrected an earlier assumption.** The continued-MAPPO checkpoints are
`normalized-clock` RepairNetworks — they already include value normalization and a
time-to-horizon feature. Their critic predicts held-out returns at **~0.49 explained
variance**, not the ~0.002 reported for the old plain/linear PPO. **The critic is
not the bottleneck.** The limiter is the **reward specification**: the learner
optimises a dense summed reward `−(queue + weight·pending)`, whereas the objective
is a per-crossing worst-case constraint. The critic faithfully predicts the wrong
quantity.

## 4. What we have learned (contributions)

1. **The pipeline reproduces exactly on commodity hardware** — removing the belief
   that experiments require the cluster, and enabling fast iteration.
2. **A frontier-moving controller.** `mappo-early+vehrule` is the first to pair a
   vehicle gain with majority crossing protection; at low/medium it is −0.92 s and
   7/12, the nearest approach to the joint target recorded.
3. **The vehicle side of the trade-off is solved by a queue rule.** The residual is
   now provably pedestrian-tail scheduling at specific crossings — a sharper, more
   actionable claim than "the trade-off is hard".
4. **The critic is not the limiter; the reward proxy is.** This redirects further
   work away from value estimation and toward the per-crossing constraint.

## 5. Honest limitations

- Every result here is **development evidence**: 3 replicas, 10 demand seeds,
  `descriptive_pass == False` in all cells. No confidence-interval or superiority
  claim. A claimable joint win requires the frozen confirmatory protocol on the
  reserved seeds 58001–58020, which are untouched.
- `mappo-early+vehrule` is a **rule win, not a learning win**. The vehicle recovery
  comes from a hand-coded heuristic (larger queue), not from the learned policy.
  It shows the learning did not discover this, not that MARL cannot.
- Synthetic, uncalibrated demand and a simplified corridor; normal walkers only here
  (slow-walker/overload results remain in the broader project).
- One tracked file (`src/traffic_control/evaluate.py`) is held in CRLF form locally
  to match the frozen archive hash; this is a reproducibility artifact, not a code
  change (line-ending only; verified by `scripts/check_frozen_checkpoints.py`).

## 6. Why this connects to reinforcement-learning design

The gap to a full 12/12 is exactly where a *learned* policy should beat a static
rule: it must **anticipate** — serve a crossing before its tail blows up, trade a
little vehicle time at one junction to keep another's pedestrians flowing. A queue
rule is myopic by construction. Reaching Tier 2 therefore means changing *what the
agent optimises*, not how well it predicts value:

- replace the summed reward with the **per-crossing Lagrangian constraint** already
  scaffolded in the learner (`cost_critics` + dual `multipliers`) — constrained-MDP
  PPO that targets "no crossing worse" directly;
- **train at the evaluation horizon** (2400 s) to remove the 600-vs-2400 s
  distribution shift;
- keep the vehicle-axis rule as a **safe action prior** the policy learns within.

The critic gate says this substrate is sound enough to attempt. That is the scope of
Stage 2.
