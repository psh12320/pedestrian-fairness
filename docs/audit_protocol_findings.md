# Independent protocol and results audit — 3 November 2026

This report records the independent pre-repair inspection. The binary demand-writing
and insertion-delay reporting changes were subsequently implemented; see the
[consolidated audit](deep_audit_2026_11_03.md) for their verification and limitations.

This audit reads existing source and the downloaded PPO-repair archive independently. It does not edit controllers, checkpoints, old result files, or their protocols. Reproduce the numerical checks with `python scripts/deep_audit_protocol.py`; the compact evidence is `runs/deep-protocol-audit.json`. The other independent audits cover PPO mathematics and physical safety.

## Findings that can affect learning or interpretation

### 1. The actual learning budget is only about 2,400 useful choices per actor

The repaired pilot trains each actor for 64 episodes of 600 seconds, but most seconds are locked minimum greens, yellow, all-red, pedestrian admission or clearance. Reading all 18 archived learning logs gives **2,230–2,736 free decisions per actor over the entire training run**, roughly 35–43 choices per episode. The remaining transitions can train the critic but do not provide independent action choices for actor learning. There is no actor parameter sharing across junctions.

Each of the four training settings is seen only 16 times. The batch-four arm has 16 optimizer batches versus 64 in the other arms. These are legitimate equal-simulation-budget comparisons, but not evidence that all methods converged. Validation pedestrian-tail diagnostics are still changing between episodes 32 and 64; vehicle outcomes are nonmonotonic. A large total across all experiments does not equal a large training budget for one policy.

Relevant code: `ppo_repair/pilot.py:25`, `:42`, `:149`, `:153`, `:207`; actor filtering in the learner. Guided exploration inherits 600-second episodes and uses 128 episodes, so it also needs decision-count and convergence diagnostics.

**Discriminating test:** after fixing any confirmed learning defects, run a small matched seed pilot with an explicit free-decision budget and log effective actor updates, KL, action probabilities, and validation curves. Extend all comparison arms equally under a predeclared plateau rule. Do not launch a new large budget merely because this concern exists.

### 2. The primary network gives limited opportunity for coordination

The generated route distribution assigns 90% of vehicles equally to eight through routes: two traverse all three junctions, six traverse one junction. The remaining 10% take single-junction near-side turns. Consequently the expected corridor-traversing share is **22.5%**. Replaying all six archived medium-demand seeds gives **1,278 of 5,744 measurement-cohort vehicles (22.25%)** on the three-junction routes. The rest interact with one traffic light.

Medium traffic is stationary, mostly homogeneous, and balanced; a fixed cycle is naturally a competitive reference. Three physical junctions do not imply that most demand requires coordinated control. This is a design limitation, not a route-generation error.

Relevant code: `connected.py:137–158`, `fairness/demand.py:47–54`. The existing short-spacing scenario changes spacing from 140 m to 65 m **and** demand from 1,920 to 4,800 vehicles/hour, so it cannot separately identify a coordination benefit.

**Discriminating test:** retain the original case and add an explicit corridor-share parameter. Compare low/high corridor shares while holding total volume, turning share, spacing and pedestrian arrivals fixed. Then separately vary spacing or arrival platoons. Do not replace the original result with a favorable new geometry.

### 3. MAPPO is learning local objectives with a centralized value estimator

Both PPO variants receive their junction's queue, external insertion backlog, and pedestrian-pending penalty. The central MAPPO critic sees the whole network but predicts that agent's **local** return. It does not add downstream delay to the objective merely by seeing downstream information. This is a valid specified local-reward variant, not an incorrect definition of centralized training. It does limit incentives to coordinate external effects between junctions.

Relevant code: `fairness/environment.py:171–177`, `ppo_repair/learning.py:118`, `fairness/learning.py:52`, `guided_exploration/learning.py:40`.

**Discriminating test:** after basic PPO learns a hand-checkable policy, compare local versus shared network reward under both IPPO and MAPPO, with identical reward scaling and demand. This is an explicit reward ablation, not a silent algorithm correction. Pending-count penalties optimize a queue-area proxy; they do not mathematically guarantee each crossing's p95 constraint.

### 4. The binary service gate is substantially stricter than an aggregate service comparison

`fairness/analysis.py:35–57` rejects any paired run with even one additional pending request at any crossing at the 2,100-second cutoff, even if every request clears before 2,400 seconds. It also rejects any increased unfinished count or decreased completed count. `ppo_repair/reporting.py:87–94` requires **zero** such failures across all 18 replica-by-demand pairs. The original checkpoint selection additionally sums overlapping vehicle deficits multiple times: one fewer completion and one more unfinished vehicle can contribute three units to the score.

Independent medium-demand audit:

| Configuration | Failed pairs / 18 | Failures caused only by cutoff crossing counts |
|---|---:|---:|
| IPPO original | 17 | 9 |
| IPPO normalized-clock | 15 | 10 |
| IPPO normalized-clock-batch4 | 18 | 12 |
| MAPPO original | 17 | 10 |
| MAPPO normalized-clock | 17 | 5 |
| MAPPO normalized-clock-batch4 | 17 | 11 |

This is an intentionally conservative implementation, but “service failure” must not be read as a collision, a permanently abandoned pedestrian, or proof of worse mean throughput. Finite-cutoff phase alignment alone can trigger it. **Removing the gate would not turn these pilots into successes:** independently recomputed mean vehicle-loss deltas remain positive and all six have worse pooled p95 at all 12 crossings.

**Correction:** display component counts, cutoff versus followup outcomes, and paired service differences beside the strict gate. Preserve the original gate for existing studies. If the team prefers a statistical aggregate non-inferiority definition, specify and justify it before new tests; do not retroactively relabel old failures. Avoid silently using the current overlapping scalar as an unexplained weighted optimization target.

### 5. Training and evaluation have different time regimes

Training resets an empty network every 600 seconds, introduces demand through second 600, and terminates there. Evaluation lasts 2,400 seconds, excludes the first 300 seconds, admits scheduled demand through 2,100, and allows pedestrian followup until 2,400. Slow walkers are evaluation-only in this pilot. This is not test leakage; it is an explicit distribution shift. A 600-second startup controller may not learn the queue states reached after 1,500 seconds of overload.

**Discriminating test:** inspect state/queue/age distributions and evaluate the same policy on both 600- and 2,400-second settings. A later controlled training comparison can use longer continuing rollouts with correct truncation handling. Keep fresh demand seeds and report the unchanged baseline on both horizons.

## Confirmed reproducibility issue, unrelated to poor relative performance

`make_demand` hashes raw XML bytes after `ElementTree.write(path, ...)`. On Windows this writes CRLF newlines; on the cluster it writes LF. The same demand therefore gets different byte hashes across operating systems. For seed 830001, local raw hash starts `f143360a…`, archived cluster hash starts `6f680516…`; normalizing only CRLF to LF exactly reproduces the full cluster hash. This was verified for all six evaluation seeds, with unchanged generator/config source hashes.

**Future fix:** write using a binary stream or explicitly canonicalize the serialized representation and retain both raw-file and semantic hashes. Version this change; preserve old manifests. Within each experiment all compared controllers already have matching demand hashes, so this does not invalidate their relative results or explain the failure.

## Suspicious observations that were checked and are not errors

- Fixed-time vehicle results are identical across medium, burst, and slow-walker settings. For all four local seeds, demand hashes and pedestrian results differ, but fixed signal histories are identical. Vehicles are generated before pedestrians, and the same fixed timing already clears walkers at the configured 0.8 m/s minimum. Identical vehicle outcomes are therefore expected; these are not copied cached results.
- The 54.14-second local actuated reference and the earlier 50.78-second reference use different demand seeds. They are not a before/after algorithm comparison. The 30-minute vehicle cohort is based on scheduled departures in `[300, 2100)`, so roughly 960 scheduled vehicles is consistent with 1,920 vehicles/hour.
- Current repaired and guided evaluation sample categorical actions with controlled streams, matching the training action distribution. The older deterministic argmax failure remains relevant to the original/service-priority studies, but must not be presented as an unfixed defect in repaired pilots.
- The recovered archive contains all 18 completed models' learning logs. Independently recomputed medium p95s, vehicle accounting, demand pairing, and reported point estimates agree. No missing-run or p95 arithmetic error was found in these checks.
- Vehicle time loss excludes departure/insertion waiting; completed counts alone cannot detect that omitted delay. The simulation audit independently verified this in official SUMO TripInfo documentation. Add `departDelay` and all-scheduled accrued-delay diagnostics alongside the required time loss and service counts. Do not silently redefine the old metric.

## Recommended order

First fix and verify any concrete learner or simulation defects found by the other audits. Second require the learner to recover a simple known policy on a small stationary decision task. Third inspect action-level learning in this environment and use free-decision budgets. Fourth test separate reward/cooperation and time-regime hypotheses on development seeds. Preserve the main three-junction proposal question, original four controllers, all failed studies, and untouched final seeds. None of these observations establishes that a policy dominating pedestrian-actuated control at every crossing must exist in the present action space.
