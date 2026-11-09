# Deep planning: routes to the joint win

_2026-11-05. Grounded in measured evidence, not inference. The goal: one controller
that, vs. actuated on matched demand, has vehicle Δ<0 AND p95 no-worse at all 12
crossings AND a clean service gate — ideally with statistical support on reserved
seeds._

## What the evidence actually says (measured, not guessed)

- **Stage 1 (rule):** `early+vehrule` = vehicle Δ −0.47 s with **6/12** at med/med;
  −0.92 s with **7/12** at low/med. The current frontier. A myopic queue rule.
- **Stage 1.5 (critic):** held-out EV ~0.49. **Value learning is NOT the limiter.**
- **Stage 2 (constrained RL) failed, and we now know exactly why:**
  - λ trajectory: 0.015 → 0.285 at ep32, **still rising linearly, no plateau.** The
    constraint was cut off mid-ascent; it needed to be several× stronger.
  - Policy behaviour: serves PED only **35%** of the time when PED is legal
    (mean P(PED|legal)=0.39). A vehicle-biased actor the weak λ never overcame.
  - Vehicle numbers were excellent (−4 to −14 s); pedestrians collapsed (0–1/12).
  **Conclusion: not a dead approach — an under-powered, under-trained one.**

This changes the planning posture: the constrained-RL route is **unfinished, not
refuted.** Several levers were never pushed.

## The solution space — seven distinct routes

Grouped by how far they stray from the current code. For each: idea, the RL/traffic
concept, why it could work, cost, and honest odds of reaching ≥1 cell joint-win.

### A. Finish the constrained-RL route (Stage 2b) — push the levers we left slack
The direct fix for the measured failure. Four knobs, all under-set in Stage 2:
1. **Stronger dual ascent.** Step 0.05 → 0.3–0.5, or initialise λ warm (≈1.0) so the
   constraint bites from episode 1 instead of ramping for 32 episodes.
2. **More episodes.** 32 → 128+. λ was still climbing; give it room to converge.
3. **Sampled or entropy-floored eval**, not argmax — argmax re-triggers the E06
   collapse; the policy had 39% PED mass that argmax throws away.
4. **Train from a pedestrian-biased or neutral start**, not the vehicle-maximising
   continued-MAPPO (warm-start basin bias).
- Concept: Lagrangian constrained MDP, properly tuned. Odds: **moderate** — this is
  the textbook fix for "constraint didn't bind." Cost: hours–1 day (retrain).

### B. Reward reshaping instead of constraint — target the p95 directly
The reward is `−(queue + w·pending)`: a *mean* pedestrian signal. The objective is a
*tail* (p95). Replace/augment with a **tail-sensitive pedestrian term**: penalise the
single oldest pending request per crossing (max, not sum), or a soft-max over ages.
- Concept: reward design that matches the metric (tail vs. mean). The hybrid
  `early+vehrule` already proves early PED service protects tails; this teaches the
  actor to do it. Odds: **moderate**; risk: reward shaping confounds (must ablate).
  Cost: hours (reward edit + retrain). Composable with A.

### C. Residual/constrained learning ON TOP of the rule — learn the correction only
Instead of learning the whole policy, **freeze `early+vehrule` as the base** and let
RL learn a small, bounded deviation (serve PED slightly earlier/later at specific
crossings). The base already gives 6–7/12; the actor only needs to fix the last
5–6 crossings' scheduling.
- Concept: residual RL (the literature the repo already cites) + action envelope.
  Why it could work: tiny learning problem, starts from the best known controller,
  can't regress below it if the residual is bounded. Odds: **moderate-high** — this
  is the most targeted route and the hardest to make worse. Cost: 1 day (new wrapper
  + bounded-residual learner).

### D. Per-crossing offset/timing search — no RL at all
The residual blocker is *scheduling*: when each crossing's green recurs. Treat it as
a small **optimisation over per-junction phase offsets / PED insertion timing** on
top of the rule, searched directly (grid/CMA-ES/coordinate ascent) against the
12-crossing p95 objective on dev seeds.
- Concept: direct policy search / control-parameter tuning (not deep RL). Why: the
  decision space is tiny (a few offsets per junction), the objective is exactly the
  metric, no credit-assignment problem. Odds: **moderate-high** for a *cell*, and
  it's the most interpretable. Cost: hours (search harness). Strong fallback.

### E. Stronger vehicle-axis rule + pedestrian-tail rule combined — pure heuristic
Extend Stage 1: add a **per-crossing max-wait deadline** to the rule (force PED at a
crossing whose oldest request exceeds its actuated p95), composed with the larger-
queue vehicle rule. This is `service_priority`'s deadline idea unified with
`early+vehrule`.
- Concept: two coupled service rules, no learning. Why: directly attacks the
  unprotected crossings; `service_priority` already showed deadlines protect tails.
  Odds: **moderate-high** for ≥1 cell; honest caveat — a rule win, not a learning
  win. Cost: hours. Lowest risk of all routes.

### F. Change the action space — give the agent the real lever
Today the actor picks EW/NS/PED phase. It cannot choose *PED duration* or *which
crossing* within a junction. Expand the action to include a short/long PED or a
per-crossing priority, so the policy can actually shape tails.
- Concept: action-space design. Why: the current action may be too coarse to express
  the fix (you can't protect crossing c3 specifically if PED serves all 4 together).
  Odds: **uncertain** — could unlock it or just enlarge the search. Cost: days
  (touches environment + networks + checkpoint schema). Highest-risk, highest-ceiling.

### G. Accept the frontier — make the trade-off itself the result
If A–F don't yield a clean 12/12, the deliverable is the **frontier + mechanism**:
"a queue rule reaches 6–7/12 with a vehicle gain; constrained RL at feasible budget
reverts to vehicle-maximisation; the residual is per-crossing scheduling." This is
already true, documented, and publishable.
- Odds of being the final story: **high.** Cost: writing only.

## Recommended sequence (cheap→expensive, each gated)

```
1. E  (unified rule: larger-queue + per-crossing deadline)  — hours, lowest risk
      └ if a cell hits 12/12 + gate clean + vehicle Δ<0 → Stage 3 (reserved seeds)
2. D  (offset/timing search on top of the rule)             — hours, interpretable
      └ squeezes the last crossings a pure rule can't
3. A+B (finish constrained RL: strong λ, 128 ep, tail reward, sampled eval)
      └ the "learning" result; moderate odds, 1 day
4. C  (residual RL on the rule) if A+B underperforms the rule — 1 day
5. G  (write up frontier+mechanism) — always, in parallel
```

Rationale for the order: **E and D are the highest-probability, lowest-cost routes
to an actual 12/12 in at least one cell**, and they attack the *measured* blocker
(scheduling of specific crossings) directly. A+B is the scientifically valuable
"does learning help" arm, now properly powered — worth running because Stage 2's
failure was under-powering, not the idea. C is the elegant fallback that can't
regress below the rule. G is the guaranteed deliverable.

## What "reaching the goal" realistically looks like

- **Most likely win:** a *rule-based* 12/12 in one or two low-demand cells (E/D),
  honestly framed as "a carefully designed service rule achieves the joint target in
  easy conditions; learning does not beat it." That is a legitimate joint win on the
  stated objective.
- **Stretch:** constrained/residual RL (A+B/C) matches or beats the rule — the
  stronger "learning helps" claim. Lower odds.
- **Hardest:** 12/12 at medium/medium with statistical support on reserved seeds.
  Unlikely; the trade-off is tightest there.

## Guardrails (unchanged)

Every route: masks only remove/redirect legal actions; never shorten safety phases;
one controller must clear all three conditions together; dev seeds only until a
frozen candidate; reserved 58001–58020 used once, for Stage 3; report service counts
+ all 12 tails + uncertainty; distinguish descriptive vs statistical pass.

## Decision for this session

Start with **E** (unified rule) now — it is hours of work, attacks the measured
blocker, and either lands a cell (→ Stage 3) or tightens the negative result. Run
**D** next if E leaves 1–3 crossings short. Keep **A+B** as the funded "did learning
help" experiment. Write **G** in parallel regardless.
