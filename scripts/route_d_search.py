"""Route D: per-junction green-hold timing search (frozen model, NO training).

The scheduling blocker (confirmed by Stage 2 + Route E): the stubborn crossings'
p95 is set by WHEN each junction's PED green recurs, which is governed by how long
vehicle green is held. The policy/rule holds vehicle green up to 30 s; a starved
junction needs PED sooner.

Lever: a per-junction green-hold threshold g_j in [12, 30]. On top of early+vehrule,
once a vehicle green's age >= g_j AND a pedestrian is pending AND PED is legal, force
the phase to end toward PED (request PED). Lower g_j at a junction => PED recurs
sooner => protects that junction's crossing tails, at vehicle cost. This directly
trades vehicle delay for crossing protection per junction.

We search g = (g_C0, g_C1, g_C2) by coordinate ascent on the 12-crossing objective
(maximize crossings-no-worse, tie-break lower vehicle delta) over dev seeds, then
evaluate the best g on all cells. Pure control-parameter search; no RL, no
credit-assignment. Can't regress below early+vehrule (g=30 recovers it).

    python -m scripts.route_d_search --quick
    python -m scripts.route_d_search --workers 5
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics

from src.traffic_control.fairness.analysis import quantile, selection_score
from src.traffic_control.fairness.demand import scenarios
import scripts.demand_service_test as base
import scripts.vehicle_rule_test as vr

shared = base.shared
ROOT = base.ROOT
JUNCTIONS = ("C0", "C1", "C2")


class GreenHoldPolicy(vr.VehicleAxisPolicy):
    """early+vehrule + per-junction green-hold. Forces PED once the current vehicle
    green has run >= hold[junction] seconds and PED is pending+legal. The green-age
    is read from the observation feature 'phase_age' (value*30 = seconds)."""
    def __init__(self, model, hold):
        super().__init__(model)
        self.hold = hold  # {junction: seconds}
        self.age_idx = {j: names.index("phase_age") for j, names in self.features.items()}
        # pending indicator: any local crossing /count > 0 (already in VehicleAxisPolicy indices)

    def distribution(self, junction, observations, masks):
        torch = shared.torch
        dist = super().distribution(junction, observations, masks)
        probs = dist.probs
        obs = torch.as_tensor(observations, dtype=probs.dtype, device=probs.device)
        legal = torch.as_tensor(masks, dtype=torch.bool, device=probs.device)
        age = obs[..., self.age_idx[junction]] * 30.0
        pending = obs[..., self.indices[junction]].gt(0).any(dim=-1)  # self.indices from EarlyServicePolicy
        force = (age + 1e-6 >= self.hold[junction]) & pending & legal[..., 2]
        point = torch.zeros_like(probs); point[..., 2] = 1.0
        return torch.distributions.Categorical(probs=torch.where(force[..., None], point, probs))


def eval_hold(args):
    """Run one (cell, seed, replica, hold) episode; return crossing waits + loss +
    gate. Reuses the demand_service_test episode machinery."""
    cell, seed, replica, hold = args
    p = base.make_protocol(False)
    model, _ = base.load_model(p, "mappo-continue", replica)
    policy = GreenHoldPolicy(model, hold)
    out = shared.episode(Path("runs/route-d"), p, cell, seed, "mappo-greenhold", policy)
    res = out["result"]
    return {"cell": cell, "seed": seed, "replica": replica,
            "loss": res["vehicles"]["completed_mean_time_loss_seconds"],
            "waits": {c: d["waits"] for c, d in res["crossings"].items()},
            "gate": res}


def actuated_ref(cell, seeds):
    p = base.make_protocol(False)
    refs, p95, loss = {}, {}, []
    for s in seeds:
        out = shared.episode(Path("runs/route-d"), p, cell, s, "actuated", None)
        refs[s] = out["result"]
        loss.append(out["result"]["vehicles"]["completed_mean_time_loss_seconds"])
        for c, d in out["result"]["crossings"].items():
            p95.setdefault(c, []).extend(d["waits"])
    return refs, {c: quantile(v) for c, v in p95.items()}, statistics.mean(x for x in loss if x is not None)


def score_hold(pool, cell, seeds, replicas, hold, refs, act_p95, act_loss):
    """Return (crossings_no_worse, vehicle_delta, gate_fail) for a hold vector."""
    jobs = [(cell, s, r, hold) for r in replicas for s in seeds]
    recs = list(pool.map(eval_hold, jobs))
    per_rep = {c: [] for c in act_p95}
    loss, gate_fail = [], 0
    by_rep = {}
    for rec in recs:
        by_rep.setdefault(rec["replica"], []).append(rec)
    for r, rlist in by_rep.items():
        rp = {c: [] for c in act_p95}
        for rec in rlist:
            if rec["loss"] is not None:
                loss.append(rec["loss"])
            if selection_score(rec["gate"], refs[rec["seed"]])[0] != 0:
                gate_fail += 1
            for c in act_p95:
                rp[c].extend(rec["waits"].get(c, []))
        for c in act_p95:
            per_rep[c].append(quantile(rp[c]))
    no_worse = sum(1 for c, a in act_p95.items()
                   if a is not None and [x for x in per_rep[c] if x is not None]
                   and statistics.mean([x for x in per_rep[c] if x is not None]) <= a + 1e-9)
    veh = statistics.mean(x for x in loss if x is not None) - act_loss
    return no_worse, round(veh, 2), gate_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--search-cell", default="vlow-pmedium")  # closest to a win (7/12)
    args = ap.parse_args()
    Path("runs/route-d").mkdir(parents=True, exist_ok=True)
    seeds = [942001] if args.quick else list(range(942001, 942006))
    replicas = [0] if args.quick else [0, 1, 2]
    grid = [30, 24, 18, 12]  # hold candidates per junction; 30 == early+vehrule
    cell = args.search_cell

    refs, act_p95, act_loss = actuated_ref(cell, seeds)
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        # Baseline: hold=30 everywhere (== early+vehrule)
        best = {j: 30 for j in JUNCTIONS}
        best_score = score_hold(pool, cell, seeds, replicas, best, refs, act_p95, act_loss)
        results["start_30_30_30"] = best_score
        print(f"start hold=30/30/30 -> crossings {best_score[0]}/12  Δveh {best_score[1]}  gate_fail {best_score[2]}", flush=True)
        # Coordinate ascent: for each junction, try lower holds, keep if more crossings
        # protected without losing the vehicle gain beyond a tolerance.
        for _ in range(2):  # two sweeps
            for j in JUNCTIONS:
                for g in grid[1:]:
                    cand = dict(best); cand[j] = g
                    key = "_".join(str(cand[k]) for k in JUNCTIONS)
                    if key in results:
                        continue
                    sc = score_hold(pool, cell, seeds, replicas, cand, refs, act_p95, act_loss)
                    results[key] = sc
                    print(f"hold {key} -> crossings {sc[0]}/12  Δveh {sc[1]}  gate_fail {sc[2]}", flush=True)
                    # accept if strictly more crossings protected (vehicle delta allowed to rise
                    # but must stay < 0 to remain a joint-win candidate)
                    better = (sc[0] > best_score[0]) or (sc[0] == best_score[0] and sc[1] < best_score[1])
                    if better and sc[1] < 0:
                        best, best_score = cand, sc
        summary = {"search_cell": cell, "seeds": seeds, "replicas": replicas,
                   "best_hold": best, "best_score": {"crossings_no_worse": best_score[0],
                       "vehicle_delta": best_score[1], "gate_fail": best_score[2]},
                   "all_evaluated": {k: {"crossings": v[0], "veh_delta": v[1], "gate_fail": v[2]}
                                     for k, v in results.items()},
                   "note": "Dev seeds only; reserved 58001-58020 untouched. g=30 == early+vehrule."}
        # Evaluate best hold on all cells for the final comparison
        if not args.quick:
            all_cells = {}
            for c2 in ("vlow-plow", "vlow-pmedium", "vmedium-plow", "vmedium-pmedium"):
                r2, p2, l2 = actuated_ref(c2, seeds)
                sc = score_hold(pool, c2, seeds, replicas, best, r2, p2, l2)
                all_cells[c2] = {"actuated_loss": round(l2, 2), "crossings_no_worse": sc[0],
                                 "vehicle_delta": sc[1], "gate_fail": sc[2]}
                print(f"BEST hold on {c2}: crossings {sc[0]}/12  Δveh {sc[1]}  gate_fail {sc[2]}", flush=True)
            summary["best_hold_all_cells"] = all_cells
    Path("runs/route-d/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ROUTE D COMPLETE. best hold:", best, "=>", best_score)


if __name__ == "__main__":
    main()
