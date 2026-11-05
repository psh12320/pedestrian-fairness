"""Parallel evaluation of Stage 2 constrained-MAPPO vs actuated and the frontier.

Runs each (cell, seed, controller) as an independent job across workers, then pools
per-crossing p95 (seeds within replica, averaged over replicas) exactly like the
Stage-1 report. Writes runs/stage2-h2400/eval_summary.json.

    python -m scripts.stage2_eval --workers 5
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import random
import shutil
import statistics
from pathlib import Path

import torch
from src.traffic_control.ppo_repair.learning import RepairNetworks
from src.traffic_control.fairness.environment import run as env_run
from src.traffic_control.fairness.demand import scenarios
from src.traffic_control.fairness.analysis import summarise, selection_score, quantile

CELLS = ["vlow-plow", "vlow-pmedium", "vmedium-plow", "vmedium-pmedium"]
SEEDS = list(range(942001, 942011))   # 10 dev eval seeds, matching Stage 1
REPLICAS = [0, 1, 2]
RUN_ROOT = Path("runs/stage2-h2400")


def load_model(replica):
    s = torch.load(RUN_ROOT / f"constrained-mappo-r{replica}/final.pt",
                   map_location="cpu", weights_only=True)
    m = RepairNetworks.from_snapshot(s["model"])
    m.observation_schema = "legacy-v2"
    m.eval()
    return m


def one_job(args):
    cell, seed, replica = args          # replica=None => actuated reference
    cfg = dict(scenarios()[cell])        # full 2400s eval defaults
    folder = RUN_ROOT / f"_e-{cell}-{seed}-{replica}"
    folder.mkdir(parents=True, exist_ok=True)
    trace = []
    if replica is None:
        met = env_run("actuated", seed, folder, overrides=cfg, trace_sink=trace)
    else:
        m = load_model(replica)
        rng = {t: random.Random(900000 + seed + i * 1000003) for i, t in enumerate(sorted(m.features))}

        def pol(tls, o, g, mask):
            legal = [i for i, ok in enumerate(mask) if ok]
            if len(legal) == 1:
                return legal[0]
            with torch.no_grad():
                pr = m.distribution(tls, o, mask).probs.tolist()
            return max(legal, key=lambda i: pr[i])
        met = env_run("policy", seed, folder, overrides=cfg, policy=pol, trace_sink=trace)
    events = json.loads((folder / "events.json").read_text())
    res = summarise(met, events)
    shutil.rmtree(folder, ignore_errors=True)
    waits = {c: d["waits"] for c, d in res["crossings"].items()}
    return {"cell": cell, "seed": seed, "replica": replica,
            "loss": res["vehicles"]["completed_mean_time_loss_seconds"], "waits": waits,
            "summary_for_gate": {"vehicles": res["vehicles"], "persons": res.get("persons"),
                                 "crossings": {c: {k: d[k] for k in ("p95", "pending_at_cutoff",
                                     "unresolved_after_followup", "unknown_cohort_entries", "lost")}
                                               for c, d in res["crossings"].items()}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    jobs = [(c, s, r) for c in CELLS for s in SEEDS for r in [None] + REPLICAS]
    records = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for rec in pool.map(one_job, jobs):
            records[(rec["cell"], rec["seed"], rec["replica"])] = rec
            print("done", rec["cell"], rec["seed"], rec["replica"], flush=True)

    out = {}
    for cell in CELLS:
        refs = {s: records[(cell, s, None)] for s in SEEDS}
        crossings = list(refs[SEEDS[0]]["waits"].keys())
        act_p95 = {c: quantile([w for s in SEEDS for w in refs[s]["waits"][c]]) for c in crossings}
        act_loss = statistics.mean(refs[s]["loss"] for s in SEEDS if refs[s]["loss"] is not None)
        per_rep = {c: [] for c in crossings}
        model_loss, gate_fail = [], 0
        for r in REPLICAS:
            rp = {c: [] for c in crossings}
            for s in SEEDS:
                rec = records[(cell, s, r)]
                if rec["loss"] is not None:
                    model_loss.append(rec["loss"])
                if selection_score(rec["summary_for_gate"], refs[s]["summary_for_gate"])[0] != 0:
                    gate_fail += 1
                for c in crossings:
                    rp[c].extend(rec["waits"][c])
            for c in crossings:
                per_rep[c].append(quantile(rp[c]))
        no_worse = sum(1 for c in crossings
                       if act_p95[c] is not None
                       and [x for x in per_rep[c] if x is not None]
                       and statistics.mean([x for x in per_rep[c] if x is not None]) <= act_p95[c] + 1e-9)
        mean_loss = statistics.mean(x for x in model_loss if x is not None)
        out[cell] = {"actuated_loss": round(act_loss, 2), "constrained_loss": round(mean_loss, 2),
                     "vehicle_delta": round(mean_loss - act_loss, 2),
                     "crossings_no_worse": no_worse, "gate_fail_runs": gate_fail,
                     "total_model_runs": len(SEEDS) * len(REPLICAS)}
        print(f"{cell} | actuated {act_loss:.2f} | constrained {mean_loss:.2f} | "
              f"delta_veh {mean_loss-act_loss:+.2f} | crossings_no_worse {no_worse}/12 | "
              f"gate_fail {gate_fail}/{len(SEEDS)*len(REPLICAS)}", flush=True)
    (RUN_ROOT / "eval_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("EVAL COMPLETE")


if __name__ == "__main__":
    main()
