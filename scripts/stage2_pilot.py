"""Stage 2 training pilot: warm-started constrained MAPPO.

Warm-starts from continued-MAPPO checkpoints and continues training with
ConstrainedRepairLearner (per-crossing Lagrangian). Factors: horizon (600 vs 2400)
and optional vehicle-axis envelope. Development seeds only; reserved 58001-58020
untouched. Resumable; saves final.pt that loads through the Stage-1 eval path.

    python -m scripts.stage2_pilot --quick --output runs/stage2-smoke
    python -m scripts.stage2_pilot --horizon 2400 --episodes 32 --replicas 3 \
        --output runs/stage2-h2400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from src.traffic_control.fairness.analysis import summarise, selection_score
from src.traffic_control.fairness.demand import scenarios
from src.traffic_control.fairness.environment import run as env_run
from src.traffic_control.fairness.pipeline import atomic_json
from src.traffic_control.ppo_repair.learning import RepairNetworks
import scripts.demand_service_test as dst
from scripts.stage2_constrained import ConstrainedRepairLearner

shared = dst.shared
torch = shared.torch
ROOT = dst.ROOT

TRAIN_SCENARIOS = ("vmedium-pmedium", "vmedium-plow", "vlow-pmedium", "vlow-plow")
VAL_SCENARIOS = ("vmedium-pmedium", "vmedium-plow")
EVAL_SCENARIOS = ("vlow-plow", "vlow-pmedium", "vmedium-plow", "vmedium-pmedium")


def cfg_for(horizon, scenario, quick, training=False):
    cfg = dict(scenarios()[scenario])
    if quick:
        cfg.update(end_seconds=180, warmup_seconds=0, measurement_end_seconds=120,
                   demand_end_seconds=90)
    elif training:
        cfg.update(end_seconds=horizon, warmup_seconds=0,
                   measurement_end_seconds=horizon, demand_end_seconds=horizon)
    # else: evaluation uses the scenario/config defaults (2400 / 300 / 2100).
    return cfg


def rollout(root, model, scenario, seed, cfg, train):
    """One episode; returns (summary_result, trace, events, raw_metrics)."""
    streams = {t: random.Random(900000 + seed + i * 1000003)
               for i, t in enumerate(sorted(model.features))}
    trace = []

    def policy(tls, obs, glob, mask):
        legal = [i for i, ok in enumerate(mask) if ok]
        if len(legal) == 1:
            return legal[0]
        with torch.no_grad():
            probs = model.distribution(tls, obs, mask).probs.tolist()
        if train:
            return streams[tls].choices(legal, weights=[probs[i] for i in legal])[0]
        return max(legal, key=lambda i: probs[i])  # deterministic eval

    folder = Path(root) / f"ep-{scenario}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    metrics = env_run("policy", seed, folder, overrides=cfg, policy=policy, trace_sink=trace)
    events = json.loads((folder / "events.json").read_text())
    import shutil
    shutil.rmtree(folder, ignore_errors=True)
    return summarise(metrics, events), trace, events, metrics


def thresholds_from_actuated(root, horizon, quick):
    """Per-crossing threshold = actuated p95 on the primary validation cell.
    Falls back to a fixed 20 s if a crossing has too few samples."""
    cfg = cfg_for(horizon, "vmedium-pmedium", quick)
    folder = Path(root) / "thr-actuated"
    folder.mkdir(parents=True, exist_ok=True)
    trace = []
    metrics = env_run("actuated", 941001, folder, overrides=cfg, trace_sink=trace)
    events = json.loads((folder / "events.json").read_text())
    res = summarise(metrics, events)
    import shutil
    shutil.rmtree(folder, ignore_errors=True)
    thr = {}
    for crossing, data in res["crossings"].items():
        thr[crossing] = data["p95"] if data["p95"] is not None else 20.0
    return thr


def train_replica(root, protocol, replica):
    root = Path(root)
    label = f"constrained-mappo-r{replica}"
    directory = root / label
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        return label

    parent, parent_sha = dst.load_model(dst.make_protocol(protocol["quick"]),
                                         "mappo-continue", replica)
    model = RepairNetworks.from_snapshot(parent.snapshot())
    model.observation_schema = "legacy-v2"
    torch.manual_seed(940001 + replica)
    learner = ConstrainedRepairLearner(model, weight=3, thresholds=protocol["thresholds"],
                                        cost_seed=940000 + replica)

    best = (float("inf"),) * 3
    for i in range(protocol["episodes"]):
        scenario = TRAIN_SCENARIOS[i % len(TRAIN_SCENARIOS)]
        seed = 940001 + replica * 1000 + i
        cfg = cfg_for(protocol["horizon"], scenario, protocol["quick"], training=True)
        _, trace, events, _ = rollout(directory, model, scenario, seed, cfg, train=True)
        stats = learner.update([{"trace": trace, "horizon": cfg["end_seconds"]}], events=events)
        atomic_json(directory / f"episode-{i+1}.json",
                    {"episode": i + 1, "scenario": scenario, "seed": seed,
                     "multipliers": stats[sorted(stats)[0]]["multipliers"]})
        # Validation + lexicographic selection every validate_every episodes.
        if (i + 1) % protocol["validate_every"] == 0 or i + 1 == protocol["episodes"]:
            import statistics
            scores = []
            for scen in VAL_SCENARIOS:
                for vseed in protocol["validation_seeds"]:
                    out, _, _, _ = rollout(directory, model, scen, vseed,
                                           cfg_for(protocol["horizon"], scen, protocol["quick"]), train=False)
                    scores.append(selection_score(out, baseline_actuated(directory, protocol, scen, vseed)))
            score = tuple(statistics.mean(s[j] for s in scores) for j in range(3))
            if score < best:
                best = score
                shared.atomic_torch(directory / "best.pt", {"model": model.snapshot()})
                atomic_json(directory / "selected.json", {"episode": i + 1, "score": list(best)})

    # Final checkpoint = selected best; loadable through from_snapshot / load_model path.
    best_snapshot = (torch.load(directory / "best.pt", map_location="cpu", weights_only=True)["model"]
                     if (directory / "best.pt").exists() else model.snapshot())
    selected = RepairNetworks.from_snapshot(best_snapshot)
    selected.observation_schema = "legacy-v2"
    shared.atomic_torch(directory / "final.pt",
                        dict(fingerprint=protocol["fingerprint"], label=label,
                             parent_sha256=parent_sha, episodes=protocol["episodes"],
                             model=selected.snapshot()))
    atomic_json(directory / "complete.json", {"episodes": protocol["episodes"]})
    return label


_ACT_CACHE = {}


def baseline_actuated(root, protocol, scenario, seed):
    key = (scenario, seed)
    if key in _ACT_CACHE:
        return _ACT_CACHE[key]
    cfg = cfg_for(protocol["horizon"], scenario, protocol["quick"])
    folder = Path(root).parent / f"act-{scenario}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    trace = []
    metrics = env_run("actuated", seed, folder, overrides=cfg, trace_sink=trace)
    events = json.loads((folder / "events.json").read_text())
    import shutil
    shutil.rmtree(folder, ignore_errors=True)
    _ACT_CACHE[key] = summarise(metrics, events)
    return _ACT_CACHE[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=2400)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)

    thr = thresholds_from_actuated(root, args.horizon, args.quick)
    protocol = {"schema": "stage2-constrained-v1", "quick": args.quick,
                "horizon": 180 if args.quick else args.horizon,
                "episodes": 2 if args.quick else args.episodes,
                "replicas": 1 if args.quick else args.replicas,
                "validate_every": 2 if args.quick else 8,
                "validation_seeds": [941001] if args.quick else [941001, 941002],
                "thresholds": thr,
                "reserved_final_seeds_not_used": list(range(58001, 58021)),
                "seed_policy": "training 940001+, validation 941001-2; reserved 58001-58020 untouched"}
    import hashlib
    protocol["fingerprint"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    atomic_json(root / "protocol.json", protocol)

    labels = []
    for r in range(protocol["replicas"]):
        labels.append(train_replica(root, protocol, r))
        print("trained", labels[-1], flush=True)
    atomic_json(root / "status.json", {"state": "complete", "labels": labels})
    print("Stage 2 training complete:", labels)


if __name__ == "__main__":
    main()
