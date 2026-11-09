"""Stage 1.5: critic-fitting gate (uses the repair module's own machinery).

The continued-MAPPO checkpoints are RepairNetworks of the 'normalized-clock'
variant: they ALREADY include value normalization (RunningValueScale) and a
time-to-horizon clock feature. So the gate question is sharpened:

  Does this already-normalized critic predict held-out GAE returns well?
    - If held-out explained variance is decent, value learning is NOT the ceiling;
      Stage 2 (training the actor within a pedestrian envelope) is viable.
    - If it is still ~0, the value target itself is hard to fit here, and more
      training is unlikely to help -> write up Tier 1.

Also fits a fresh critic from scratch on the same (inputs -> returns) as a floor,
to separate 'the checkpoint's critic is stale' from 'the target is unlearnable'.

No SUMO training; frozen rollouts only; development seeds (2200001+); reserved
58001-58020 untouched.

    python -m scripts.critic_gate --output runs/critic-gate
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import statistics

from src.traffic_control.fairness.environment import run
from src.traffic_control.fairness.learning import advantages
from src.traffic_control.fairness.demand import scenarios
from src.traffic_control.ppo_repair.learning import prepare_batch, prediction_stats
import scripts.demand_service_test as dst

shared = dst.shared
torch = shared.torch


def rollout(root, model, seed, scenario="vmedium-pmedium", horizon=600):
    cfg = {**scenarios()[scenario], "end_seconds": horizon, "warmup_seconds": 0,
           "measurement_end_seconds": horizon, "demand_end_seconds": horizon}
    streams = {t: random.Random(900000 + seed + i * 1000003)
               for i, t in enumerate(sorted(model.features))}
    trace = []

    def policy(tls, obs, glob, mask):
        legal = [i for i, ok in enumerate(mask) if ok]
        if len(legal) == 1:
            return legal[0]
        with torch.no_grad():
            probs = model.distribution(tls, obs, mask).probs.tolist()
        return streams[tls].choices(legal, weights=[probs[i] for i in legal])[0]

    (Path(root) / f"t{seed}").mkdir(parents=True, exist_ok=True)
    run("policy", seed, Path(root) / f"t{seed}", overrides=cfg, policy=policy, trace_sink=trace)
    return {"trace": trace, "horizon": horizon}


def fit_fresh_normalized(inputs, returns, updates=600, lr=3e-4):
    """Fresh MLP critic on z-scored targets (denormalized to report). Isolates
    'is the target learnable at all' from the checkpoint's own critic quality."""
    from src.traffic_control.fairness.networks import mlp
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        net = mlp(inputs.shape[-1], 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    mu, sd = returns.mean(), returns.std(unbiased=False).clamp_min(1e-6)
    z = (returns - mu) / sd
    for _ in range(updates):
        loss = (net(inputs).squeeze(-1) - z).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(inputs).squeeze(-1) * sd + mu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=dst.ROOT / "runs/critic-gate")
    ap.add_argument("--replica", type=int, default=0)
    ap.add_argument("--train-seeds", type=int, default=6)
    ap.add_argument("--test-seeds", type=int, default=3)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    p = dst.make_protocol(quick=False)
    model, _ = dst.load_model(p, "mappo-continue", args.replica)
    base = 2200001  # fresh development seeds; not training/eval/reserved
    train_seeds = list(range(base, base + args.train_seeds))
    test_seeds = list(range(base + 100, base + 100 + args.test_seeds))

    print(f"collecting {len(train_seeds)} train + {len(test_seeds)} test rollouts (continue-r{args.replica})...")
    train_roll = [rollout(args.output, model, s) for s in train_seeds]
    test_roll = [rollout(args.output, model, s) for s in test_seeds]

    results = {}
    for tls in sorted(model.features):
        tr = prepare_batch(model, tls, train_roll)
        te = prepare_batch(model, tls, test_roll)
        # (a) the checkpoint's own (already value-normalized) critic on held-out returns
        with torch.no_grad():
            model_pred = model.prediction(tls, te["inputs"])
        ckpt = prediction_stats(te["returns"], model_pred)
        # (b) a fresh critic fit on train inputs->returns, evaluated on held-out
        fresh_net_pred = None
        fresh = None
        if tr["inputs"].shape[0] >= 20:
            fit = fit_fresh_normalized(tr["inputs"], tr["returns"])  # fit on train
            # refit-predict on test by training then predicting test inputs:
            from src.traffic_control.fairness.networks import mlp
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(12345); net = mlp(tr["inputs"].shape[-1], 1)
            opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
            mu, sd = tr["returns"].mean(), tr["returns"].std(unbiased=False).clamp_min(1e-6)
            z = (tr["returns"] - mu) / sd
            for _ in range(600):
                loss = (net(tr["inputs"]).squeeze(-1) - z).square().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                fresh_pred = net(te["inputs"]).squeeze(-1) * sd + mu
            fresh = prediction_stats(te["returns"], fresh_pred)
        results[tls] = {
            "n_train": int(tr["inputs"].shape[0]), "n_test": int(te["inputs"].shape[0]),
            "checkpoint_critic_ev": ckpt["explained_variance"], "checkpoint_rmse": round(ckpt["rmse"], 3),
            "fresh_critic_ev": fresh["explained_variance"] if fresh else None,
            "target_mean": round(ckpt["target_mean"], 2)}
        ce = ckpt["explained_variance"]; fe = fresh["explained_variance"] if fresh else None
        print(f"  {tls}: checkpoint EV={ce:+.4f}" + (f"  fresh-fit EV={fe:+.4f}" if fe is not None else ""))

    ev_ckpt = [r["checkpoint_critic_ev"] for r in results.values() if r["checkpoint_critic_ev"] is not None]
    ev_fresh = [r["fresh_critic_ev"] for r in results.values() if r["fresh_critic_ev"] is not None]
    mck = statistics.mean(ev_ckpt) if ev_ckpt else 0.0
    mfr = statistics.mean(ev_fresh) if ev_fresh else 0.0
    if mfr > 0.3:
        verdict = "VALUE TARGET IS LEARNABLE (fresh critic fits) -> Stage 2 viable; checkpoint critic was under-fit"
    elif mck > 0.3:
        verdict = "CHECKPOINT CRITIC ALREADY FITS -> value is not the ceiling; Stage 2 bottleneck is the actor/reward"
    else:
        verdict = "NEITHER CRITIC FITS HELD-OUT RETURNS -> value learning is the ceiling; write up Tier 1"
    summary = {"replica": args.replica, "train_seeds": train_seeds, "test_seeds": test_seeds,
               "architecture": "RepairNetworks normalized-clock (value normalization + time-to-horizon)",
               "per_junction": results, "mean_checkpoint_ev": round(mck, 4),
               "mean_fresh_critic_ev": round(mfr, 4), "verdict": verdict,
               "note": "Held-out GAE-return regression on frozen rollouts; development seeds 2200001+; reserved 58001-58020 untouched."}
    (args.output / "critic_gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nMEAN held-out EV  checkpoint={mck:+.4f}  fresh-fit={mfr:+.4f}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
