"""Resumable training/evaluation. --quick is an engineering check, not research.

python -m src.traffic_control.fairness.pipeline --quick --workers 1
python -m src.traffic_control.fairness.pipeline --workers 8 --episodes 512
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time
import platform
import subprocess
import shutil

from .analysis import summarise, selection_score, quantile, primary_inference
from .demand import scenarios
from .environment import run
from ..scenario import ROOT, executable

VARIANTS = [(m, o) for m in ("ippo", "mappo") for o in ("vehicle", "pedestrian", "constrained")]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def baseline_job(job):
    root, protocol, seed, purpose = job
    root = Path(root)
    path = root / f"{purpose}-{seed}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as stream:
            saved = json.load(stream)
        if saved["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Cached preparation data belongs to another source/protocol")
        return saved
    result, metrics, _, _ = episode("actuated", seed, protocol["evaluation_config"], root)
    saved = {"fingerprint": protocol["fingerprint"], "seed": seed, "result": result,
             "features": metrics["observation_features"]}
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt") as stream:
        json.dump(saved, stream)
    temporary.replace(path)
    return saved


def episode(controller, seed, cfg, root, model=None, learner=None, collect=False):
    from .environment import run
    traces = [] if collect else None
    if shutil.disk_usage(root).free < 256 * 1024 ** 2:
        raise OSError("Less than 256 MiB free. Stop rather than exhaust storage.")
    callback = learner.policy if learner else (
        (lambda tls, obs, state, mask: model.act(tls, obs, mask, True)[0]) if model else None)
    with tempfile.TemporaryDirectory(prefix="rollout-", dir=root) as directory:
        folder = Path(directory)
        try:
            metrics = run(controller, seed, folder, 3, cfg, callback, traces)
            events = json.loads((folder / "events.json").read_text())
        except Exception as exc:
            failure = json.loads((folder / "failure.json").read_text()) if (folder / "failure.json").exists() else {"error": repr(exc)}
            failure["seed"] = seed
            failures = root / "failures"
            failures.mkdir(exist_ok=True)
            atomic_json(failures / f"{controller}-{seed}-{time.time_ns()}.json", failure)
            raise
    return summarise(metrics, events), metrics, events, traces


def train_job(job):
    import torch
    from .learning import Learner
    from .networks import JunctionNetworks
    root, protocol, method, objective, replica = job
    root = Path(root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(25001 + replica)
    label = f"{method}-{objective}-r{replica}"
    directory = root / label
    directory.mkdir(exist_ok=True)
    model = JunctionNetworks(method, protocol["features"], 25001 + replica)
    learner = Learner(model, 0 if objective == "vehicle" else 3,
                      protocol["thresholds"] if objective == "constrained" else None)
    latest, best = directory / "latest.pt", directory / "best.pt"
    start, score = 0, (float("inf"),) * 3
    if latest.exists():
        data = torch.load(latest, weights_only=True)
        if data["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Training source/protocol changed; use a new output folder")
        model.load_state_dict(data["weights"])
        learner.optimizer.load_state_dict(data["optimizer"])
        learner.multipliers = data["multipliers"]
        start, score = data["episode"], tuple(data["best_score"])
        torch.set_rng_state(data["rng"])
    validation = json.loads((root / "validation_baselines.json").read_text())
    until = min(protocol["episodes"], protocol.get("train_until", protocol["episodes"]))
    for i in range(start, until):
        episode_started = time.perf_counter()
        cfg = dict(protocol["training_config"])
        factor = (.75, 1., 1.25)[i % 3]
        cfg.update(vehicle_rate_per_hour=1920 * factor, pedestrian_rate_per_hour=1200 * factor)
        if protocol["quick"]:
            cfg.update(vehicle_rate_per_hour=720, pedestrian_rate_per_hour=720)
        outcome, _, events, trace = episode("policy", 100000 + replica * 10000 + i, cfg,
                                            directory, learner=learner, collect=True)
        rollout_seconds = time.perf_counter() - episode_started
        update_started = time.perf_counter()
        update = learner.update(trace, events)
        update_seconds = time.perf_counter() - update_started
        candidate = None
        if (i + 1) % protocol["validate_every"] == 0 or i + 1 == protocol["episodes"]:
            scores = []
            for seed in protocol["validation_seeds"]:
                result, _, _, _ = episode("policy", seed, protocol["evaluation_config"], directory, model=model)
                scores.append(selection_score(result, validation[str(seed)], objective != "vehicle"))
            candidate = tuple(statistics.mean(s[j] for s in scores) for j in range(3))
            if candidate < score:
                score = candidate
                model.save(best.with_suffix(".tmp"))
                best.with_suffix(".tmp").replace(best)
                atomic_json(directory / "selected.json", {"episode": i + 1, "score": score,
                            "feasible_on_validation": score[0] == 0 and score[1] == 0})
        with (directory / "training.jsonl").open("a") as stream:
            stream.write(json.dumps({"episode": i + 1, "update": update,
                                    "rollout_wall_seconds": rollout_seconds, "update_wall_seconds": update_seconds,
                                    "vehicle_mean_time_loss": outcome["vehicles"]["completed_mean_time_loss_seconds"],
                                    "crossing_p95": {c: v["p95"] for c, v in outcome["crossings"].items()},
                                    "validation_score": candidate,
                                    "multipliers": learner.multipliers}) + "\n")
        torch.save({"fingerprint": protocol["fingerprint"], "episode": i + 1,
                    "weights": model.state_dict(), "optimizer": learner.optimizer.state_dict(),
                    "multipliers": learner.multipliers, "best_score": score,
                    "rng": torch.get_rng_state()}, latest.with_suffix(".tmp"))
        latest.with_suffix(".tmp").replace(latest)
    if until == protocol["episodes"]:
        atomic_json(directory / "complete.json", {"episodes": protocol["episodes"], "fingerprint": protocol["fingerprint"]})
    return label


def evaluation_job(job):
    import torch
    from .networks import JunctionNetworks
    root, protocol, scenario, seed = job
    root = Path(root)
    destination = root / "evaluation" / f"{scenario}-{seed}.json.gz"
    if destination.exists():
        with gzip.open(destination, "rt") as stream:
            if json.load(stream)["fingerprint"] != protocol["fingerprint"]:
                raise ValueError("Evaluation cache belongs to another protocol")
        return str(destination)
    torch.set_num_threads(1)
    cfg = {**protocol["evaluation_config"], **scenarios()[scenario]}
    if protocol["quick"]:
        cfg.update(vehicle_rate_per_hour=720, pedestrian_rate_per_hour=720)
    partial = destination.with_name(destination.name + ".partial")
    records = []
    if partial.exists():
        with gzip.open(partial, "rt") as stream:
            saved = json.load(stream)
        if saved["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Partial evaluation belongs to another protocol")
        records = saved["records"]
    def cache():
        temporary = partial.with_suffix(".tmp")
        with gzip.open(temporary, "wt") as stream:
            json.dump({"fingerprint": protocol["fingerprint"], "records": records}, stream)
        temporary.replace(partial)
    for label in ("fixed", "actuated", "pressure"):
        if any(r["controller"] == label for r in records):
            continue
        result, _, _, _ = episode(label, seed, cfg, root)
        records.append({"controller": label, "replica": None, "result": result})
        cache()
    for method, objective in VARIANTS:
        for replica in range(protocol["replicas"]):
            label = f"{method}-{objective}"
            if any(r["controller"] == label and r["replica"] == replica for r in records):
                continue
            completion = root / f"{label}-r{replica}" / "complete.json"
            if not completion.exists():
                raise ValueError(f"Training is incomplete for {label} replica {replica}")
            model = JunctionNetworks.load(root / f"{label}-r{replica}" / "best.pt", protocol["features"])
            result, _, _, _ = episode("policy", seed, cfg, root, model=model)
            records.append({"controller": label, "replica": replica, "result": result})
            cache()
    if len({r["result"]["demand_sha256"] for r in records}) != 1:
        raise AssertionError("Mismatched evaluation demand")
    data = {"fingerprint": protocol["fingerprint"], "scenario": scenario, "seed": seed, "records": records}
    temporary = destination.with_suffix(".tmp")
    with gzip.open(temporary, "wt") as stream:
        json.dump(data, stream)
    temporary.replace(destination)
    partial.unlink()
    return str(destination)


def report(root, protocol):
    groups = []
    for path in sorted((root / "evaluation").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            groups.append(json.load(stream))
    rows = []
    for scenario in protocol["scenarios"]:
        selected = [g for g in groups if g["scenario"] == scenario]
        if len(selected) != len(protocol["test_seeds"]):
            raise ValueError("Incomplete evaluation; cannot build final tables")
        for method, objective in VARIANTS:
            label = f"{method}-{objective}"
            comparisons = []
            for g in selected:
                baseline = next(r["result"] for r in g["records"] if r["controller"] == "actuated")
                for r in g["records"]:
                    if r["controller"] == label:
                        score = selection_score(r["result"], baseline)
                        comparisons.append({"seed": g["seed"], "replica": r["replica"],
                            "vehicle_difference": score[2] - (baseline["vehicles"]["completed_mean_time_loss_seconds"] or 0),
                            "service_violation": score[0], "worst_positive_crossing_difference": score[1]})
            rows.append({"scenario": scenario, "controller": label, "paired_runs": comparisons})
    atomic_json(root / "comparisons.json", rows)
    primary = primary_inference(groups)
    atomic_json(root / "primary_inference.json", primary)
    lines = ["# Fairness v2 execution results", "", "Engineering smoke only." if protocol["quick"] else
             "Synthetic development protocol. See audit and primary inference files before making any research claim.", "",
             "| Scenario | Controller | Mean paired vehicle difference (s) | Runs with service/measurement violations |",
             "|---|---|---:|---:|"]
    for row in rows:
        pairs = row["paired_runs"]
        lines.append(f"| {row['scenario']} | {row['controller']} | {statistics.mean(p['vehicle_difference'] for p in pairs):.3f} | {sum(p['service_violation'] > 0 for p in pairs)} |")
    lines += ["", "Per-run p95 comparisons are diagnostics. primary_inference.json uses pooled events within replicas.",
              "Only mappo-constrained is confirmatory; unknown waits and insufficient samples block statistical support.",
              "Compressed evaluation records retain every crossing wait, count and missing-event flag."]
    (root / "REPORT.md").write_text("\n".join(lines))
    from .figures import export_figures
    export_figures(root, groups, primary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("all", "prepare", "train", "evaluate", "report"), default="all")
    parser.add_argument("--task-index", type=int, help="One training replica or scenario/seed group (zero-based)")
    parser.add_argument("--train-until", type=int, help="Checkpoint after this many episodes; full protocol budget stays fixed")
    args = parser.parse_args()
    if args.workers < 1 or args.episodes < 1:
        parser.error("workers and episodes must be positive")
    root = args.output or ROOT / "runs" / ("fairness-v2-quick" if args.quick else "fairness-v2-study")
    root.mkdir(parents=True, exist_ok=True)
    (root / "evaluation").mkdir(exist_ok=True)
    short = {"end_seconds": 360, "warmup_seconds": 0, "measurement_end_seconds": 240,
             "demand_end_seconds": 90, "vehicle_rate_per_hour": 720, "pedestrian_rate_per_hour": 720}
    protocol = {"quick": args.quick, "episodes": 2 if args.quick else args.episodes,
        "replicas": 1 if args.quick else 5, "validate_every": 64,
        "calibration_seeds": [19001] if args.quick else list(range(22001, 22021)),
        "validation_seeds": [19002] if args.quick else list(range(23001, 23011)),
        "test_seeds": [19003] if args.quick else list(range(34001, 34021)),
        "scenarios": ["vmedium-pmedium"] if args.quick else list(scenarios()),
        "evaluation_config": short if args.quick else {},
        "training_config": ({**short, "end_seconds": 120, "measurement_end_seconds": 120} if args.quick else
                            {"end_seconds": 1200, "measurement_end_seconds": 1200, "demand_end_seconds": 1200}),
        "status": "development execution; not a frozen confirmatory protocol"}
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted([* (ROOT / "src/traffic_control").rglob("*.py"), * (ROOT / "config").glob("*.json")])}
    # Normalise separators so Linux can resume a transferred Windows bundle.
    source = {k.replace("\\", "/"): v for k, v in source.items()}
    protocol["source_sha256"] = source
    import torch
    protocol["runtime_versions"] = {"python": platform.python_version(), "torch": torch.__version__,
        "sumo": subprocess.check_output([executable("sumo"), "--version"], text=True).splitlines()[0]}
    protocol["fingerprint"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest = root / "protocol.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Source/config changed. Use a new output directory; do not mix experiments.")
        protocol = saved
    else:
        if args.stage not in ("all", "prepare"):
            raise ValueError("Run --stage prepare before a separate training/evaluation task")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            calibration = list(pool.map(baseline_job, [(str(root), protocol, seed, "calibration") for seed in protocol["calibration_seeds"]]))
            validation_rows = list(pool.map(baseline_job, [(str(root), protocol, seed, "validation") for seed in protocol["validation_seeds"]]))
        protocol["features"] = calibration[0]["features"]
        protocol["thresholds"] = {}
        protocol["calibration_counts"] = {}
        for c in calibration[0]["result"]["crossings"]:
            waits = [w for row in calibration for w in row["result"]["crossings"][c]["waits"]]
            if not waits:
                if not args.quick:
                    raise ValueError(f"No calibration entries at {c}")
                waits = [60.]  # engineering fallback only; never used in full mode
            protocol["thresholds"][c] = quantile(waits)
            protocol["calibration_counts"][c] = len(waits)
            if not args.quick and len(waits) < 100:
                raise ValueError(f"Calibration at {c} has fewer than 100 observed entries; fix coverage before training")
        protocol["calibration_unknown_waits"] = sum(c["unknown_cohort_entries"] for row in calibration for c in row["result"]["crossings"].values())
        validation = {str(row["seed"]): row["result"] for row in validation_rows}
        atomic_json(root / "validation_baselines.json", validation)
        atomic_json(manifest, protocol)
    if args.stage == "prepare":
        return
    if args.train_until is not None:
        if not 1 <= args.train_until <= protocol["episodes"]:
            parser.error("train-until must be within the frozen episode budget")
        protocol["train_until"] = args.train_until
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage in ("all", "train"):
            jobs = [(str(root), protocol, m, o, r) for m, o in VARIANTS for r in range(protocol["replicas"])]
            if args.task_index is not None:
                jobs = [jobs[args.task_index]]
            for label in pool.map(train_job, jobs):
                print("Trained", label, flush=True)
        if args.stage in ("all", "evaluate"):
            jobs = [(str(root), protocol, scenario, seed) for scenario in protocol["scenarios"] for seed in protocol["test_seeds"]]
            if args.task_index is not None:
                jobs = [jobs[args.task_index]]
            for path in pool.map(evaluation_job, jobs):
                print("Evaluated", path, flush=True)
    if args.stage in ("all", "report"):
        report(root, protocol)
    atomic_json(root / f"runtime-{args.stage}-{args.task_index}.json", {"stage_wall_seconds": time.perf_counter() - started,
        "workers": args.workers, "device": "cpu", "quick": args.quick})


if __name__ == "__main__":
    main()
