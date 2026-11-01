"""Train independent IPPO/MAPPO replicas with validation-only checkpoint selection.

SUMO episode files live in temporary folders so full training stays within a
small disk budget. Final-test seeds never influence model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import tempfile
from pathlib import Path

from .cli import run
from .ppo import PPOModel
from .scenario import ROOT


def controller_label(method: str, weight: float) -> str:
    return f"{method}-w{weight:g}"


def objective_for_weight(weight: float) -> str:
    return "vehicle" if weight == 0 else "pedestrian"


def validation_score(metrics: list[dict], weight: float, settings: dict) -> float:
    """Prespecified score includes penalties for traffic left unserved."""
    scores = []
    for item in metrics:
        vehicles = item["vehicles"]
        crossing = item["pedestrians_by_crossing"]
        entries = sum(value["entered_count"] for value in crossing.values())
        wait_sum = sum((value["mean_wait_seconds"] or 0) * value["entered_count"]
                       for value in crossing.values())
        unserved = sum(value["unserved_count"] for value in crossing.values())
        vehicle_loss = vehicles["completed_mean_time_loss_seconds"]
        if vehicle_loss is None:
            vehicle_loss = settings["unfinished_vehicle_penalty_seconds"]
        vehicle_loss += settings["unfinished_vehicle_penalty_seconds"] * (
            vehicles["unfinished_count"] + vehicles["uninserted_count"]
        ) / max(1, vehicles["scheduled_count"])
        pedestrian_loss = (wait_sum + settings["unserved_request_penalty_seconds"]
                           * unserved) / max(1, entries + unserved)
        scores.append(vehicle_loss + weight * pedestrian_loss)
    return statistics.mean(scores)


def train(episodes: int | None = None, output: Path | None = None,
          model_dir: Path | None = None, run_limit: int | None = None,
          weights: list[float] | None = None,
          settings_path: Path | None = None, smoke: bool = False) -> dict:
    settings = json.loads((settings_path or ROOT / "config" / "training.json")
                          .read_text(encoding="utf-8"))
    if smoke:
        settings["validation_seeds"] = settings["validation_seeds"][:1]
        episodes, run_limit, weights = 1, 1, [0]
        output = output or ROOT / "runs" / "pilot-local-reward-training"
        model_dir = model_dir or ROOT / "runs" / "pilot-local-reward-models"
    episodes = settings["episodes"] if episodes is None else episodes
    if episodes < 1:
        raise ValueError("Training needs at least one episode")
    all_run_seeds = list(settings["training_run_seeds"])
    run_seeds = all_run_seeds if run_limit is None else all_run_seeds[:run_limit]
    weights = list(settings["reward_weights"] if weights is None else weights)
    if not run_seeds or not weights or any(weight < 0 or not math.isfinite(weight)
                                           for weight in weights):
        raise ValueError("Select run seeds and finite nonnegative reward weights")
    output = output or ROOT / "runs" / "training-v2"
    model_dir = model_dir or ROOT / "models" / "v2"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    manifest_path = model_dir / "training_manifest.json"
    expected_full_protocol = (
        episodes in (settings["episodes"], settings["extended_episodes"])
        and run_seeds == all_run_seeds and weights == settings["reward_weights"])
    manifest = {
        "version": 2, "settings": settings, "episodes_requested": episodes,
        "complete_protocol": False,
        "models": [],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous.get("settings") != settings
                or previous.get("episodes_requested") != episodes):
            raise ValueError("Existing training manifest uses different settings or budget")
        manifest["models"] = [
            entry for entry in previous["models"]
            if (model_dir / Path(entry["model_path"]).name).exists()]
    completed = {(entry["controller"], entry["training_run"])
                 for entry in manifest["models"]}
    # A run interrupted before its model was saved is retrained from episode 1.
    # Remove only that partial run's log rows; completed replicas are preserved.
    for name in ("episodes.jsonl", "validation.jsonl"):
        path = output / name
        if path.exists():
            retained = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (item["controller"], item["training_run"]) in completed:
                    retained.append(line)
            path.write_text("\n".join(retained) + ("\n" if retained else ""),
                            encoding="utf-8")
    with (output / "episodes.jsonl").open("a", encoding="utf-8") as episode_log, \
            (output / "validation.jsonl").open("a", encoding="utf-8") as validation_log:
        for method in ("ippo", "mappo"):
            for weight in weights:
                label = controller_label(method, weight)
                objective = objective_for_weight(weight)
                for run_index, run_seed in enumerate(run_seeds):
                    if (label, run_index) in completed:
                        print(f"{label} seed {run_seed}: already complete", flush=True)
                        continue
                    model = PPOModel(method, objective, ["C0", "C1", "C2"], settings,
                                     seed=run_seed)
                    best_score, best_episode = math.inf, 0
                    checkpoint = checkpoint_dir / f"{label}-r{run_index}.json"
                    for episode in range(episodes):
                        factor = settings["episode_demand_factors"][
                            episode % len(settings["episode_demand_factors"])]
                        base = settings["training_scenario"]
                        overrides = {
                            **base,
                            "vehicle_count_per_axis": round(
                                base["vehicle_count_per_axis"] * factor),
                            "pedestrian_count": round(base["pedestrian_count"] * factor),
                        }
                        demand_seed = settings["demand_seed_start"] + episode
                        trace: dict = {}
                        with tempfile.TemporaryDirectory(prefix="episode-", dir=output) as scratch:
                            metrics = run(
                                "policy", demand_seed, Path(scratch),
                                ROOT / "config" / "milestone.json", junctions=3,
                                config_overrides=overrides,
                                policy_callback=lambda tls, obs, global_obs, eligible:
                                    model.select(tls, obs, global_obs, eligible, training=True),
                                trace_sink=trace, objective=objective,
                                pedestrian_reward_weight=weight)
                        update = model.update(trace)
                        episode_log.write(json.dumps({
                            "controller": label, "training_run": run_index,
                            "training_seed": run_seed, "episode": episode + 1,
                            "demand_seed": demand_seed, "demand_factor": factor,
                            "vehicle_time_loss": metrics["vehicles"][
                                "completed_mean_time_loss_seconds"],
                            "completed": metrics["vehicles"]["completed_count"],
                            "unfinished": metrics["vehicles"]["unfinished_count"],
                            "uninserted": metrics["vehicles"]["uninserted_count"],
                            "pedestrian_entries": sum(
                                value["entered_count"] for value in
                                metrics["pedestrians_by_crossing"].values()),
                            "update": update,
                        }, sort_keys=True) + "\n")
                        episode_log.flush()
                        if ((episode + 1) % settings["validation_every_episodes"] == 0
                                or episode + 1 == episodes):
                            val_metrics = []
                            for val_seed in settings["validation_seeds"]:
                                with tempfile.TemporaryDirectory(prefix="validation-",
                                                                 dir=output) as scratch:
                                    result = run(
                                        "policy", val_seed, Path(scratch),
                                        ROOT / "config" / "milestone.json", junctions=3,
                                        config_overrides=settings["validation_scenario"],
                                        policy_callback=lambda tls, obs, global_obs, eligible:
                                            model.select(tls, obs, global_obs, eligible,
                                                         training=False),
                                        objective=objective,
                                        pedestrian_reward_weight=weight)
                                val_metrics.append(result)
                            score = validation_score(val_metrics, weight, settings)
                            validation_log.write(json.dumps({
                                "controller": label, "training_run": run_index,
                                "training_seed": run_seed, "episode": episode + 1,
                                "score": score,
                            }, sort_keys=True) + "\n")
                            validation_log.flush()
                            if score < best_score:
                                best_score, best_episode = score, episode + 1
                                model.save(checkpoint)
                        print(f"{label} seed {run_seed}: {episode + 1}/{episodes}", flush=True)
                    path = model_dir / f"{label}-r{run_index}.json"
                    shutil.copyfile(checkpoint, path)
                    checkpoint.unlink()
                    manifest["models"].append({
                        "controller": label, "method": method, "reward_weight": weight,
                        "objective": objective, "training_run": run_index,
                        "training_seed": run_seed, "episodes_run": episodes,
                        "selected_episode": best_episode,
                        "validation_score": best_score,
                        "model_path": str(path.resolve()),
                    })
                    # Alias only for GUI demonstration. Research evaluation uses every replica.
                    if run_index == 0 and weight in (0, 3):
                        alias = f"{method}-{'vehicle' if weight == 0 else 'pedestrian'}.json"
                        shutil.copyfile(path, model_dir / alias)
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    manifest["complete_protocol"] = expected_full_protocol and len(manifest["models"]) == (
        2 * len(weights) * len(all_run_seeds))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--runs", type=int, help="Use fewer training seeds for a smoke run")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny pilot in runs/pilot-*; never marks the protocol complete")
    args = parser.parse_args()
    train(args.episodes, args.output, args.model_dir, args.runs, args.weights,
          smoke=args.smoke)


if __name__ == "__main__":
    main()
