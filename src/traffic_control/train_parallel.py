"""Train independent PPO replicas in parallel, then merge them deterministically.

Do not run this beside the serial trainer against the same output/model folders.
Each worker owns one replica and its temporary SUMO files. A completed replica
has an atomic result.json marker, so an interrupted invocation can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .cli import run
from .ppo import PPOModel
from .scenario import ROOT
from .train import controller_label, objective_for_weight, validation_score


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _settings_hash(settings: dict, episodes: int) -> str:
    payload = json.dumps({"settings": settings, "episodes": episodes},
                         sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key(entry: dict) -> tuple[str, int]:
    return entry["controller"], int(entry["training_run"])


def _read_result(shard: Path, model_dir: Path, fingerprint: str) -> dict | None:
    result_path = shard / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("fingerprint") != fingerprint:
        raise ValueError(f"Shard {shard} uses different settings or episode budget")
    model = model_dir / Path(result["entry"]["model_path"]).name
    if not model.exists() or hashlib.sha256(model.read_bytes()).hexdigest() != result["model_sha256"]:
        return None
    if not (shard / "episodes.jsonl").exists() or not (
            shard / "validation.jsonl").exists():
        return None
    return result


def _worker(task: dict) -> dict:
    settings = task["settings"]
    episodes = task["episodes"]
    method = task["method"]
    weight = task["weight"]
    run_index = task["training_run"]
    run_seed = task["training_seed"]
    output = Path(task["output"])
    model_dir = Path(task["model_dir"])
    fingerprint = task["fingerprint"]
    label = controller_label(method, weight)
    shard = output / "shards" / f"{label}-r{run_index}"
    shard.mkdir(parents=True, exist_ok=True)
    prior = _read_result(shard, model_dir, fingerprint)
    if prior is not None:
        return prior["entry"]
    model = PPOModel(method, objective_for_weight(weight), ["C0", "C1", "C2"],
                     settings, seed=run_seed)
    best_score, best_episode = math.inf, 0
    checkpoint = shard / "best-checkpoint.json"
    with (shard / "episodes.jsonl").open("w", encoding="utf-8") as episode_log, \
            (shard / "validation.jsonl").open("w", encoding="utf-8") as validation_log:
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
            with tempfile.TemporaryDirectory(prefix="episode-", dir=shard) as scratch:
                metrics = run(
                    "policy", demand_seed, Path(scratch),
                    ROOT / "config" / "milestone.json", junctions=3,
                    config_overrides=overrides,
                    policy_callback=lambda tls, obs, global_obs, eligible:
                        model.select(tls, obs, global_obs, eligible, training=True),
                    trace_sink=trace, objective=objective_for_weight(weight),
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
                validation = []
                for seed in settings["validation_seeds"]:
                    with tempfile.TemporaryDirectory(
                            prefix="validation-", dir=shard) as scratch:
                        item = run(
                            "policy", seed, Path(scratch),
                            ROOT / "config" / "milestone.json", junctions=3,
                            config_overrides=settings["validation_scenario"],
                            policy_callback=lambda tls, obs, global_obs, eligible:
                                model.select(tls, obs, global_obs, eligible,
                                             training=False),
                            objective=objective_for_weight(weight),
                            pedestrian_reward_weight=weight)
                    validation.append(item)
                score = validation_score(validation, weight, settings)
                validation_log.write(json.dumps({
                    "controller": label, "training_run": run_index,
                    "training_seed": run_seed, "episode": episode + 1,
                    "score": score,
                }, sort_keys=True) + "\n")
                validation_log.flush()
                if score < best_score:
                    best_score, best_episode = score, episode + 1
                    model.save(checkpoint)
    model_path = model_dir / f"{label}-r{run_index}.json"
    temporary_model = model_path.with_suffix(".json.tmp")
    shutil.copyfile(checkpoint, temporary_model)
    os.replace(temporary_model, model_path)
    checkpoint.unlink()
    entry = {
        "controller": label, "method": method, "reward_weight": weight,
        "objective": objective_for_weight(weight), "training_run": run_index,
        "training_seed": run_seed, "episodes_run": episodes,
        "selected_episode": best_episode, "validation_score": best_score,
        "model_path": str(model_path.resolve()),
    }
    _atomic_json(shard / "result.json", {
        "fingerprint": fingerprint, "entry": entry,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    })
    return entry


def _merge(output: Path, model_dir: Path, settings: dict, episodes: int,
           fingerprint: str, serial_entries: list[dict]) -> dict:
    shard_results = {}
    for method in ("ippo", "mappo"):
        for weight in settings["reward_weights"]:
            for index in range(len(settings["training_run_seeds"])):
                label = controller_label(method, weight)
                shard = output / "shards" / f"{label}-r{index}"
                result = _read_result(shard, model_dir, fingerprint)
                if result is not None:
                    shard_results[(label, index)] = result["entry"]
    entries = {_key(item): item for item in serial_entries
               if (model_dir / Path(item["model_path"]).name).exists()}
    entries.update(shard_results)
    order = {(controller_label(method, weight), index): ordinal
             for ordinal, (method, weight, index) in enumerate(
                 (method, weight, index)
                 for method in ("ippo", "mappo")
                 for weight in settings["reward_weights"]
                 for index in range(len(settings["training_run_seeds"])))}
    selected = sorted(entries.values(), key=lambda item: order[_key(item)])
    manifest = {
        "version": 2, "settings": settings, "episodes_requested": episodes,
        "complete_protocol": (
            episodes in (settings["episodes"], settings["extended_episodes"])
            and len(selected) == len(order)
            and {_key(item) for item in selected} == set(order)),
        "models": selected,
    }
    # Keep the preexisting serial log rows only for replicas without shard output.
    serial_keys = {_key(item) for item in serial_entries} - set(shard_results)
    for filename in ("episodes.jsonl", "validation.jsonl"):
        old_path = output / filename
        by_key = {}
        if old_path.exists():
            for line in old_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (item["controller"], int(item["training_run"]))
                if key in serial_keys:
                    by_key.setdefault(key, []).append(line)
        for key in shard_results:
            shard = output / "shards" / f"{key[0]}-r{key[1]}"
            by_key[key] = (shard / filename).read_text(
                encoding="utf-8").splitlines()
        lines = [line for key in sorted(by_key, key=lambda item: order[item])
                 for line in sorted(by_key[key],
                                    key=lambda raw: json.loads(raw)["episode"])]
        temporary = old_path.with_suffix(old_path.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + ("\n" if lines else ""),
                             encoding="utf-8")
        temporary.replace(old_path)
    for method in ("ippo", "mappo"):
        for weight in (0, 3):
            label = controller_label(method, weight)
            source = model_dir / f"{label}-r0.json"
            if source.exists():
                alias = model_dir / f"{method}-{'vehicle' if weight == 0 else 'pedestrian'}.json"
                shutil.copyfile(source, alias)
    _atomic_json(model_dir / "training_manifest.json", manifest)
    return manifest


def train_parallel(output: Path | None = None, model_dir: Path | None = None,
                   workers: int = 4, pilot: bool = False,
                   episodes_override: int | None = None) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("Use between one and four SUMO workers")
    settings = json.loads((ROOT / "config" / "training.json").read_text(
        encoding="utf-8"))
    if pilot:
        if episodes_override is not None:
            raise ValueError("Pilot episode count is fixed at one")
        settings["validation_seeds"] = settings["validation_seeds"][:1]
        episodes = 1
        output = output or ROOT / "runs" / "pilot-parallel-local-reward-training"
        model_dir = model_dir or ROOT / "runs" / "pilot-parallel-local-reward-models"
        methods, weights, run_indices = ("ippo", "mappo"), (0,), (0,)
    else:
        episodes = settings["episodes"] if episodes_override is None else episodes_override
        if episodes not in (settings["episodes"], settings["extended_episodes"]):
            raise ValueError("Use the prespecified 150- or 300-episode budget")
        output = output or ROOT / "runs" / "training-v2"
        model_dir = model_dir or ROOT / "models" / "v2"
        methods, weights = ("ippo", "mappo"), settings["reward_weights"]
        run_indices = range(len(settings["training_run_seeds"]))
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    fingerprint = _settings_hash(settings, episodes)
    manifest_path = model_dir / "training_manifest.json"
    serial_entries = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("settings") != settings or existing.get(
                "episodes_requested") != episodes:
            raise ValueError("Existing manifest has a different protocol or budget")
        serial_entries = existing["models"]
    existing_keys = {_key(item) for item in serial_entries
                     if (model_dir / Path(item["model_path"]).name).exists()}
    pending = []
    for method in methods:
        for weight in weights:
            for index in run_indices:
                label = controller_label(method, weight)
                shard = output / "shards" / f"{label}-r{index}"
                if (label, index) in existing_keys or _read_result(
                        shard, model_dir, fingerprint) is not None:
                    continue
                pending.append({
                    "settings": settings, "episodes": episodes, "method": method,
                    "weight": weight, "training_run": index,
                    "training_seed": settings["training_run_seeds"][index],
                    "output": str(output.resolve()), "model_dir": str(model_dir.resolve()),
                    "fingerprint": fingerprint,
                })
    print(f"{len(pending)} independent replicas pending; {workers} worker(s)", flush=True)
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            label = controller_label(task["method"], task["weight"])
            try:
                item = future.result()
            except Exception as exc:
                failures.append(f"{label}-r{task['training_run']}: {exc}")
                print(f"FAILED {failures[-1]}", flush=True)
            else:
                print(f"completed {item['controller']}-r{item['training_run']}", flush=True)
            # Merge completed replicas after every worker. An interrupted parent
            # loses at most the still-running replica; its shard is retrained.
            _merge(output, model_dir, settings, episodes, fingerprint, serial_entries)
    manifest = _merge(output, model_dir, settings, episodes, fingerprint,
                      serial_entries)
    if failures:
        raise RuntimeError("; ".join(failures))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int,
                        help="150 by default; use 300 only after the validation rule triggers")
    parser.add_argument("--pilot", action="store_true",
                        help="Two one-episode replicas in separate pilot folders")
    args = parser.parse_args()
    result = train_parallel(args.output, args.model_dir, args.workers,
                            args.pilot, args.episodes)
    print(f"{len(result['models'])} replicas complete; "
          f"full protocol={result['complete_protocol']}")


if __name__ == "__main__":
    main()
