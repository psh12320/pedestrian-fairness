"""Held-out, matched-seed comparison of six controllers and crossing-level outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .cli import run
from .ppo import PPOModel
from .scenario import ROOT


CONTROLLERS = ("fixed", "actuated", "ippo-vehicle", "ippo-pedestrian",
               "mappo-vehicle", "mappo-pedestrian")
MEASURES = ("completed", "unfinished", "uninserted", "vehicle_time_loss",
            "pedestrian_mean_wait", "pedestrian_p95_wait", "worst_crossing_p95",
            "unserved_requests")


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(proportion * len(ordered)) - 1)]


def bootstrap_interval(values: list[float], samples: int, seed: int) -> dict:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    rng = random.Random(seed)
    estimates = [statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)]
    estimates.sort()
    return {"mean": statistics.mean(values),
            "ci95_low": percentile(estimates, 0.025),
            "ci95_high": percentile(estimates, 0.975), "n": len(values)}


def run_rows(output: Path, scenario: str, seed: int, label: str, metrics: dict) -> tuple[dict, list[dict]]:
    events = json.loads((output / "crossing_events.json").read_text(encoding="utf-8"))
    waits = [event["wait_seconds"] for event in events if event["wait_seconds"] is not None]
    crossings = metrics["pedestrians_by_crossing"]
    vehicle = metrics["vehicles"]
    row = {"scenario": scenario, "seed": seed, "controller": label,
           "demand_sha256": metrics["demand_sha256"],
           "completed": vehicle["completed_count"], "unfinished": vehicle["unfinished_count"],
           "uninserted": vehicle["uninserted_count"],
           "vehicle_time_loss": vehicle["completed_mean_time_loss_seconds"],
           "pedestrian_mean_wait": statistics.mean(waits) if waits else None,
           "pedestrian_p95_wait": percentile(waits, 0.95),
           "worst_crossing_p95": max((data["p95_wait_seconds"] or 0)
                                     for data in crossings.values()),
           "unserved_requests": sum(data["unserved_count"] for data in crossings.values())}
    crossing_rows = [{"scenario": scenario, "seed": seed, "controller": label,
                      "crossing": crossing,
                      "crossing_edges": metrics["crossing_edges"].get(crossing, ""),
                      **data} for crossing, data in sorted(crossings.items())]
    return row, crossing_rows


def evaluate(output: Path | None = None, seeds: list[int] | None = None,
             scenarios: list[str] | None = None) -> dict:
    settings = json.loads((ROOT / "config" / "evaluation.json").read_text(encoding="utf-8"))
    output = output or ROOT / "runs" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    seeds = seeds or settings["seeds"]
    scenarios_spec = []
    for vehicle_name, vehicle_count in settings["vehicle_levels"].items():
        for pedestrian_name, pedestrian_count in settings["pedestrian_levels"].items():
            scenarios_spec.append((f"v{vehicle_name}-p{pedestrian_name}",
                                   vehicle_count, pedestrian_count, False))
    if settings["include_slow_medium"]:
        scenarios_spec.append(("vmedium-pmedium-slow", settings["vehicle_levels"]["medium"],
                               settings["pedestrian_levels"]["medium"], True))
    if scenarios is not None:
        scenarios_spec = [entry for entry in scenarios_spec if entry[0] in scenarios]
    if not scenarios_spec:
        raise ValueError("No evaluation scenarios selected")
    models = {}
    for label in CONTROLLERS[2:]:
        model = PPOModel.load(ROOT / "models" / f"{label}.json")
        if model.episodes_trained < 1:
            raise ValueError(f"{label} has not been trained")
        models[label] = model

    rows = []
    crossing_rows = []
    for scenario, vehicle_count, pedestrian_count, slow in scenarios_spec:
        for seed in seeds:
            demand_hash = None
            for label in CONTROLLERS:
                destination = output / scenario / f"seed-{seed}" / label
                overrides = {"vehicle_count_per_axis": vehicle_count,
                             "pedestrian_count": pedestrian_count,
                             "vehicle_depart_last_second": settings["departure_last_second"],
                             "pedestrian_depart_last_second": settings["departure_last_second"]}
                if label in models:
                    model = models[label]
                    metrics = run("policy", seed, destination, ROOT / "config" / "milestone.json",
                                  junctions=3, slow_pedestrians=slow, config_overrides=overrides,
                                  policy_callback=lambda tls, obs, global_obs, eligible:
                                      model.select(tls, obs, global_obs, eligible, training=False))
                else:
                    metrics = run(label, seed, destination, ROOT / "config" / "milestone.json",
                                  junctions=3, slow_pedestrians=slow, config_overrides=overrides)
                if demand_hash is None:
                    demand_hash = metrics["demand_sha256"]
                elif metrics["demand_sha256"] != demand_hash:
                    raise AssertionError("Controllers received mismatched demand")
                row, crossing = run_rows(destination, scenario, seed, label, metrics)
                rows.append(row)
                crossing_rows.extend(crossing)
                print(f"evaluated {scenario} seed {seed} {label}", flush=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "crossings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(crossing_rows[0]))
        writer.writeheader()
        writer.writerows(crossing_rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["controller"])].append(row)
    summary = {}
    for (scenario, label), group in sorted(groups.items()):
        summary.setdefault(scenario, {})[label] = {
            measure: bootstrap_interval([float(row[measure]) for row in group
                                         if row[measure] is not None],
                                        settings["bootstrap_replicates"],
                                        settings["bootstrap_seed"])
            for measure in MEASURES}
    paired = {}
    for scenario in summary:
        fixed = {row["seed"]: row for row in groups[(scenario, "fixed")]}
        for label in CONTROLLERS[1:]:
            candidate = {row["seed"]: row for row in groups[(scenario, label)]}
            paired.setdefault(scenario, {})[label] = {
                measure: bootstrap_interval(
                    [float(candidate[seed][measure]) - float(fixed[seed][measure])
                     for seed in seeds if candidate[seed][measure] is not None
                     and fixed[seed][measure] is not None],
                    settings["bootstrap_replicates"], settings["bootstrap_seed"])
                for measure in MEASURES}
    training = json.loads((ROOT / "config" / "training.json").read_text(encoding="utf-8"))
    result = {"evaluation_settings": settings,
              "training_seed_range": [training["seed_start"],
                                      training["seed_start"] + training["episodes"] - 1],
              "summary": summary, "paired_vs_fixed": paired,
              "note": "95% percentile bootstrap intervals across held-out seeds; completed-trip time loss must be read with unfinished and uninserted counts."}
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--scenarios", nargs="+")
    args = parser.parse_args()
    evaluate(args.output, args.seeds, args.scenarios)


if __name__ == "__main__":
    main()
