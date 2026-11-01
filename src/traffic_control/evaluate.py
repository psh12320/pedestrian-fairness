"""Matched-demand evaluation across training replicas and locked demand seeds."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from .cli import prepare_inputs, run
from .ppo import PPOModel
from .scenario import ROOT


BASELINES = ("fixed", "actuated", "pressure")
MEASURES = ("scheduled", "completed", "unfinished", "uninserted",
            "vehicle_time_loss", "pedestrian_mean_wait", "pedestrian_p95_wait",
            "worst_crossing_p95", "unserved_requests",
            "connector_near_capacity_seconds",
            "connector_near_capacity_with_upstream_queue_seconds",
            "connector_max_occupancy_fraction",
            "connector_max_halted")


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(proportion * len(ordered)) - 1)]


def bootstrap_interval(values: list[float], samples: int, seed: int) -> dict:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    rng = random.Random(seed)
    estimates = [statistics.mean(rng.choice(values) for _ in values)
                 for _ in range(samples)]
    return {"mean": statistics.mean(values),
            "ci95_low": percentile(estimates, 0.025),
            "ci95_high": percentile(estimates, 0.975), "n": len(values)}


def scenario_specs(settings: dict) -> dict[str, tuple[dict, bool]]:
    specs = {}
    for vehicle_name, vehicle_count in settings["vehicle_levels"].items():
        for pedestrian_name, pedestrian_count in settings["pedestrian_levels"].items():
            specs[f"v{vehicle_name}-p{pedestrian_name}"] = ({
                "vehicle_count_per_axis": vehicle_count,
                "pedestrian_count": pedestrian_count,
                "vehicle_depart_last_second": settings["departure_last_second"],
                "pedestrian_depart_last_second": settings["departure_last_second"],
            }, False)
    if settings["include_slow_medium"]:
        base = dict(specs["vmedium-pmedium"][0])
        specs["vmedium-pmedium-slow"] = (base, True)
    for name, overrides in settings.get("extra_scenarios", {}).items():
        if name in specs:
            raise ValueError(f"Duplicate scenario name {name}")
        specs[name] = (overrides, False)
    return specs


def run_rows(metrics: dict, events: list[dict], scenario: str, seed: int,
             label: str, training_run: int | None, training_seed: int | None) -> tuple[dict, list[dict], list[dict]]:
    waits = [event["wait_seconds"] for event in events
             if event["wait_seconds"] is not None]
    crossings = metrics["pedestrians_by_crossing"]
    vehicle = metrics["vehicles"]
    connector = metrics.get("connector_diagnostics", {})
    common = {"scenario": scenario, "seed": seed, "controller": label,
              "training_run": "" if training_run is None else training_run,
              "training_seed": "" if training_seed is None else training_seed}
    row = {**common, "demand_sha256": metrics["demand_sha256"],
           "sumo_version": metrics["sumo_version"][1],
           "scheduled": vehicle["scheduled_count"],
           "completed": vehicle["completed_count"],
           "unfinished": vehicle["unfinished_count"],
           "uninserted": vehicle["uninserted_count"],
           "vehicle_time_loss": vehicle["completed_mean_time_loss_seconds"],
           "pedestrian_mean_wait": statistics.mean(waits) if waits else None,
           "pedestrian_p95_wait": percentile(waits, 0.95),
           "worst_crossing_p95": max((data["p95_wait_seconds"] or 0)
                                     for data in crossings.values()),
           "unserved_requests": sum(data["unserved_count"]
                                     for data in crossings.values()),
           "connector_near_capacity_seconds": connector.get("near_capacity_seconds"),
           "connector_near_capacity_with_upstream_queue_seconds":
               connector.get("near_capacity_with_upstream_queue_seconds"),
           "connector_max_occupancy_fraction": connector.get("max_occupancy_fraction"),
           "connector_max_halted": connector.get("max_halted_on_connector")}
    crossing_rows = [{**common, "crossing": crossing,
                      "crossing_edges": metrics["crossing_edges"].get(crossing, ""),
                      **data} for crossing, data in sorted(crossings.items())]
    event_rows = [{**common, **event} for event in events]
    return row, crossing_rows, event_rows


def hierarchical_interval(rows: list[dict], measure: str, replicates: int,
                          seed: int, reference: list[dict] | None = None) -> dict:
    """Bootstrap training initialisations and matched demand seeds separately."""
    by_run = defaultdict(dict)
    for row in rows:
        if row[measure] not in (None, ""):
            by_run[str(row["training_run"])][int(row["seed"])] = float(row[measure])
    if not by_run:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    ref_by_seed = {}
    if reference is not None:
        ref_by_seed = {int(row["seed"]): float(row[measure]) for row in reference
                       if row[measure] not in (None, "")}
    seeds = sorted(set.intersection(*(set(group) for group in by_run.values())))
    if reference is not None:
        seeds = sorted(set(seeds) & set(ref_by_seed))
    run_ids = sorted(by_run)
    if not seeds:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    point = statistics.mean(by_run[run_id][demand_seed] -
                            (ref_by_seed[demand_seed] if reference is not None else 0)
                            for run_id in run_ids for demand_seed in seeds)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled_runs = [rng.choice(run_ids) for _ in run_ids]
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        estimates.append(statistics.mean(
            by_run[run_id][demand_seed] -
            (ref_by_seed[demand_seed] if reference is not None else 0)
            for run_id in sampled_runs for demand_seed in sampled_seeds))
    return {"mean": point, "ci95_low": percentile(estimates, 0.025),
            "ci95_high": percentile(estimates, 0.975),
            "n": len(seeds), "training_runs": len(run_ids)}


def crossing_comparison(events: list[dict], scenario: str, controller: str,
                        seeds: list[int], training_runs: list[int], replicates: int,
                        rng_seed: int, minimum_entries: int,
                        crossing_ids: set[str] | None = None) -> tuple[list[dict], dict]:
    """Crossing p95 uses pooled event waits; bootstrap resamples matched seeds."""
    entered = defaultdict(list)
    unserved = defaultdict(int)
    crossings = set(crossing_ids or [])
    for row in events:
        if row["scenario"] != scenario or row["controller"] not in (controller, "actuated"):
            continue
        crossing = row["crossing_id"]
        crossings.add(crossing)
        run_id = (int(row["training_run"]) if row["training_run"] not in ("", None)
                  else -1)
        key = (row["controller"], run_id, int(row["seed"]), crossing)
        if row["wait_seconds"] in (None, ""):
            unserved[key] += 1
        else:
            entered[key].append(float(row["wait_seconds"]))
    rng = random.Random(rng_seed)
    schedules = [([rng.choice(training_runs) for _ in training_runs],
                  [rng.choice(seeds) for _ in seeds])
                 for _ in range(replicates)]
    rows = []
    max_differences = [float("-inf")] * replicates
    complete_counts = [0] * replicates
    for crossing in sorted(crossings):
        def pooled(label: str, run_id: int, selected_seeds: list[int]) -> list[float]:
            return [wait for demand_seed in selected_seeds
                    for wait in entered[(label, run_id, demand_seed, crossing)]]
        baseline = pooled("actuated", -1, seeds)
        per_run = [pooled(controller, run_id, seeds) for run_id in training_runs]
        baseline_p95 = percentile(baseline, 0.95)
        run_p95s = [percentile(waits, 0.95) for waits in per_run]
        valid = baseline_p95 is not None and all(item is not None for item in run_p95s)
        point = (statistics.mean(run_p95s) - baseline_p95) if valid else None
        sampled = []
        if valid:
            for iteration, (sampled_runs, sampled_seeds) in enumerate(schedules):
                base_p95 = percentile(pooled("actuated", -1, sampled_seeds), 0.95)
                model_p95 = [percentile(pooled(controller, run_id, sampled_seeds), 0.95)
                             for run_id in sampled_runs]
                if base_p95 is None or any(item is None for item in model_p95):
                    continue
                delta = statistics.mean(model_p95) - base_p95
                sampled.append(delta)
                max_differences[iteration] = max(max_differences[iteration], delta)
                complete_counts[iteration] += 1
        min_model_entries = min((len(waits) for waits in per_run), default=0)
        rows.append({
            "scenario": scenario, "controller": controller, "crossing": crossing,
            "actuated_p95_seconds": baseline_p95,
            "model_mean_p95_seconds": statistics.mean(run_p95s) if valid else None,
            "p95_difference_seconds": point,
            "ci95_low": percentile(sampled, 0.025),
            "ci95_high": percentile(sampled, 0.975),
            "actuated_entries": len(baseline),
            "minimum_model_entries_per_training_run": min_model_entries,
            "actuated_unserved": sum(unserved[("actuated", -1, seed, crossing)]
                                    for seed in seeds),
            "mean_model_unserved": statistics.mean(
                sum(unserved[(controller, run_id, seed, crossing)] for seed in seeds)
                for run_id in training_runs),
            "adequate_entries": min(len(baseline), min_model_entries) >= minimum_entries,
        })
    valid_max = [item for item, count in zip(max_differences, complete_counts)
                 if count == len(crossings)]
    complete_fraction = len(valid_max) / replicates if replicates else 0.0
    point_max = max((row["p95_difference_seconds"] for row in rows
                     if row["p95_difference_seconds"] is not None), default=None)
    return rows, {"mean": point_max, "ci95_low": percentile(valid_max, 0.025),
                  "ci95_high": percentile(valid_max, 0.975),
                  "complete_bootstrap_fraction": complete_fraction,
                  "all_crossings_measured": bool(rows) and all(
                      row["p95_difference_seconds"] is not None for row in rows),
                  "all_crossings_adequate": bool(rows) and all(
                      row["adequate_entries"] for row in rows)}


def primary_criterion(differences: dict, maximum: dict,
                      crossings: list[dict]) -> tuple[bool, bool]:
    """Require per-crossing service as well as entered-person tail waits.

    A controller can lower a crossing's entered-only p95 by leaving its
    longest-waiting requests pending. A lower network-wide pending total does
    not rule out that shift, so each crossing must be checked separately.
    """
    no_more_pending_at_any_crossing = bool(crossings) and all(
        row["mean_model_unserved"] <= row["actuated_unserved"]
        for row in crossings)
    point_pass = (
        differences["vehicle_time_loss"]["mean"] is not None
        and differences["vehicle_time_loss"]["mean"] < 0
        and maximum["all_crossings_measured"]
        and maximum["mean"] is not None and maximum["mean"] <= 0
        and no_more_pending_at_any_crossing
        and differences["completed"]["mean"] >= 0
        and all(differences[name]["mean"] <= 0 for name in
                ("unfinished", "uninserted", "unserved_requests")))
    robust_pass = (point_pass and maximum["all_crossings_adequate"]
                   and maximum["complete_bootstrap_fraction"] >= 0.95
                   and differences["vehicle_time_loss"]["ci95_high"] < 0
                   and maximum["ci95_high"] is not None
                   and maximum["ci95_high"] <= 0)
    return point_pass, robust_pass


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        if not rows:
            return
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output: Path | None = None, seeds: list[int] | None = None,
             scenarios: list[str] | None = None, model_dir: Path | None = None,
             settings_path: Path | None = None, smoke: bool = False,
             group_only: bool = False) -> dict:
    settings = json.loads((settings_path or ROOT / "config" / "evaluation.json")
                          .read_text(encoding="utf-8"))
    if smoke:
        output = output or ROOT / "runs" / "pilot-local-reward-evaluation"
        model_dir = model_dir or ROOT / "runs" / "pilot-local-reward-models"
        seeds = seeds or [6001]
        scenarios = scenarios or ["vmedium-pmedium"]
    output = output or ROOT / "runs" / "evaluation-v4"
    model_dir = model_dir or ROOT / "models" / "v3"
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(settings["seeds"] if seeds is None else seeds)
    if not seeds or (not smoke and set(seeds) & set(settings["exploratory_seeds"])):
        raise ValueError("Use nonempty final seeds distinct from exploratory seeds")
    trace_seed = seeds[0] if smoke else settings["seeds"][0]
    specs = scenario_specs(settings)
    if scenarios is not None:
        unknown = set(scenarios) - set(specs)
        if unknown:
            raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
        specs = {name: specs[name] for name in scenarios}
    if not specs:
        raise ValueError("No evaluation scenarios selected")
    manifest = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    if not manifest["models"]:
        raise ValueError("Train models before evaluating")
    if not smoke:
        training = manifest["settings"]
        expected_models = {(method, weight, index)
                           for method in ("ippo", "mappo")
                           for weight in training["reward_weights"]
                           for index in range(len(training["training_run_seeds"]))}
        actual_models = {(item["method"], item["reward_weight"],
                          item["training_run"]) for item in manifest["models"]}
        if (not manifest.get("complete_protocol")
                or actual_models != expected_models
                or len(manifest["models"]) != len(expected_models)):
            raise ValueError("Final evaluation requires all independent trained replicas")
    model_entries = []
    # A resumed group is valid only for the same simulation implementation,
    # milestone configuration, model bytes, and evaluation protocol.
    source_paths = (
        "config/milestone.json", "config/training.json",
        "src/traffic_control/cli.py", "src/traffic_control/connected.py",
        "src/traffic_control/scenario.py", "src/traffic_control/signals.py",
        "src/traffic_control/metrics.py", "src/traffic_control/ppo.py",
        "src/traffic_control/train.py", "src/traffic_control/evaluate.py",
    )
    source_sha256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in source_paths}
    signature = hashlib.sha256(json.dumps(
        {"settings": settings, "manifest": manifest,
         "source_sha256": source_sha256}, sort_keys=True).encode("utf-8"))
    for entry in manifest["models"]:
        path = model_dir / Path(entry["model_path"]).name
        signature.update(path.read_bytes())
        model = PPOModel.load(path)
        if model.episodes_trained < 1:
            raise ValueError(f"Untrained model: {path}")
        expected_critic_length = (model.obs_dim if model.method == "ippo"
                                  else model.global_dim)
        if (set(model.critics) != set(model.junction_ids)
                or any(len(weights) != expected_critic_length
                       for weights in model.critics.values())):
            raise ValueError(f"Incompatible value-estimator layout in {path}")
        model_entries.append((entry, model))
    rows, crossing_rows, event_rows = [], [], []
    action_rows, phase_rows = [], []
    group_dir = output / "groups"
    group_dir.mkdir(exist_ok=True)
    fingerprint = signature.hexdigest()
    trace_scenarios = {settings["primary_scenario"], "short-spacing"} & set(specs)
    for scenario, (overrides, slow) in specs.items():
        for seed in seeds:
            group_path = group_dir / f"{scenario}-seed-{seed}.json.gz"
            legacy_path = group_dir / f"{scenario}-seed-{seed}.json"
            existing_group = group_path if group_path.exists() else legacy_path
            if existing_group.exists():
                if existing_group.suffix == ".gz":
                    with gzip.open(existing_group, "rt", encoding="utf-8") as file:
                        previous = json.load(file)
                else:
                    previous = json.loads(existing_group.read_text(encoding="utf-8"))
                if previous["fingerprint"] != fingerprint:
                    raise ValueError(
                        f"Existing group {existing_group} used different settings or models")
                rows.extend(previous["rows"])
                crossing_rows.extend(previous["crossings"])
                event_rows.extend(previous["events"])
                action_rows.extend(previous["actions"])
                phase_rows.extend(previous["phases"])
                print(f"reused {scenario} seed {seed}", flush=True)
                continue
            starts = tuple(len(items) for items in
                           (rows, crossing_rows, event_rows, action_rows, phase_rows))
            # One group owns these read-only files; controller episodes write
            # their tripinfo and metrics into separate short-lived directories.
            prepared_scope = tempfile.TemporaryDirectory(
                prefix="evaluation-inputs-", dir=output)
            try:
                prepared = prepare_inputs(
                    Path(prepared_scope.name), seed,
                    ROOT / "config" / "milestone.json", junctions=3,
                    slow_pedestrians=slow, config_overrides=overrides)
                demand_hash = None
                for label in BASELINES:
                    with tempfile.TemporaryDirectory(prefix="evaluation-", dir=output) as scratch:
                        destination = Path(scratch)
                        metrics = run(label, seed, destination,
                                      ROOT / "config" / "milestone.json", junctions=3,
                                      slow_pedestrians=slow, config_overrides=overrides,
                                      prepared_inputs=prepared)
                        events = json.loads((destination / "crossing_events.json").read_text(
                            encoding="utf-8"))
                    demand_hash = metrics["demand_sha256"] if demand_hash is None else demand_hash
                    if metrics["demand_sha256"] != demand_hash:
                        raise AssertionError("Controllers received mismatched demand")
                    run_row, crossings, event_items = run_rows(
                        metrics, events, scenario, seed, label, None, None)
                    rows.append(run_row)
                    crossing_rows.extend(crossings)
                    event_rows.extend(event_items)
                    if scenario in trace_scenarios and seed == trace_seed:
                        for tls, history in metrics["signal_history"].items():
                            for second, stage in history:
                                phase_rows.append({
                                    "scenario": scenario, "seed": seed,
                                    "controller": label, "training_run": "",
                                    "junction": tls, "second": second, "stage": stage})
                    print(f"evaluated {scenario} seed {seed} {label}", flush=True)
                for entry, model in model_entries:
                    record_actions = (scenario in trace_scenarios
                                      and seed == trace_seed
                                      and entry["training_run"] == 0)
                    step_index = defaultdict(int)
                    def policy(tls, obs, global_obs, eligible):
                        step_index[tls] += 1
                        selection = model.select(tls, obs, global_obs, eligible, training=False)
                        if record_actions and eligible:
                            action, logp, _ = selection
                            probability = math.exp(logp) if action else 1 - math.exp(logp)
                            action_rows.append({
                                "scenario": scenario, "seed": seed,
                                "controller": entry["controller"],
                                "training_run": entry["training_run"],
                                "junction": tls, "step_index": step_index[tls],
                                "local_observation": json.dumps(obs),
                                "request_switch": action,
                                "switch_probability": probability,
                            })
                        return selection
                    with tempfile.TemporaryDirectory(prefix="evaluation-", dir=output) as scratch:
                        destination = Path(scratch)
                        metrics = run(
                            "policy", seed, destination,
                            ROOT / "config" / "milestone.json", junctions=3,
                            slow_pedestrians=slow, config_overrides=overrides,
                            policy_callback=policy,
                            objective=entry["objective"],
                            pedestrian_reward_weight=entry["reward_weight"],
                            prepared_inputs=prepared)
                        events = json.loads((destination / "crossing_events.json").read_text(
                            encoding="utf-8"))
                    if metrics["demand_sha256"] != demand_hash:
                        raise AssertionError("Controllers received mismatched demand")
                    run_row, crossings, event_items = run_rows(
                        metrics, events, scenario, seed, entry["controller"],
                        entry["training_run"], entry["training_seed"])
                    rows.append(run_row)
                    crossing_rows.extend(crossings)
                    event_rows.extend(event_items)
                    if record_actions:
                        for tls, history in metrics["signal_history"].items():
                            for second, stage in history:
                                phase_rows.append({
                                    "scenario": scenario, "seed": seed,
                                    "controller": entry["controller"],
                                    "training_run": entry["training_run"],
                                    "junction": tls, "second": second, "stage": stage})
                    print(f"evaluated {scenario} seed {seed} {entry['controller']}"
                          f" replica {entry['training_run']}", flush=True)
            finally:
                prepared_scope.cleanup()
            payload = {
                "fingerprint": fingerprint,
                "rows": rows[starts[0]:],
                "crossings": crossing_rows[starts[1]:],
                "events": event_rows[starts[2]:],
                "actions": action_rows[starts[3]:],
                "phases": phase_rows[starts[4]:],
            }
            temporary = group_path.with_suffix(".gz.tmp")
            with gzip.open(temporary, "wt", encoding="utf-8",
                           compresslevel=6) as file:
                json.dump(payload, file, separators=(",", ":"))
            temporary.replace(group_path)
    if group_only:
        return {"groups_completed": len(specs) * len(seeds),
                "fingerprint": fingerprint}
    observed_versions = {row["sumo_version"] for row in rows}
    if len(observed_versions) != 1:
        raise ValueError("Evaluation groups used different SUMO versions")
    _write_csv(output / "runs.csv", rows)
    _write_csv(output / "crossings.csv", crossing_rows)
    _write_csv(output / "crossing_events.csv", event_rows)
    _write_csv(output / "action_decisions.csv", action_rows)
    _write_csv(output / "phase_history.csv", phase_rows)
    controllers = list(BASELINES) + sorted({entry["controller"]
                                           for entry, _ in model_entries})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["controller"])].append(row)
    summary, paired = {}, {}
    for scenario in specs:
        summary[scenario], paired[scenario] = {}, {}
        reference = grouped[(scenario, "actuated")]
        for label in controllers:
            group = grouped[(scenario, label)]
            summary[scenario][label] = {
                measure: hierarchical_interval(
                    group, measure, settings["bootstrap_replicates"],
                    settings["bootstrap_seed"])
                for measure in MEASURES}
            if label != "actuated":
                paired[scenario][label] = {
                    measure: hierarchical_interval(
                        group, measure, settings["bootstrap_replicates"],
                        settings["bootstrap_seed"], reference)
                    for measure in MEASURES}
    primary = settings["primary_scenario"]
    strict = {}
    crossing_comparisons = []
    if primary in specs:
        baseline_crossings = {row["crossing"] for row in crossing_rows
                              if row["scenario"] == primary
                              and row["controller"] == "actuated"}
        expected_count = settings["expected_primary_crossings"]
        if len(baseline_crossings) != expected_count:
            raise ValueError(f"Primary scenario has {len(baseline_crossings)} "
                             f"crossings; expected {expected_count}")
        crossing_sets = defaultdict(set)
        for row in crossing_rows:
            if row["scenario"] == primary:
                crossing_sets[(row["seed"], row["controller"],
                               row["training_run"])].add(row["crossing"])
        incomplete = [key for key, crossing_set in crossing_sets.items()
                      if crossing_set != baseline_crossings]
        if incomplete:
            raise ValueError(f"Primary crossing IDs differ across runs: "
                             f"{incomplete[:3]}")
        for label in controllers:
            if label == "actuated":
                continue
            training_runs = sorted({int(row["training_run"]) for row in
                                    grouped[(primary, label)]
                                    if row["training_run"] != ""}) or [-1]
            comparisons, maximum = crossing_comparison(
                event_rows, primary, label, seeds, training_runs,
                settings["bootstrap_replicates"], settings["bootstrap_seed"],
                settings["minimum_pooled_crossing_entries"],
                baseline_crossings)
            crossing_comparisons.extend(comparisons)
            differences = paired[primary][label]
            point_pass, robust_pass = primary_criterion(
                differences, maximum, comparisons)
            strict[label] = {
                "descriptive_criterion_met": point_pass,
                "statistically_supported": robust_pass,
                "maximum_crossing_p95_difference": maximum,
                "crossings_with_higher_p95": [
                    row["crossing"] for row in comparisons
                    if row["p95_difference_seconds"] is not None
                    and row["p95_difference_seconds"] > 0],
                "crossings_with_more_unserved": [
                    row["crossing"] for row in comparisons
                    if row["mean_model_unserved"] > row["actuated_unserved"]],
            }
    overload = None
    if "vehicle-overload" in summary:
        fixed = summary["vehicle-overload"]["fixed"]
        scheduled = fixed["scheduled"]["mean"]
        missing = fixed["unfinished"]["mean"] + fixed["uninserted"]["mean"]
        overload = {
            "fixed_fraction_unfinished_or_uninserted": (
                missing / scheduled if scheduled else None),
            "target_at_least_five_percent_met": bool(
                scheduled and missing / scheduled >= 0.05),
        }
    _write_csv(output / "crossing_comparison.csv", crossing_comparisons)
    result = {
        "settings": settings, "training_manifest": manifest,
        "source_sha256": source_sha256,
        "sumo_version": next(iter(observed_versions)),
        "evaluation_seeds": seeds, "scenarios": list(specs),
        "controllers": controllers, "summary": summary,
        "paired_vs_actuated": paired, "primary_crossing_criterion": strict,
        "overload_check": overload,
        "note": "Hierarchical bootstrap resamples training initialisations and demand seeds. "
        "The crossing test uses pooled event waits within each training replica and a "
        "paired max-across-crossings interval. Final test seeds must remain untouched "
        "during method or checkpoint selection.",
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny pilot using the pilot model directory")
    args = parser.parse_args()
    evaluate(args.output, args.seeds, args.scenarios, args.model_dir,
             smoke=args.smoke)


if __name__ == "__main__":
    main()
