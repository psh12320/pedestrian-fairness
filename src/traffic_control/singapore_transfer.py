"""Repeated transfer check on an uncalibrated NUS-derived synthetic corridor.

This is separate from the locked synthetic evaluation. Only approximate site
centroids and spacing come from the Singapore map audit; route and pedestrian
demand, connecting roads, crossings, and signal plans are project assumptions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from .cli import prepare_inputs, run
from .evaluate import (crossing_comparison, hierarchical_interval, percentile,
                       primary_criterion, run_rows)
from .scenario import ROOT
from .singapore import derived_case_spec, load_transfer_models, prepare
from .validation_decision import check_selected_budget


BASELINES = ("fixed", "actuated", "pressure")
MEASURE_NAMES = ("scheduled", "completed", "unfinished", "uninserted",
                 "vehicle_time_loss", "pedestrian_mean_wait",
                 "pedestrian_p95_wait", "unserved_requests")
SOURCE_FILES = (
    "config/milestone.json", "config/training.json",
    "src/traffic_control/cli.py", "src/traffic_control/connected.py",
    "src/traffic_control/scenario.py", "src/traffic_control/signals.py",
    "src/traffic_control/metrics.py", "src/traffic_control/ppo.py",
    "src/traffic_control/train.py", "src/traffic_control/evaluate.py",
    "src/traffic_control/singapore.py", "src/traffic_control/singapore_transfer.py",
    "src/traffic_control/validation_decision.py",
)


def load_settings(path: Path, smoke: bool = False) -> tuple[dict, list[int]]:
    settings = json.loads(path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in settings["seeds"]]
    if len(seeds) != 10 or len(set(seeds)) != 10 or any(seed < 1 for seed in seeds):
        raise ValueError("The transfer protocol requires ten distinct positive seeds")
    other = json.loads((ROOT / "config" / "evaluation.json").read_text(
        encoding="utf-8"))
    reserved = set(other["seeds"]) | set(other["exploratory_seeds"])
    if set(seeds) & reserved:
        raise ValueError("Transfer test seeds must be separate from main evaluation seeds")
    if settings["bootstrap_replicates"] < 100 or settings["expected_crossings"] != 12:
        raise ValueError("Transfer protocol needs at least 100 bootstrap draws and 12 crossings")
    return settings, [int(settings["exploratory_seed"])] if smoke else seeds


def fingerprint(settings: dict, site_audit: dict, model_dir: Path,
                models: list[dict], decision: dict | None) -> tuple[str, dict[str, str]]:
    source_sha = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in SOURCE_FILES}
    source_sha[settings["case_config"]] = hashlib.sha256(
        (ROOT / settings["case_config"]).read_bytes()).hexdigest()
    source_sha[settings["derived_config"]] = hashlib.sha256(
        (ROOT / settings["derived_config"]).read_bytes()).hexdigest()
    manifest = json.loads((model_dir / "training_manifest.json").read_text(
        encoding="utf-8"))
    payload = {"settings": settings, "site_audit": site_audit,
               "manifest": manifest, "source_sha256": source_sha,
               "models": {item["label"]: item["model_sha256"] for item in models},
               "validation_decision": decision}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest(), source_sha


def selected_model_budget(model_dir: Path,
                          decision_path: Path = ROOT / "runs" / "training-v3" /
                          "extension_decision.json") -> tuple[int, dict]:
    """Reject a transfer that uses a nonselected or incomplete training budget."""
    if not decision_path.is_file():
        raise FileNotFoundError(
            f"Validation-only training decision is missing: {decision_path}")
    manifest = json.loads((model_dir / "training_manifest.json").read_text(
        encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    return check_selected_budget(manifest, decision), decision


def aggregate_crossings(events: list[dict], rows: list[dict],
                        crossings: set[str]) -> list[dict]:
    """Keep an explicit entry denominator and pending count for every crossing."""
    observations: dict[tuple[str, int, str], dict] = defaultdict(
        lambda: {"waits": [], "pending": 0})
    run_ids: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        run_ids[row["controller"]].add(
            -1 if row["training_run"] in ("", None) else int(row["training_run"]))
    for event in events:
        run_id = (-1 if event["training_run"] in ("", None)
                  else int(event["training_run"]))
        group = observations[(event["controller"], run_id, event["crossing_id"])]
        if event["wait_seconds"] in ("", None):
            group["pending"] += 1
        else:
            group["waits"].append(float(event["wait_seconds"]))
    result = []
    for label in sorted(run_ids):
        for crossing in sorted(crossings):
            groups = [observations[(label, run_id, crossing)]
                      for run_id in sorted(run_ids[label])]
            means = [statistics.mean(group["waits"]) if group["waits"] else None
                     for group in groups]
            p95s = [percentile(group["waits"], 0.95) for group in groups]
            result.append({
                "controller": label, "crossing": crossing,
                "training_replicas": len(groups),
                "mean_entries_per_replica": statistics.mean(
                    len(group["waits"]) for group in groups),
                "minimum_entries_per_replica": min(
                    len(group["waits"]) for group in groups),
                "mean_wait_seconds": (
                    statistics.mean(means) if all(value is not None for value in means)
                    else None),
                "mean_p95_wait_seconds": (
                    statistics.mean(p95s) if all(value is not None for value in p95s)
                    else None),
                "mean_pending_requests_per_replica": statistics.mean(
                    group["pending"] for group in groups),
            })
    return result


def write_csv(path: Path, rows: list[dict], compressed: bool = False) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(settings: dict, seeds: list[int], models: list[dict],
                  rows: list[dict], crossing_rows: list[dict], events: list[dict],
                  site_audit: dict, selected_site: dict, source_sha: dict[str, str],
                  study_fingerprint: str) -> tuple[dict, list[dict], list[dict]]:
    scenario = settings["scenario"]
    labels = list(BASELINES) + sorted({
        item["label"].rsplit("-r", 1)[0] for item in models})
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["controller"]].append(row)
    expected = {label: len(seeds) * (1 if label in BASELINES else
                                    len({item["label"].rsplit("-r", 1)[1]
                                         for item in models
                                         if item["label"].startswith(label + "-r")}))
                for label in labels}
    for label in labels:
        if len(grouped[label]) != expected[label]:
            raise ValueError(f"Incomplete {label} transfer runs: "
                             f"{len(grouped[label])} of {expected[label]}")
    references = grouped["actuated"]
    summaries, paired, criterion = {}, {}, {}
    comparison_rows = []
    crossing_ids = {row["crossing"] for row in crossing_rows
                    if row["controller"] == "actuated"}
    if len(crossing_ids) != settings["expected_crossings"]:
        raise ValueError(f"Expected {settings['expected_crossings']} crossings, "
                         f"observed {len(crossing_ids)}")
    for row in crossing_rows:
        if row["crossing"] not in crossing_ids:
            raise ValueError("Crossing IDs differ across transfer runs")
    for label in labels:
        summaries[label] = {
            measure: hierarchical_interval(grouped[label], measure,
                                           settings["bootstrap_replicates"],
                                           settings["bootstrap_seed"])
            for measure in MEASURE_NAMES}
        if label == "actuated":
            continue
        paired[label] = {
            measure: hierarchical_interval(grouped[label], measure,
                                           settings["bootstrap_replicates"],
                                           settings["bootstrap_seed"], references)
            for measure in MEASURE_NAMES}
        replica_ids = sorted({int(row["training_run"]) for row in grouped[label]
                              if row["training_run"] not in ("", None)}) or [-1]
        comparisons, maximum = crossing_comparison(
            events, scenario, label, seeds, replica_ids,
            settings["bootstrap_replicates"], settings["bootstrap_seed"],
            settings["minimum_pooled_crossing_entries_per_replica"],
            crossing_ids)
        comparison_rows.extend(comparisons)
        point, supported = primary_criterion(paired[label], maximum, comparisons)
        criterion[label] = {
            "descriptive_criterion_met": point,
            "statistically_supported": supported,
            "maximum_crossing_p95_difference": maximum,
            "crossings_with_higher_p95": [row["crossing"] for row in comparisons
                                          if row["p95_difference_seconds"] is not None
                                          and row["p95_difference_seconds"] > 0],
            "crossings_with_more_pending": [row["crossing"] for row in comparisons
                                            if row["mean_model_unserved"] >
                                            row["actuated_unserved"]],
        }
    crossing_summary = aggregate_crossings(events, rows, crossing_ids)
    result = {
        "status": "UNCALIBRATED NUS-DERIVED GEOMETRY TRANSFER; synthetic demand; simulation only",
        "scenario": scenario, "seeds": seeds,
        "replicas_per_learned_configuration": len({
            item["label"].rsplit("-r", 1)[1] for item in models}),
        "run_count": len(rows), "controller_configurations": labels,
        "observed_crossings": len(crossing_ids), "study_fingerprint": study_fingerprint,
        "source_sha256": source_sha,
        "source_osm_xml_sha256": site_audit["osm"]["xml_sha256"],
        "candidate_site_groups": selected_site["signal_groups"],
        "settings": settings,
        "summary": summaries, "paired_vs_actuated": paired,
        "crossing_criterion_vs_actuated": criterion,
        "limits": [
            "Three approximate candidate-site centroids and spacings are map-derived; "
            "site identities, geometry and crossing locations are not field-verified.",
            "Connecting links, side arms, sidewalks, protected crossings and all demand "
            "are synthetic project assumptions; no Singapore turning or pedestrian "
            "counts or deployed signal timing plans were used.",
            "The raw OSM import had no signal-controlled pedestrian crossings, "
            "so the protected derived network is not a reconstructed field junction.",
            "A sparse crossing p95 or incomplete bootstrap makes the every-crossing "
            "criterion inconclusive even if its point estimate appears favorable.",
        ],
    }
    return result, crossing_summary, comparison_rows


def write_report(output: Path, summary: dict) -> None:
    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f}"

    lines = ["# NUS-derived corridor transfer check", "",
             "**Uncalibrated synthetic-demand simulation.** The three approximate "
             "candidate-site centroids and spacings came from a local OSM audit. "
             "The connecting roads, approaches, crossings, routes, pedestrian "
             "arrivals and phase plans are project assumptions. These results "
             "do not estimate an improvement to deployed Singapore signals.", "",
             f"The {'exploratory smoke check used' if summary['smoke'] else 'frozen transfer protocol used'} "
             f"{len(summary['seeds'])} matched demand seed(s), "
             f"{summary['replicas_per_learned_configuration']} independent training "
             f"replicas per PPO configuration, and {summary['run_count']} SUMO runs. "
             "Each controller received the same demand within each seed and used "
             "the shared legal phase and pedestrian-clearance machine.", "",
             "| Controller | Vehicle time loss (s) | Paired difference vs actuated "
             "[95% CI] (s) | Completed | Pending requests | Worst crossing p95 "
             "difference [95% CI] (s) | Descriptive crossing target |", 
             "|---|---:|---:|---:|---:|---:|---|"]
    for label in summary["controller_configurations"]:
        group = summary["summary"][label]
        if label == "actuated":
            vehicle_delta = "reference"
            worst = "reference"
            criterion = "reference"
        else:
            paired = summary["paired_vs_actuated"][label]["vehicle_time_loss"]
            maximum = summary["crossing_criterion_vs_actuated"][label][
                "maximum_crossing_p95_difference"]
            vehicle_delta = (f"{number(paired['mean'])} "
                             f"[{number(paired['ci95_low'])}, {number(paired['ci95_high'])}]")
            worst = (f"{number(maximum['mean'])} "
                     f"[{number(maximum['ci95_low'])}, {number(maximum['ci95_high'])}]")
            criterion = ("yes" if summary["crossing_criterion_vs_actuated"][label][
                "descriptive_criterion_met"] else "no / inconclusive")
        lines.append(f"| {label} | {number(group['vehicle_time_loss']['mean'])} | "
                     f"{vehicle_delta} | {number(group['completed']['mean'])} | "
                     f"{number(group['unserved_requests']['mean'])} | {worst} | {criterion} |")
    lines.extend(["", "Each crossing's entry denominator, mean wait, p95 wait, pending count "
                  "and paired p95 interval is in `crossing_summary.csv` and "
                  "`crossing_comparison.csv`. The complete per-run results are in "
                  "`runs.csv`; raw crossing events are compressed in "
                  "`crossing_events.csv.gz`. A p95 based on fewer than 100 pooled "
                  "entries per training replica is labelled inadequate for a "
                  "strong every-crossing claim.", "",
                  "## Limits", ""])
    lines.extend(f"- {item}" for item in summary["limits"])
    lines.extend(["", "This transfer check is deliberately separate from the "
                  "locked 13-scenario synthetic study. " + (
                      "This smoke run uses the already inspected seed 61 and may use "
                      "models trained under a superseded clearance rule; it is not "
                      "transfer evidence." if summary["smoke"] else
                      "The earlier seed-61 Singapore pilot and its v2 policies "
                      "were exploratory and trained under the superseded clearance "
                      "rule; neither is mixed into this table.")])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_transfer(output: Path, model_dir: Path, settings_path: Path,
                      smoke: bool = False) -> dict:
    settings, seeds = load_settings(settings_path, smoke)
    output.mkdir(parents=True, exist_ok=True)
    site_audit = prepare(ROOT / settings["case_config"], output / "site-audit")
    case, selected, overrides = derived_case_spec(
        site_audit, ROOT / settings["derived_config"])
    models = load_transfer_models(model_dir, all_replicas=not smoke)
    if not smoke and len(models) != settings["expected_full_models"]:
        raise ValueError(f"Expected {settings['expected_full_models']} trained models")
    selected_budget, decision = (None, None) if smoke else selected_model_budget(model_dir)
    study_fingerprint, source_sha = fingerprint(
        settings, site_audit, model_dir, models, decision)
    scenario = settings["scenario"]
    groups = output / "groups"
    groups.mkdir(exist_ok=True)
    rows, crossing_rows, event_rows = [], [], []
    base_config = ROOT / case["base_experiment_config"]
    for seed in seeds:
        group_file = groups / f"seed-{seed}.json.gz"
        if group_file.exists():
            with gzip.open(group_file, "rt", encoding="utf-8") as handle:
                saved = json.load(handle)
            if saved["fingerprint"] != study_fingerprint:
                raise ValueError(f"Existing transfer group has different inputs: {group_file}")
            print(f"reused NUS-derived transfer seed {seed}", flush=True)
        else:
            with tempfile.TemporaryDirectory(prefix="sg-inputs-", dir=output) as folder:
                prepared = prepare_inputs(Path(folder), seed, base_config,
                                          junctions=3, config_overrides=overrides)
                if len(prepared.crossings) != settings["expected_crossings"]:
                    raise ValueError("Derived transfer network changed its crossing count")
                selected_rows, selected_crossings, selected_events = [], [], []
                for label, policy in ([(name, None) for name in BASELINES] +
                                      [(item["label"], item) for item in models]):
                    callback = (None if policy is None else
                                lambda tls, obs, global_obs, eligible,
                                model=policy["model"]: model.select(
                                    tls, obs, global_obs, eligible, training=False))
                    with tempfile.TemporaryDirectory(prefix="sg-run-", dir=output) as scratch:
                        destination = Path(scratch)
                        metrics = run(
                            "policy" if policy else label, seed, destination,
                            base_config, junctions=3, config_overrides=overrides,
                            policy_callback=callback, prepared_inputs=prepared,
                            objective=policy["objective"] if policy else "vehicle",
                            pedestrian_reward_weight=(policy["reward_weight"]
                                                      if policy else 0.0))
                        events = json.loads((destination / "crossing_events.json").read_text(
                            encoding="utf-8"))
                    if metrics["demand_sha256"] != prepared.demand["sha256"]:
                        raise AssertionError("Controllers received different transfer demand")
                    controller = (label.rsplit("-r", 1)[0] if policy else label)
                    replica = (int(label.rsplit("-r", 1)[1]) if policy else None)
                    run_row, crossings, event_items = run_rows(
                        metrics, events, scenario, seed, controller, replica,
                        policy["training_seed"] if policy else None)
                    selected_rows.append(run_row)
                    selected_crossings.extend(crossings)
                    selected_events.extend(event_items)
                    print(f"NUS-derived seed {seed}: {label}", flush=True)
            saved = {"fingerprint": study_fingerprint,
                     "demand_sha256": prepared.demand["sha256"],
                     "network_sha256": prepared.network_sha256,
                     "rows": selected_rows, "crossings": selected_crossings,
                     "events": selected_events}
            temporary = group_file.with_suffix(".gz.tmp")
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(saved, handle, separators=(",", ":"))
            temporary.replace(group_file)
        rows.extend(saved["rows"])
        crossing_rows.extend(saved["crossings"])
        event_rows.extend(saved["events"])
    expected_runs = len(seeds) * (len(BASELINES) + len(models))
    if len(rows) != expected_runs:
        raise ValueError(f"Only {len(rows)} of {expected_runs} transfer runs are present")
    if len({row["sumo_version"] for row in rows}) != 1:
        raise ValueError("Transfer runs used different SUMO versions")
    summary, crossing_summary, comparisons = build_summary(
        settings, seeds, models, rows, crossing_rows, event_rows,
        site_audit, selected, source_sha, study_fingerprint)
    summary["smoke"] = smoke
    if smoke:
        summary["status"] = ("EXPLORATORY SINGLE-SEED MECHANISM CHECK; "
                             "not transfer performance evidence")
    summary["model_dir"] = str(model_dir.resolve())
    summary["selected_training_episodes"] = selected_budget or json.loads(
        (model_dir / "training_manifest.json").read_text(encoding="utf-8"))[
            "episodes_requested"]
    summary["validation_decision"] = decision
    summary["sumo_version"] = rows[0]["sumo_version"]
    write_csv(output / "runs.csv", rows)
    write_csv(output / "crossings.csv", crossing_rows)
    write_csv(output / "crossing_summary.csv", crossing_summary)
    write_csv(output / "crossing_comparison.csv", comparisons)
    write_csv(output / "crossing_events.csv.gz", event_rows, compressed=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path,
                        default=ROOT / "config" / "singapore_transfer.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "v3")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "singapore-transfer")
    parser.add_argument("--smoke", action="store_true",
                        help="Use inspected seed 61 and four replica-zero models only")
    args = parser.parse_args()
    result = evaluate_transfer(args.output, args.model_dir, args.settings, args.smoke)
    print(f"{result['run_count']} matched uncalibrated NUS-derived SUMO runs complete; "
          f"report: {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
