"""Compare 0.1-second and one-second observations of one SUMO trajectory.

This is a development measurement audit, not an additional controller trial.
Both trackers observe the same fine-step trajectory and recorded signal replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

from .cli import run
from .metrics import CrossingTracker
from .scenario import ROOT, executable
from .viewer import write_replay


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(output: Path, horizon: int = 180) -> dict:
    import traci

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resolution-", dir=output) as scratch:
        source = Path(scratch) / "source"
        metrics = run("fixed", 6001, source, ROOT / "config" / "milestone.json",
                      junctions=3, config_overrides={"vehicle_count_per_axis": 3,
                          "pedestrian_count": 24, "vehicle_depart_last_second": 60,
                          "pedestrian_depart_last_second": 60})
        _, replay = write_replay(source, Path(scratch) / "replay")
        crossing_ids = set(metrics["crossing_lengths_metres"])
        fine, coarse = CrossingTracker(crossing_ids, 0.1), CrossingTracker(crossing_ids, 1.0)
        traces = []
        traci.start([executable("sumo"), "--net-file", str(source / "connected.net.xml"),
                     "--route-files", str(source / "demand.rou.xml"),
                     "--additional-files", str(replay), "--step-length", "0.1",
                     "--end", str(horizon), "--seed", "6001",
                     "--pedestrian.striping.dawdling", "0", "--time-to-teleport", "-1",
                     "--no-step-log", "true"])
        try:
            ends = {crossing: (traci.lane.getShape(crossing + "_0")[0],
                               traci.lane.getShape(crossing + "_0")[-1])
                    for crossing in crossing_ids}
            for step in range(1, horizon * 10 + 1):
                traci.simulationStep()
                now = round(traci.simulation.getTime(), 1)
                observations = {}
                for person in traci.person.getIDList():
                    road = traci.person.getRoadID(person)
                    following = traci.person.getNextEdge(person)
                    stopped = traci.person.getWaitingTime(person)
                    distance = None
                    if following in crossing_ids:
                        position = traci.person.getPosition(person)
                        distance = min(math.dist(position, end) for end in ends[following])
                    observations[person] = (road, following, stopped, distance)
                    crossing = road if road in crossing_ids else following
                    if crossing in crossing_ids:
                        traces.append({"time": now, "person_id": person,
                                       "crossing_id": crossing, "road": road,
                                       "next_edge": following, "stopped_seconds": stopped,
                                       "endpoint_distance_metres": distance,
                                       "sampled_by_one_second_tracker": step % 10 == 0})
                fine.observe(now, observations)
                if step % 10 == 0:
                    coarse.observe(now, observations)
            version = traci.getVersion()[1]
        finally:
            traci.close()

    comparisons = []
    for key in sorted(set(fine.events) | set(coarse.events)):
        a, b = fine.events.get(key), coarse.events.get(key)
        both_entered = a is not None and b is not None and a.wait is not None and b.wait is not None
        comparisons.append({
            "person_id": key[0], "crossing_id": key[1],
            "fine_arrival": a.reached_at if a else None,
            "coarse_arrival": b.reached_at if b else None,
            "fine_entry": a.entered_at if a else None,
            "coarse_entry": b.entered_at if b else None,
            "fine_wait": a.wait if a else None, "coarse_wait": b.wait if b else None,
            "wait_difference_coarse_minus_fine": b.wait - a.wait if both_entered else None,
            "fine_source": a.arrival_source if a else "undetected",
            "coarse_source": b.arrival_source if b else "undetected",
            "fine_pending": a is not None and a.entered_at is None,
            "coarse_pending": b is not None and b.entered_at is None,
        })
    errors = [row["wait_difference_coarse_minus_fine"] for row in comparisons
              if row["wait_difference_coarse_minus_fine"] is not None]
    direct_free = [row["wait_difference_coarse_minus_fine"] for row in comparisons
                   if row["wait_difference_coarse_minus_fine"] is not None
                   and row["fine_source"] != "entry_without_approach"
                   and row["coarse_source"] != "entry_without_approach"]
    selected = {}
    for category, predicate in (
        ("immediate_entry", lambda row: row["fine_wait"] is not None and row["fine_wait"] <= 1),
        ("queued_wait", lambda row: row["fine_wait"] is not None and row["fine_wait"] >= 5),
        ("pending_at_horizon", lambda row: row["fine_pending"]),
    ):
        selected[category] = next((row for row in comparisons if predicate(row)), None)
    hand_trace = []
    for category, row in selected.items():
        if row is None:
            continue
        boundaries = [value for value in (row["fine_arrival"], row["fine_entry"],
                                          horizon if row["fine_pending"] else None)
                      if value is not None]
        hand_trace.extend({"case": category, **trace} for trace in traces
                          if (trace["person_id"], trace["crossing_id"]) ==
                          (row["person_id"], row["crossing_id"])
                          and any(abs(trace["time"] - boundary) <= 1.1 for boundary in boundaries))
    result = {
        "purpose": "Measurement sampling audit on one shared fine-step trajectory",
        "seed": 6001, "horizon_seconds": horizon, "sumo_version": version,
        "demand_sha256": metrics["demand_sha256"],
        "experiment_config": metrics["config"],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("src/traffic_control/measurement_resolution.py",
                              "src/traffic_control/cli.py", "src/traffic_control/metrics.py",
                              "src/traffic_control/signals.py", "src/traffic_control/scenario.py",
                              "src/traffic_control/connected.py", "src/traffic_control/viewer.py",
                              "config/milestone.json")},
        "fine_step_seconds": 0.1, "coarse_step_seconds": 1.0,
        "matched_entered_events": len(errors),
        "mean_signed_wait_error_seconds": statistics.mean(errors) if errors else None,
        "mean_absolute_wait_error_seconds": statistics.mean(map(abs, errors)) if errors else None,
        "maximum_absolute_wait_error_seconds": max(map(abs, errors), default=None),
        "maximum_absolute_error_with_both_approaches_observed_seconds":
            max(map(abs, direct_free), default=None),
        "fine_event_count": len(fine.events), "coarse_event_count": len(coarse.events),
        "fine_pending": sum(row["fine_pending"] for row in comparisons),
        "coarse_pending": sum(row["coarse_pending"] for row in comparisons),
        "fine_direct_imputations": sum(row["fine_source"] == "entry_without_approach" for row in comparisons),
        "coarse_direct_imputations": sum(row["coarse_source"] == "entry_without_approach" for row in comparisons),
        "hand_check_cases": selected,
        "limitations": "One development trajectory; 0.1-second operational events are a finer reference, not physical ground truth. This isolates observation downsampling, not SUMO step-length effects on dynamics or the 1.2 m entrance definition.",
    }
    if not errors or any(value < -1e-8 for row in comparisons
                         for value in (row["fine_wait"], row["coarse_wait"]) if value is not None):
        raise AssertionError("No comparable entries or a negative crossing wait")
    write_csv(output / "event_comparison.csv", comparisons)
    write_csv(output / "hand_trace.csv", hand_trace)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "measurement-resolution")
    parser.add_argument("--horizon", type=int, default=180)
    args = parser.parse_args()
    result = audit(args.output, args.horizon)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
