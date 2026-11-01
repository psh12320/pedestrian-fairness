"""Executable measurement, load and warm-up audits; never tune on test seeds.

python -m src.traffic_control.fairness.audit --workers 2
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import tempfile
import time

from .analysis import summarise, quantile
from .environment import run
from .events import EventTracker
from ..scenario import ROOT


def sampling_audit(root):
    trackers = {}
    samples_for_review = []
    def observe(now, samples, endpoints):
        if not trackers:
            trackers.update(fine=EventTracker(endpoints), coarse=EventTracker(endpoints))
        trackers["fine"].observe(now, samples)
        if round(now * 10) % 2 == 0:
            trackers["coarse"].observe(now, samples)
        # Compact, hand-inspectable trajectory for the first two scheduled people.
        for person in ("p000000", "p000001", "p000002"):
            if person in samples:
                s = samples[person]
                if s.next_edge in endpoints or s.road in endpoints:
                    samples_for_review.append({"time": now, "person": person,
                        "road": s.road, "next_edge": s.next_edge,
                        "position": s.position, "speed": s.speed})
    cfg = {"step_seconds": .1, "end_seconds": 360, "warmup_seconds": 0,
           "measurement_end_seconds": 240, "demand_end_seconds": 90,
           "vehicle_rate_per_hour": 720, "pedestrian_rate_per_hour": 720,
           "burst_count": 20, "burst_start_seconds": 20}
    with tempfile.TemporaryDirectory(dir=root) as scratch:
        metrics = run("actuated", 21011, Path(scratch), overrides=cfg, sample_sink=observe)
    def keyed(tracker):
        return {(e["person_id"], e["crossing_id"], e["visit"]): e for e in tracker.rows()}
    fine, coarse = (keyed(trackers[k]) for k in ("fine", "coarse"))
    differences, interval_misses = [], []
    for key in sorted(fine.keys() & coarse.keys()):
        a, b = fine[key], coarse[key]
        if a["wait_seconds"] is not None and b["wait_seconds"] is not None:
            differences.append(abs(a["wait_seconds"] - b["wait_seconds"]))
            if a["wait_upper_seconds"] < b["wait_lower_seconds"] - 1e-6 or b["wait_upper_seconds"] < a["wait_lower_seconds"] - 1e-6:
                interval_misses.append(list(key))
    result = {"definition": "Same SUMO trajectory at 0.1 s, observed at 0.1 and 0.2 s; not two independently simulated trajectories",
        "matched_observed_visits": len(differences), "unmatched_visit_count": len(fine.keys() ^ coarse.keys()),
        "unknown_fine": sum(e["wait_seconds"] is None and e["entered_at"] is not None for e in fine.values()),
        "unknown_coarse": sum(e["wait_seconds"] is None and e["entered_at"] is not None for e in coarse.values()),
        "maximum_absolute_wait_difference": max(differences, default=None),
        "p95_absolute_wait_difference": quantile(differences), "nonoverlapping_intervals": interval_misses,
        "fine_events": list(fine.values()), "coarse_events": list(coarse.values()),
        "review_trajectories": samples_for_review, "endpoints": trackers["fine"].endpoints,
        "sumo_version": metrics["sumo_version"]}
    (root / "sampling.json").write_text(json.dumps(result, indent=2))
    return {k: v for k, v in result.items() if k not in ("fine_events", "coarse_events", "review_trajectories", "endpoints")}


def load_job(job):
    root, rate, seed = job
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=root) as scratch:
        directory = Path(scratch)
        metrics = run("fixed", seed, directory, overrides={"vehicle_rate_per_hour": rate})
        events = json.loads((directory / "events.json").read_text())
        result = summarise(metrics, events)
    v = result["vehicles"]
    result.update(rate=rate, seed=seed, wall_seconds=time.perf_counter() - started,
        service_gap_fraction=(v["unfinished_count"] + v["uninserted_count"]) / max(1, v["scheduled_count"]),
        sumo_version=metrics["sumo_version"], source_sha256=metrics["source_sha256"],
        warmup_sensitivity={str(start): summarise({**metrics, "config": {**metrics["config"], "warmup_seconds": start}}, events)["crossings"]
                           for start in (0, 300, 600)})
    (Path(root) / f"load-{rate}-{seed}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/fairness-v2-audit")
    parser.add_argument("--sampling-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sampling = sampling_audit(args.output)
    print(json.dumps(sampling), flush=True)
    if args.sampling_only:
        return
    jobs = [(str(args.output), rate, seed) for rate in (960, 1920, 2880, 8000) for seed in (21021, 21022)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        loads = list(pool.map(load_job, jobs))
    summary = {"sampling": sampling, "loads": [{k: r[k] for k in ("rate", "seed", "wall_seconds", "service_gap_fraction")} for r in loads],
        "overload_at_least_5_percent_each_seed": all(r["service_gap_fraction"] >= .05 for r in loads if r["rate"] == 8000),
        "median_episode_wall_seconds": statistics.median(r["wall_seconds"] for r in loads),
        "limitations": "Two exploratory seeds. Warm-up sensitivity compares event cohorts within the same trajectory, not convergence to stationarity. Arrival rates are synthetic, not LTA observations."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
