"""Measurement cohorts and paired descriptive comparisons; no zero imputation."""
import math
import statistics
import random


def quantile(values, p=.95):
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)] if values else None


def summarise(metrics, events):
    cfg = metrics["config"]
    start, end = cfg["warmup_seconds"], cfg["measurement_end_seconds"]
    crossings = {}
    for crossing in metrics["pedestrians_by_crossing_end"]:
        rows = [e for e in events if e["crossing_id"] == crossing and e["reached_at"] is not None
                and start <= e["reached_at"] < end]
        ambiguous = [e for e in events if e["crossing_id"] == crossing and e["reached_at"] is None
                     and e["entered_at"] >= start]
        waits = [e["wait_seconds"] for e in rows if e["wait_seconds"] is not None]
        crossings[crossing] = {"waits": waits, "mean": statistics.mean(waits) if waits else None,
            "wait_lower": [e["wait_lower_seconds"] for e in rows if e["wait_seconds"] is not None],
            "wait_upper": [e["wait_upper_seconds"] for e in rows if e["wait_seconds"] is not None],
            "speed_groups": {speed: {"waits": [e["wait_seconds"] for e in rows if e["speed_group"] == speed and e["wait_seconds"] is not None],
                "pending": sum(e["entered_at"] is None for e in rows if e["speed_group"] == speed)}
                for speed in sorted({e["speed_group"] for e in rows})},
            "p95": quantile(waits), "entries": len(waits),
            "pending_at_cutoff": sum(e["entered_at"] is None or e["entered_at"] >= end for e in rows),
            "unresolved_after_followup": sum(e["entered_at"] is None for e in rows),
            "unknown_cohort_entries": len(ambiguous), "lost": sum(e["lost"] for e in rows)}
    return {"vehicles": metrics.get("vehicles_measurement_cohort", metrics["vehicles_at_measurement_cutoff"]), "crossings": crossings,
            "persons": metrics["persons"], "demand_sha256": metrics["demand_sha256"]}


def selection_score(result, baseline, pedestrian=True):
    v, b = result["vehicles"], baseline["vehicles"]
    service = max(0, b["completed_count"] - v["completed_count"])
    service += max(0, v["unfinished_count"] + v["uninserted_count"] - b["unfinished_count"] - b["uninserted_count"])
    service += max(0, v["uninserted_count"] - b["uninserted_count"])
    service += max(0, v["unfinished_count"] - b["unfinished_count"])
    if "persons" in result and "persons" in baseline:
        p, q = result["persons"], baseline["persons"]
        service += max(0, q.get("completed", 0) - p.get("completed", 0))
        service += max(0, p.get("unfinished", 0) - q.get("unfinished", 0))
        service += max(0, p.get("uninserted", 0) - q.get("uninserted", 0))
    differences = []
    for c, value in result["crossings"].items():
        ref = baseline["crossings"][c]
        service += max(0, value["pending_at_cutoff"] - ref["pending_at_cutoff"])
        service += max(0, value["unresolved_after_followup"] - ref["unresolved_after_followup"])
        service += value["unknown_cohort_entries"] + value["lost"] + ref["unknown_cohort_entries"] + ref["lost"]
        if value["p95"] is None or ref["p95"] is None:
            differences.append(1e6)
        else:
            differences.append(value["p95"] - ref["p95"])
    loss = v["completed_mean_time_loss_seconds"]
    return (service, max(0, max(differences)) if pedestrian else 0, loss if loss is not None else 1e6)


def primary_inference(groups, bootstrap=1000):
    groups = sorted([g for g in groups if g["scenario"] == "vmedium-pmedium"], key=lambda g: g["seed"])
    baselines = [next(r["result"] for r in g["records"] if r["controller"] == "actuated") for g in groups]
    labels = sorted({r["controller"] for g in groups for r in g["records"] if r["replica"] is not None})
    if not groups:
        return {}
    crossings = sorted(baselines[0]["crossings"])
    output = {}
    for label in labels:
        replicas = sorted({r["replica"] for r in groups[0]["records"] if r["controller"] == label})
        models = [[next(r["result"] for r in g["records"] if r["controller"] == label and r["replica"] == replica)
                   for g in groups] for replica in replicas]
        minimum = min([sum(b["crossings"][c]["entries"] for b in baselines) for c in crossings] +
                      [sum(m["crossings"][c]["entries"] for m in replica) for replica in models for c in crossings])
        def estimates(seeds, selected_replicas):
            tails = []
            for c in crossings:
                baseline_tail = quantile([w for s in seeds for w in baselines[s]["crossings"][c]["waits"]])
                replica_tails = [quantile([w for s in seeds for w in models[r][s]["crossings"][c]["waits"]])
                                 for r in selected_replicas]
                if baseline_tail is None or any(t is None for t in replica_tails):
                    return None
                tails.append(statistics.mean(replica_tails) - baseline_tail)
            vehicle_differences = []
            for r in selected_replicas:
                for s in seeds:
                    a = models[r][s]["vehicles"]["completed_mean_time_loss_seconds"]
                    b = baselines[s]["vehicles"]["completed_mean_time_loss_seconds"]
                    if a is None or b is None:
                        return None
                    vehicle_differences.append(a - b)
            return statistics.mean(vehicle_differences), tails
        point = estimates(list(range(len(groups))), list(range(len(replicas))))
        service_ok = all(selection_score(models[r][s], baselines[s])[0] == 0
                         for r in range(len(replicas)) for s in range(len(groups)))
        row = {"minimum_crossing_entries": minimum, "service_and_measurement_ok": service_ok,
               "point_estimates": point, "descriptive_pass": bool(point and point[0] < 0 and max(point[1]) <= 0 and service_ok),
               "crossing_order": crossings, "statistical_support": False}
        # One predeclared confirmatory model; no post-hoc best-model promotion.
        if label == "mappo-constrained" and len(groups) >= 20 and len(replicas) >= 5 and minimum >= 300 and point:
            rng = random.Random(42462026)
            vehicle, centered_max = [], []
            for _ in range(bootstrap):
                sample = estimates([rng.randrange(len(groups)) for _ in groups],
                                   [rng.randrange(len(replicas)) for _ in replicas])
                if sample is None:
                    continue
                vehicle.append(sample[0])
                centered_max.append(max(a - b for a, b in zip(sample[1], point[1])))
            if len(vehicle) == bootstrap:
                upper_vehicle = quantile(vehicle, .975)
                upper_crossing = max(point[1]) + quantile(centered_max, .975)
                row.update(vehicle_upper_97_5=upper_vehicle, simultaneous_crossing_upper_97_5=upper_crossing,
                           statistical_support=bool(service_ok and upper_vehicle < 0 and upper_crossing <= 0))
        output[label] = row
    return output
