"""Development SUMO runner for v2 safety, measurement and baseline validation.

Run: python -m src.traffic_control.fairness.environment --controller actuated --smoke
Legacy runners and saved result schemas are deliberately not changed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from . import SCHEMA_VERSION
from .demand import make_demand, scenarios
from .events import EventTracker, PersonSample
from .signals import SafeSignal, baseline_action, PHASES
from .control_observation import SCHEMAS, controller_memory
from ..connected import incoming_groups, outgoing_groups
from ..metrics import vehicle_summary
from ..scenario import ROOT, build_network, executable, clearance_seconds
from ..signals import SignalPlan, classify_links, load_signal_conflicts


def run(controller="actuated", seed=21001, output=None, junctions=3,
        overrides=None, policy=None, trace_sink=None, sample_sink=None):
    import traci
    import traci.constants as tc
    if controller not in ("fixed", "actuated", "pressure", "policy"):
        raise ValueError(controller)
    if controller == "policy" and policy is None:
        raise ValueError("A policy callback is required")
    cfg = json.loads((ROOT / "config/fairness_v2.json").read_text())
    cfg.update(overrides or {})
    observation_schema = cfg.get("observation_schema", "legacy-v2")
    if observation_schema not in SCHEMAS:
        raise ValueError("Unknown observation schema: " + str(observation_schema))
    dt = cfg["step_seconds"]
    horizon = cfg["end_seconds"]
    cutoff = cfg["measurement_end_seconds"]
    if dt <= 0 or not 0 <= cfg["warmup_seconds"] < cutoff <= horizon:
        raise ValueError("Invalid time windows")
    for value in (horizon, cutoff, cfg["decision_seconds"]):
        if not math.isclose(value / dt, round(value / dt)):
            raise ValueError("Time boundaries must align to simulation ticks")
    if cfg["decision_seconds"] != 1:
        raise ValueError("Version 2 currently requires one-second control decisions")
    output = Path(output or ROOT / "runs/fairness-v2" / f"{controller}-{seed}")
    output.mkdir(parents=True, exist_ok=True)
    signals, samples, states = {}, {}, {}
    departed, arrived, persons_seen, persons_arrived = set(), set(), set(), set()
    subscribed_people = set()
    cutoff_counts = None
    obs_features = None
    with tempfile.TemporaryDirectory(prefix="episode-", dir=output) as scratch:
        scratch = Path(scratch)
        net, routes, tripinfo = scratch / "network.net.xml", scratch / "demand.rou.xml", scratch / "tripinfo.xml"
        crossings = build_network(net, junctions, cfg["junction_spacing_m"])
        demand = make_demand(routes, seed, cfg, junctions)
        # netconvert embeds generation time and temporary paths in XML comments.
        # Hash the actual network tree, excluding those non-semantic comments.
        network_hash = hashlib.sha256(ET.tostring(ET.parse(net).getroot())).hexdigest()
        people = {p["id"]: p for p in demand["people"]}
        log = output / "sumo.log"
        traci.start([executable("sumo"), "-n", str(net), "-r", str(routes),
                     "--step-length", str(dt), "--seed", str(seed), "--end", str(horizon),
                     "--tripinfo-output", str(tripinfo), "--tripinfo-output.write-unfinished", "true",
                     "--time-to-teleport", "-1", "--no-step-log", "true",
                     "--collision.check-junctions", "true",
                     "--pedestrian.striping.dawdling", "0", "--log", str(log),
                     "--pedestrian.striping.jamtime", str(horizon + 1),
                     "--pedestrian.striping.jamtime.crossing", str(horizon + 1),
                     "--pedestrian.striping.jamtime.narrow", str(horizon + 1)])
        try:
            endpoints = {c: (tuple(traci.lane.getShape(c + "_0")[0]),
                             tuple(traci.lane.getShape(c + "_0")[-1])) for c in crossings}
            tracker = EventTracker(endpoints)
            incoming = incoming_groups(junctions)
            outgoing = outgoing_groups(junctions)
            edge_ids = set().union(*(v for groups in [*incoming.values(), *outgoing.values()] for v in groups.values()))
            for edge in sorted(edge_ids):
                traci.edge.subscribe(edge, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                           tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_MEAN_SPEED])
            traci.simulation.subscribe([tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_IDS,
                                        tc.VAR_ARRIVED_PERSONS_IDS, tc.VAR_COLLIDING_VEHICLES_NUMBER])
            local = {tls: sorted(c for c in crossings if c.startswith(f":{tls}_c")) for tls in incoming}
            for tls in sorted(incoming):
                groups = classify_links(traci.trafficlight.getControlledLinks(tls), set(local[tls]), incoming[tls])
                plan = SignalPlan(groups, len(traci.trafficlight.getRedYellowGreenState(tls)),
                                  clearance_seconds({c: crossings[c] for c in local[tls]}, cfg), cfg,
                                  load_signal_conflicts(net, tls))
                signals[tls] = SafeSignal(plan)
                states[tls] = signals[tls].state()
                traci.trafficlight.setRedYellowGreenState(tls, states[tls])
                traci.trafficlight.subscribe(tls, [tc.TL_RED_YELLOW_GREEN_STATE])
            for tick in range(1, round(horizon / dt) + 1):
                traci.simulationStep()
                now = round(tick * dt, 8)
                simulation = traci.simulation.getSubscriptionResults()
                departed.update(simulation[tc.VAR_DEPARTED_VEHICLES_IDS])
                arrived.update(simulation[tc.VAR_ARRIVED_VEHICLES_IDS])
                persons_arrived.update(simulation[tc.VAR_ARRIVED_PERSONS_IDS])
                ids = traci.person.getIDList()
                persons_seen.update(ids)
                for p in set(ids) - subscribed_people:
                    traci.person.subscribe(p, [tc.VAR_ROAD_ID, tc.VAR_NEXT_EDGE, tc.VAR_POSITION, tc.VAR_SPEED])
                    subscribed_people.add(p)
                samples = {}
                for p in ids:
                    data = traci.person.getSubscriptionResults(p)
                    samples[p] = PersonSample(data[tc.VAR_ROAD_ID], data[tc.VAR_NEXT_EDGE],
                                              tuple(data[tc.VAR_POSITION]), data[tc.VAR_SPEED], people[p]["type"])
                    if cfg.get("subscription_crosscheck"):
                        direct = PersonSample(traci.person.getRoadID(p), traci.person.getNextEdge(p),
                                              tuple(traci.person.getPosition(p)), traci.person.getSpeed(p), people[p]["type"])
                        if samples[p] != direct:
                            raise AssertionError("Subscribed pedestrian state differs from direct TraCI query")
                tracker.observe(now, samples)
                if sample_sink is not None:
                    sample_sink(now, samples, endpoints)
                observations, diagnostics, masks = {}, {}, {}
                edge_data = traci.edge.getAllSubscriptionResults()
                for tls, signal in signals.items():
                    actual = traci.trafficlight.getSubscriptionResults(tls)[tc.TL_RED_YELLOW_GREEN_STATE]
                    signal.plan.assert_safe(actual)
                    if actual != states[tls]:
                        raise AssertionError("SUMO changed a controlled signal unexpectedly")
                    occupied = any(s.road in local[tls] for s in samples.values())
                    signal.advance(now, occupied)
                    if tick % round(cfg["decision_seconds"] / dt) == 0:
                        queues = [sum(edge_data[e][tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for e in incoming[tls][axis])
                                  for axis in ("EW", "NS")]
                        downstream = [sum(edge_data[e][tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for e in outgoing[tls][axis])
                                      for axis in ("EW", "NS")]
                        # Each incoming edge represents a locally sensed approach.
                        features, names = [], []
                        for edge in sorted(set.union(*incoming[tls].values())):
                            features.extend([edge_data[edge][tc.LAST_STEP_VEHICLE_HALTING_NUMBER] / 10,
                                             edge_data[edge][tc.LAST_STEP_VEHICLE_NUMBER] / 10,
                                             max(0, edge_data[edge][tc.LAST_STEP_MEAN_SPEED]) / 13.9])
                            names.extend([edge + suffix for suffix in ("/queue", "/count", "/speed")])
                        features.extend(q / 10 for q in downstream)
                        names.extend(("downstream_EW", "downstream_NS"))
                        pending_count = 0
                        for c in local[tls]:
                            pending = tracker.pending(c)
                            ages = [now - e.reached_at for e in pending]
                            pending_count += len(pending)
                            features.extend([len(pending) / 5, max(ages, default=0) / 60,
                                             (sum(ages) / len(ages) if ages else 0) / 60,
                                             float(any(s.road == c for s in samples.values()))])
                            names.extend([c + suffix for suffix in ("/count", "/oldest", "/mean_age", "/occupied")])
                        stages = (*PHASES, "EW_YELLOW", "NS_YELLOW", "ALL_RED", "PED_CLEARANCE", "READY")
                        features.extend([signal.age(now) / 30, *[float(signal.stage == p) for p in stages]])
                        names.extend(["phase_age", *stages])
                        if observation_schema == "controller-memory-v1":
                            memory, memory_names = controller_memory(signal, now)
                            features.extend(memory)
                            names.extend(memory_names)
                        demand_now = (queues[0] > 0, queues[1] > 0, pending_count > 0)
                        mask = signal.mask(now, demand_now)
                        features.extend(map(float, mask))
                        names.extend([f"legal_{p}" for p in PHASES])
                        observations[tls], masks[tls] = features, mask
                        diagnostics[tls] = (queues, downstream, pending_count, demand_now, occupied)
                        if obs_features is None:
                            obs_features = {}
                        obs_features[tls] = names
                if observations:
                    global_state = [v for tls in sorted(observations) for v in observations[tls]]
                    for tls, signal in signals.items():
                        queues, downstream, pending_count, demand_now, occupied = diagnostics[tls]
                        action = None
                        if any(masks[tls]):
                            action = (policy(tls, observations[tls], global_state, masks[tls])
                                      if controller == "policy" else baseline_action(
                                          controller, signal, now, queues, downstream, pending_count))
                            signal.request(now, action, demand_now, occupied)
                        if trace_sink is not None:
                            trace_sink.append({"time": now, "junction": tls, "observation": observations[tls],
                                               "global_state": global_state, "mask": masks[tls], "action": action,
                                               "vehicle_queue": sum(queues), "pending_count": pending_count,
                                               "uninserted_due": sum(v["depart"] <= now and v["id"] not in departed
                                                   and v["route"][0] in set.union(*incoming[tls].values())
                                                   for v in demand["vehicles"])})
                for tls, signal in signals.items():
                    next_state = signal.state()
                    if next_state != states[tls]:
                        traci.trafficlight.setRedYellowGreenState(tls, next_state)
                    states[tls] = next_state
                if simulation[tc.VAR_COLLIDING_VEHICLES_NUMBER]:
                    # SUMO also adds the vehicle in an intermodal collision to
                    # this counter. Preserve participants/type in failure.json.
                    raise AssertionError(f"SUMO reported a collision: {traci.simulation.getCollisions()!r}")
                if math.isclose(now, cutoff):
                    cutoff_counts = {
                        "vehicle_departed": sorted(departed), "vehicle_arrived": sorted(arrived),
                        "person_seen": sorted(persons_seen), "person_arrived": sorted(persons_arrived),
                        "crossings": tracker.summary(),
                    }
            version = traci.getVersion()[1]
        except Exception as exc:
            (output / "failure.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                "seed": seed, "controller": controller, "error": repr(exc),
                "time": traci.simulation.getTime(), "demand_sha256": demand["sha256"],
                "vehicle_departed_count": len(departed), "vehicle_arrived_count": len(arrived),
                "phase_history": {t: s.history for t, s in signals.items()}}, indent=2))
            raise
        finally:
            traci.close()
        vehicle = vehicle_summary(tripinfo, {v["id"] for v in demand["vehicles"]}, departed, arrived)
        cutoff_vehicle = vehicle_summary(tripinfo, {v["id"] for v in demand["vehicles"] if v["depart"] <= cutoff},
                                          set(cutoff_counts["vehicle_departed"]), set(cutoff_counts["vehicle_arrived"]))
        cohort_ids = {v["id"] for v in demand["vehicles"] if cfg["warmup_seconds"] <= v["depart"] < cutoff}
        cohort_vehicle = vehicle_summary(tripinfo, cohort_ids,
                                         cohort_ids & set(cutoff_counts["vehicle_departed"]),
                                         cohort_ids & set(cutoff_counts["vehicle_arrived"]))
    result = {
        "schema_version": SCHEMA_VERSION, "status": "development evidence, not final experiment",
        "controller": controller, "seed": seed, "junctions": junctions, "sumo_version": version,
        "config": cfg, "demand_sha256": demand["sha256"], "network_sha256": network_hash,
        "vehicles_end": vehicle, "vehicles_at_measurement_cutoff": cutoff_vehicle,
        "vehicles_measurement_cohort": cohort_vehicle,
        "pedestrians_by_crossing_end": tracker.summary(), "measurement_cutoff": cutoff_counts,
        "persons": {"scheduled": len(people), "seen": len(persons_seen), "completed": len(persons_arrived),
                    "uninserted": len(set(people) - persons_seen),
                    "unfinished": len(persons_seen - persons_arrived)},
        "observation_features": obs_features, "phase_history": {t: s.history for t, s in signals.items()},
        "decision_count": sum(len(s.decisions) for s in signals.values()),
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted([*Path(__file__).parent.glob("*.py"),
                                           ROOT / "config/fairness_v2.json",
                                           ROOT / "src/traffic_control/signals.py",
                                           ROOT / "src/traffic_control/connected.py",
                                           ROOT / "src/traffic_control/scenario.py",
                                           ROOT / "src/traffic_control/metrics.py"])},
        "measurement_note": "All-run diagnostic summaries include warm-up; use event timestamps for cohort analysis. Unknown waits are not imputed.",
    }
    for name, data in (("metrics.json", result), ("events.json", tracker.rows()), ("demand_manifest.json", demand),
                       ("decisions.json", {t: s.decisions for t, s in signals.items()})):
        (output / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    failure = output / "failure.json"
    if failure.exists():
        failure.unlink()  # only this explicit episode's stale failure marker
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("fixed", "actuated", "pressure"), default="actuated")
    parser.add_argument("--scenario", choices=sorted(scenarios()), default="vmedium-pmedium")
    parser.add_argument("--seed", type=int, default=21001)
    parser.add_argument("--junctions", type=int, choices=(1, 3), default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    overrides = scenarios()[args.scenario]
    if args.smoke:
        overrides = {**overrides, "end_seconds": 360, "warmup_seconds": 0,
                     "measurement_end_seconds": 240, "demand_end_seconds": 90,
                     "vehicle_rate_per_hour": 720, "pedestrian_rate_per_hour": 720}
    result = run(args.controller, args.seed, args.output, args.junctions, overrides)
    print(json.dumps({"controller": args.controller, "vehicles": result["vehicles_end"],
                      "persons": result["persons"]}, indent=2))


if __name__ == "__main__":
    main()
