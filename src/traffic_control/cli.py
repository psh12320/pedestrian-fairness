"""Run one reproducible baseline episode in SUMO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .connected import incoming_groups
from .metrics import CrossingTracker, vehicle_summary
from .scenario import ROOT, build_network, clearance_seconds, crossing_edges, executable, make_demand
from .signals import Controller, SignalPlan, classify_links


def run(controller_kind: str, seed: int, output: Path, config_path: Path,
        junctions: int = 1, slow_pedestrians: bool = False,
        config_overrides: dict | None = None, policy_callback=None,
        trace_sink: dict | None = None, objective: str = "vehicle") -> dict:
    try:
        import traci
    except ImportError as exc:
        raise RuntimeError("TraCI is missing. Install the small Python package with: pip install -r requirements.txt") from exc

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if controller_kind == "policy" and policy_callback is None:
        raise ValueError("A policy callback is required for learned control")
    if objective not in ("vehicle", "pedestrian"):
        raise ValueError(objective)
    cfg.update(config_overrides or {})
    if junctions == 3 and "end_seconds" not in (config_overrides or {}):
        cfg["end_seconds"] = cfg["end_seconds_three"]
    if cfg["step_seconds"] != 1:
        raise ValueError("This milestone's event clock and signal plan require one-second steps")
    if cfg["vehicle_green_fixed_seconds"] < cfg["vehicle_green_min_seconds"]:
        raise ValueError("Fixed green cannot be shorter than the legal minimum")
    if cfg["vehicle_green_actuated_max_seconds"] < cfg["vehicle_green_min_seconds"]:
        raise ValueError("Actuated max green cannot be shorter than the legal minimum")
    output.mkdir(parents=True, exist_ok=True)
    network_file = output / ("single.net.xml" if junctions == 1 else "connected.net.xml")
    routes_file = output / "demand.rou.xml"
    tripinfo_file = output / "tripinfo.xml"
    crossings = build_network(network_file, junctions)
    demand = make_demand(routes_file, seed, cfg, junctions, slow_pedestrians)
    clearance = clearance_seconds(crossings, cfg)
    sumo_command = [
        executable("sumo"), "--net-file", str(network_file), "--route-files", str(routes_file),
        "--begin", "0", "--end", str(cfg["end_seconds"]), "--step-length", "1",
        "--seed", str(seed), "--pedestrian.striping.dawdling", "0",
        "--time-to-teleport", "-1", "--tripinfo-output", str(tripinfo_file),
        "--tripinfo-output.write-unfinished", "true", "--no-step-log", "true",
    ]
    traci.start(sumo_command)
    departed: set[str] = set()
    arrived: set[str] = set()
    tracker = CrossingTracker(set(crossings))
    try:
        expected_ids = set(incoming_groups(junctions))
        tls_ids = set(traci.trafficlight.getIDList())
        if tls_ids != expected_ids:
            raise RuntimeError(f"Expected traffic lights {expected_ids}; got {tls_ids}")
        plans = {}
        controllers = {}
        expected_states = {}
        crossing_sets = {}
        for tls in sorted(tls_ids):
            local_crossings = {crossing for crossing in crossings if crossing.startswith(f":{tls}_c")}
            if not local_crossings:
                raise RuntimeError(f"No crossings generated at {tls}")
            crossing_sets[tls] = local_crossings
            initial_state = traci.trafficlight.getRedYellowGreenState(tls)
            groups = classify_links(traci.trafficlight.getControlledLinks(tls), local_crossings,
                                    incoming_groups(junctions)[tls])
            plans[tls] = SignalPlan(groups, len(initial_state),
                                    clearance_seconds({c: crossings[c] for c in local_crossings}, cfg), cfg)
            controllers[tls] = Controller(plans[tls], controller_kind)
            expected_states[tls] = plans[tls].state(controllers[tls].stage)
            traci.trafficlight.setRedYellowGreenState(tls, expected_states[tls])
        if trace_sink is not None:
            trace_sink.update({tls: [] for tls in sorted(tls_ids)})
        for _ in range(cfg["end_seconds"]):
            traci.simulationStep()
            now = traci.simulation.getTime()
            departed.update(traci.simulation.getDepartedIDList())
            arrived.update(traci.simulation.getArrivedIDList())
            persons = {}
            for ident in traci.person.getIDList():
                persons[ident] = (traci.person.getRoadID(ident),
                                  traci.person.getNextEdge(ident),
                                  traci.person.getWaitingTime(ident))
            tracker.observe(now, persons)
            observations = {}
            rewards = {}
            if controller_kind == "policy":
                for tls in sorted(tls_ids):
                    incoming = incoming_groups(junctions)[tls]
                    ew_queue = sum(traci.edge.getLastStepHaltingNumber(edge)
                                   for edge in incoming["EW"])
                    ns_queue = sum(traci.edge.getLastStepHaltingNumber(edge)
                                   for edge in incoming["NS"])
                    pending_count = sum(1 for event in tracker.events.values()
                                        if event.crossing_id in crossing_sets[tls]
                                        and event.entered_at is None)
                    stage = controllers[tls].stage
                    observations[tls] = [ew_queue / 10, ns_queue / 10, pending_count / 5,
                                         controllers[tls].elapsed / 30,
                                         *[float(stage == phase) for phase in
                                           ("EW", "NS", "PED", "PED_CLEARANCE")],
                                         float(stage not in ("EW", "NS", "PED", "PED_CLEARANCE"))]
                    rewards[tls] = -(ew_queue + ns_queue +
                                     (3 * pending_count if objective == "pedestrian" else 0)) / 10
                    if trace_sink is not None and trace_sink[tls]:
                        trace_sink[tls][-1]["reward"] = rewards[tls]
                global_obs = [value for tls in sorted(tls_ids) for value in observations[tls]]
            for tls in sorted(tls_ids):
                local_crossings = crossing_sets[tls]
                crossing_occupied = any(road in local_crossings for road, _, _ in persons.values())
                actual = traci.trafficlight.getRedYellowGreenState(tls)
                plans[tls].assert_safe(actual)
                if actual != expected_states[tls]:
                    raise AssertionError(f"{tls} changed unexpectedly: {actual} != {expected_states[tls]}")
                if controllers[tls].stage in ("EW", "NS") and crossing_occupied:
                    raise AssertionError(f"A pedestrian is on a crossing during {tls} vehicle green")
                action = 0
                if controller_kind == "policy":
                    stage = controllers[tls].stage
                    eligible = (stage in ("EW", "NS") and
                                controllers[tls].elapsed + 1 >= cfg["vehicle_green_min_seconds"] and
                                controllers[tls].elapsed + 1 < cfg["vehicle_green_actuated_max_seconds"])
                    action, logp, value = policy_callback(tls, observations[tls], global_obs, eligible)
                    if trace_sink is not None:
                        trace_sink[tls].append({"obs": observations[tls], "global_obs": global_obs,
                                                "action": action, "old_logp": logp,
                                                "value": value, "actor_mask": eligible,
                                                "reward": None})
                target = controllers[tls].tick(int(now), tracker.pending_for(local_crossings),
                                               crossing_occupied, action)
                traci.trafficlight.setRedYellowGreenState(tls, target)
                expected_states[tls] = target
        if trace_sink is not None:
            for tls in sorted(tls_ids):
                if trace_sink[tls] and trace_sink[tls][-1]["reward"] is None:
                    trace_sink[tls].pop()
        version = traci.getVersion()
    finally:
        traci.close()

    metrics = {
        "controller": controller_kind,
        "junctions": junctions,
        "slow_pedestrians": slow_pedestrians,
        "config": cfg,
        "seed": seed,
        "sumo_version": version,
        "demand_sha256": demand["sha256"],
        "crossing_lengths_metres": crossings,
        "crossing_edges": crossing_edges(network_file),
        "pedestrian_clearance_seconds": clearance,
        "vehicles": vehicle_summary(tripinfo_file, set(demand["vehicle_ids"]), departed, arrived),
        "pedestrians_by_crossing": tracker.summary(cfg["end_seconds"]),
        "persons_seen_at_crossing": len({event.person_id for event in tracker.events.values()}),
        "signal_history": (controllers["C"].history if junctions == 1 else
                           {tls: control.history for tls, control in controllers.items()}),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    (output / "crossing_events.json").write_text(json.dumps(tracker.event_rows(), indent=2) + "\n",
                                                    encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("fixed", "actuated"), required=True)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--junctions", type=int, choices=(1, 3), default=1)
    parser.add_argument("--slow-pedestrians", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "milestone.json")
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / f"{args.junctions}j-{args.controller}-seed{args.seed}"
    metrics = run(args.controller, args.seed, output, args.config, args.junctions,
                  args.slow_pedestrians)
    print(json.dumps({key: metrics[key] for key in ("controller", "seed", "vehicles",
                    "pedestrians_by_crossing")}, indent=2))


if __name__ == "__main__":
    main()
