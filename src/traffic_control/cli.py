"""Run one reproducible baseline episode in SUMO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .connected import incoming_groups, outgoing_groups
from .metrics import CrossingTracker, vehicle_summary
from .scenario import ROOT, build_network, clearance_seconds, crossing_edges, executable, make_demand
from .signals import Controller, SignalPlan, classify_links, load_signal_conflicts


@dataclass(frozen=True)
class PreparedInputs:
    """Read-only network and demand shared by matched controller episodes."""

    network_file: Path
    routes_file: Path
    crossings: dict[str, float]
    demand: dict
    config: dict
    seed: int
    junctions: int
    slow_pedestrians: bool
    network_sha256: str


def prepare_inputs(output: Path, seed: int, config_path: Path,
                   junctions: int = 1, slow_pedestrians: bool = False,
                   config_overrides: dict | None = None) -> PreparedInputs:
    """Build one scenario/seed's immutable SUMO inputs for matched runs.

    The caller owns ``output`` and keeps it alive until every controller run
    finishes. Each run still writes its own tripinfo and metric files.
    """
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg.update(config_overrides or {})
    cfg.pop("pedestrian_reward_weight", None)
    if junctions == 3 and "end_seconds" not in (config_overrides or {}):
        cfg["end_seconds"] = cfg["end_seconds_three"]
    output.mkdir(parents=True, exist_ok=True)
    network_file = output / ("single.net.xml" if junctions == 1 else "connected.net.xml")
    routes_file = output / "demand.rou.xml"
    spacing = cfg.get("junction_site_centroids_m",
                      cfg.get("junction_spacings_m", cfg.get("junction_spacing_m", 140)))
    crossings = build_network(network_file, junctions, spacing,
                              bool(cfg.get("left_hand", False)),
                              bool(cfg.get("include_compatible_turns", True)))
    demand = make_demand(routes_file, seed, cfg, junctions, slow_pedestrians)
    return PreparedInputs(network_file, routes_file, crossings, demand, cfg,
                          seed, junctions, slow_pedestrians,
                          hashlib.sha256(network_file.read_bytes()).hexdigest())


def run(controller_kind: str, seed: int, output: Path, config_path: Path,
        junctions: int = 1, slow_pedestrians: bool = False,
        config_overrides: dict | None = None, policy_callback=None,
        trace_sink: dict | None = None, objective: str = "vehicle",
        pedestrian_reward_weight: float | None = None,
        prepared_inputs: PreparedInputs | None = None) -> dict:
    try:
        import traci
        from traci import constants as tc
    except ImportError as exc:
        raise RuntimeError("TraCI is missing. Install the small Python package with: pip install -r requirements.txt") from exc

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if controller_kind == "policy" and policy_callback is None:
        raise ValueError("A policy callback is required for learned control")
    if objective not in ("vehicle", "pedestrian"):
        raise ValueError(objective)
    cfg.update(config_overrides or {})
    reward_weight = (3.0 if objective == "pedestrian" else 0.0) if pedestrian_reward_weight is None \
        else float(pedestrian_reward_weight)
    if not math.isfinite(reward_weight) or reward_weight < 0:
        raise ValueError("pedestrian_reward_weight must be finite and nonnegative")
    cfg["pedestrian_reward_weight"] = reward_weight
    if junctions == 3 and "end_seconds" not in (config_overrides or {}):
        cfg["end_seconds"] = cfg["end_seconds_three"]
    if cfg["step_seconds"] != 1:
        raise ValueError("This milestone's event clock and signal plan require one-second steps")
    if cfg["vehicle_green_fixed_seconds"] < cfg["vehicle_green_min_seconds"]:
        raise ValueError("Fixed green cannot be shorter than the legal minimum")
    if cfg["vehicle_green_actuated_max_seconds"] < cfg["vehicle_green_min_seconds"]:
        raise ValueError("Actuated max green cannot be shorter than the legal minimum")
    output.mkdir(parents=True, exist_ok=True)
    tripinfo_file = output / "tripinfo.xml"
    if prepared_inputs is None:
        network_file = output / ("single.net.xml" if junctions == 1 else "connected.net.xml")
        routes_file = output / "demand.rou.xml"
        spacing = cfg.get("junction_site_centroids_m",
                          cfg.get("junction_spacings_m", cfg.get("junction_spacing_m", 140)))
        crossings = build_network(network_file, junctions, spacing,
                                  bool(cfg.get("left_hand", False)),
                                  bool(cfg.get("include_compatible_turns", True)))
        demand = make_demand(routes_file, seed, cfg, junctions, slow_pedestrians)
    else:
        input_cfg = {key: value for key, value in cfg.items()
                     if key != "pedestrian_reward_weight"}
        if (prepared_inputs.seed != seed or prepared_inputs.junctions != junctions
                or prepared_inputs.slow_pedestrians != slow_pedestrians
                or prepared_inputs.config != input_cfg):
            raise ValueError("Prepared SUMO inputs do not match the episode configuration")
        network_file = prepared_inputs.network_file
        routes_file = prepared_inputs.routes_file
        if (hashlib.sha256(network_file.read_bytes()).hexdigest()
                != prepared_inputs.network_sha256
                or hashlib.sha256(routes_file.read_bytes()).hexdigest()
                != prepared_inputs.demand["sha256"]):
            raise ValueError("Prepared SUMO inputs changed after generation")
        crossings = prepared_inputs.crossings
        demand = prepared_inputs.demand
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
    connector_diagnostics = None
    try:
        expected_ids = set(incoming_groups(junctions))
        tls_ids = set(traci.trafficlight.getIDList())
        if tls_ids != expected_ids:
            raise RuntimeError(f"Expected traffic lights {expected_ids}; got {tls_ids}")
        plans = {}
        controllers = {}
        expected_states = {}
        crossing_sets = {}
        crossing_endpoints = {}
        for crossing in crossings:
            shape = traci.lane.getShape(crossing + "_0")
            if len(shape) < 2:
                raise RuntimeError(f"Crossing {crossing} has no entrance geometry")
            crossing_endpoints[crossing] = (shape[0], shape[-1])
        subscribed_people: set[str] = set()
        person_variables = (tc.VAR_ROAD_ID, tc.VAR_NEXT_EDGE,
                            tc.VAR_WAITING_TIME, tc.VAR_POSITION)
        if junctions == 3 and cfg.get("record_connector_diagnostics", False):
            connector_edges = ("h1e", "h1w", "h2e", "h2w")
            connector_upstream = {"h1e": "h0e", "h2e": "h1e",
                                  "h1w": "h2w", "h2w": "h3w"}
            connector_lengths = {edge: traci.lane.getLength(edge + "_0")
                                 for edge in connector_edges}
            connector_diagnostics = {
                "near_capacity_seconds": 0,
                "near_capacity_with_upstream_queue_seconds": 0,
                "max_occupancy_fraction": 0.0,
                "max_halted_on_connector": 0,
                "vehicle_space_assumption_metres": 7.5,
                "near_capacity_threshold": 0.8,
            }
        for tls in sorted(tls_ids):
            local_crossings = {crossing for crossing in crossings if crossing.startswith(f":{tls}_c")}
            if not local_crossings:
                raise RuntimeError(f"No crossings generated at {tls}")
            crossing_sets[tls] = local_crossings
            initial_state = traci.trafficlight.getRedYellowGreenState(tls)
            groups = classify_links(traci.trafficlight.getControlledLinks(tls), local_crossings,
                                    incoming_groups(junctions)[tls])
            plans[tls] = SignalPlan(groups, len(initial_state),
                                    clearance_seconds({c: crossings[c] for c in local_crossings}, cfg),
                                    cfg, load_signal_conflicts(network_file, tls))
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
            if connector_diagnostics is not None:
                near_capacity = False
                upstream_queued = False
                for edge, length in connector_lengths.items():
                    count = traci.edge.getLastStepVehicleNumber(edge)
                    halted = traci.edge.getLastStepHaltingNumber(edge)
                    fraction = count * 7.5 / length
                    connector_diagnostics["max_occupancy_fraction"] = max(
                        connector_diagnostics["max_occupancy_fraction"], fraction)
                    connector_diagnostics["max_halted_on_connector"] = max(
                        connector_diagnostics["max_halted_on_connector"], halted)
                    if fraction >= 0.8:
                        near_capacity = True
                        upstream_queued |= bool(traci.edge.getLastStepHaltingNumber(
                            connector_upstream[edge]))
                connector_diagnostics["near_capacity_seconds"] += int(near_capacity)
                connector_diagnostics["near_capacity_with_upstream_queue_seconds"] += int(
                    near_capacity and upstream_queued)
            persons = {}
            current_people = set(traci.person.getIDList())
            for ident in sorted(current_people - subscribed_people):
                traci.person.subscribe(ident, person_variables)
            subscribed_people = current_people
            subscriptions = traci.person.getAllSubscriptionResults()
            for ident in sorted(current_people):
                values = subscriptions.get(ident, {})
                road = values.get(tc.VAR_ROAD_ID)
                next_edge = values.get(tc.VAR_NEXT_EDGE)
                waiting = values.get(tc.VAR_WAITING_TIME)
                position = values.get(tc.VAR_POSITION)
                if road is None:
                    road = traci.person.getRoadID(ident)
                if next_edge is None:
                    next_edge = traci.person.getNextEdge(ident)
                if waiting is None:
                    waiting = traci.person.getWaitingTime(ident)
                distance = None
                if next_edge in crossing_endpoints:
                    if position is None:
                        position = traci.person.getPosition(ident)
                    distance = min(math.dist(position, endpoint)
                                   for endpoint in crossing_endpoints[next_edge])
                persons[ident] = (road, next_edge, waiting, distance)
            tracker.observe(now, persons)
            observations = {}
            rewards = {}
            pressure_inputs = {}
            if controller_kind in ("policy", "pressure"):
                for tls in sorted(tls_ids):
                    incoming = incoming_groups(junctions)[tls]
                    ew_queue = sum(traci.edge.getLastStepHaltingNumber(edge)
                                   for edge in incoming["EW"])
                    ns_queue = sum(traci.edge.getLastStepHaltingNumber(edge)
                                   for edge in incoming["NS"])
                    pending_count = sum(1 for event in tracker.events.values()
                                        if event.crossing_id in crossing_sets[tls]
                                        and event.entered_at is None)
                    if controller_kind == "pressure":
                        outgoing = outgoing_groups(junctions)[tls]
                        ew_downstream = sum(traci.edge.getLastStepHaltingNumber(edge)
                                            for edge in outgoing["EW"])
                        ns_downstream = sum(traci.edge.getLastStepHaltingNumber(edge)
                                            for edge in outgoing["NS"])
                        pressure_inputs[tls] = (max(0, ew_queue - ew_downstream),
                                                max(0, ns_queue - ns_downstream),
                                                pending_count)
                    else:
                        stage = controllers[tls].stage
                        observations[tls] = [ew_queue / 10, ns_queue / 10, pending_count / 5,
                                             controllers[tls].elapsed / 30,
                                             *[float(stage == phase) for phase in
                                               ("EW", "NS", "PED", "PED_CLEARANCE")],
                                             float(stage not in ("EW", "NS", "PED", "PED_CLEARANCE"))]
                        rewards[tls] = -(ew_queue + ns_queue + reward_weight * pending_count) / 10
                        if trace_sink is not None and trace_sink[tls]:
                            trace_sink[tls][-1]["reward"] = rewards[tls]
                if controller_kind == "policy":
                    global_obs = [value for tls in sorted(tls_ids) for value in observations[tls]]
            for tls in sorted(tls_ids):
                local_crossings = crossing_sets[tls]
                crossing_occupied = any(road in local_crossings for road, _, _, _ in persons.values())
                actual = traci.trafficlight.getRedYellowGreenState(tls)
                plans[tls].assert_safe(actual)
                if actual != expected_states[tls]:
                    raise AssertionError(f"{tls} changed unexpectedly: {actual} != {expected_states[tls]}")
                if controllers[tls].stage in ("EW", "NS") and crossing_occupied:
                    occupants = [(ident, road) for ident, (road, _, _, _) in persons.items()
                                 if road in local_crossings]
                    raise AssertionError(
                        f"A pedestrian is on a crossing during {tls} vehicle green "
                        f"at t={now}: {occupants[:8]}, stage={controllers[tls].stage}, "
                        f"recent_phases={controllers[tls].history[-6:]}")
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
                elif controller_kind == "pressure" and controllers[tls].stage in ("EW", "NS"):
                    ew_pressure, ns_pressure, ped_pending = pressure_inputs[tls]
                    ped_pressure = float(cfg.get("pressure_pedestrian_weight", 2.0)) * ped_pending
                    if controllers[tls].stage == "EW":
                        action = int(ns_pressure + ped_pressure > ew_pressure)
                    else:
                        action = int(ew_pressure + ped_pressure > ns_pressure)
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
    if connector_diagnostics is not None:
        metrics["connector_diagnostics"] = connector_diagnostics
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    (output / "crossing_events.json").write_text(json.dumps(tracker.event_rows(), indent=2) + "\n",
                                                    encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("fixed", "actuated", "pressure"), required=True)
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
