"""Seeded synthetic hypotheses, explicitly not calibrated traffic counts."""

import hashlib
import random
import xml.etree.ElementTree as ET

from ..connected import compatible_turn_routes, demand_routes


def scenarios():
    result = {}
    for vehicle, rate in (("low", 960), ("medium", 1920), ("high", 2880)):
        for pedestrian, people in (("low", 400), ("medium", 1200), ("high", 2000)):
            result[f"v{vehicle}-p{pedestrian}"] = {
                "vehicle_rate_per_hour": rate, "pedestrian_rate_per_hour": people}
    extra = {
        "slow-walkers": {"slow_fraction": 1.0},
        "mixed-walkers": {"slow_fraction": .2},
        "pedestrian-burst": {"burst_count": 20},
        "extreme-burst": {"burst_count": 80},
        "vehicle-overload": {"vehicle_rate_per_hour": 8000},
        "short-spacing": {"junction_spacing_m": 65, "vehicle_rate_per_hour": 4800},
        "directional-imbalance": {"directional_ratio": 3.0},
        "sparse-pedestrian-route": {"sparse_route_factor": .1},
        "vehicle-platoons": {"arrival_profile": "platoons"},
        "demand-ramp": {"arrival_profile": "ramp"},
        "higher-turning-share": {"turn_fraction": .3},
        "heterogeneous-drivers": {"heterogeneous_drivers": True},
    }
    for name, overrides in extra.items():
        result[name] = {**result["vmedium-pmedium"], **overrides}
    return result


def make_demand(path, seed, cfg, junctions):
    rng = random.Random(seed)
    root = ET.Element("routes")
    for name, factor in (("car", 1), ("car-slower", .85), ("car-faster", 1.15)):
        ET.SubElement(root, "vType", id=name, vClass="passenger", length="5",
                      accel="2.6", decel="4.5", sigma="0.3" if cfg.get("heterogeneous_drivers") else "0",
                      speedFactor=str(factor), speedDev="0", maxSpeed="13.9")
    for name, speed in (("normal", 1.3), ("slow", .8)):
        ET.SubElement(root, "vType", id=name, vClass="pedestrian", maxSpeed=str(speed),
                      speedFactor="1", speedDev="0")
    through, walks = demand_routes(junctions)
    turns = compatible_turn_routes(junctions)
    turn_fraction = cfg.get("turn_fraction", .1) if turns else 0.0
    if not 0 <= turn_fraction <= 1 or not 0 <= cfg.get("slow_fraction", 0) <= 1:
        raise ValueError("Fractions must lie in [0,1]")
    routes = through + turns
    through_weights = [1.0] * len(through)
    through_weights[0] *= cfg.get("directional_ratio", 1.0)
    weights = [(1 - turn_fraction) * w / sum(through_weights) for w in through_weights]
    weights += [turn_fraction / len(turns)] * len(turns) if turns else []
    walk_weights = [1.0] * len(walks)
    walk_weights[0] = cfg.get("sparse_route_factor", 1.0)
    end = cfg["demand_end_seconds"]

    def departures(rate, profile="poisson"):
        if rate < 0:
            raise ValueError("Arrival rates cannot be negative")
        if not rate:
            return
        peak = 3.0 if profile == "platoons" else 1.5 if profile == "ramp" else 1.0
        now = 0.0
        while True:
            now += rng.expovariate(rate * peak / 3600)
            if now >= end:
                return
            multiplier = (3.0 if now % 60 < 20 else 0.0) if profile == "platoons" else (
                .5 + 2 * min(now / end, 1 - now / end) if profile == "ramp" else 1.0)
            if rng.random() < multiplier / peak:
                yield round(now, 3)

    vehicles, people = [], []
    for i, depart in enumerate(departures(cfg["vehicle_rate_per_hour"], cfg.get("arrival_profile", "poisson"))):
        route = rng.choices(routes, weights)[0]
        kind = rng.choice(("car-slower", "car", "car-faster")) if cfg.get("heterogeneous_drivers") else "car"
        vehicles.append({"id": f"v{i:06d}", "depart": depart, "route": route, "type": kind})
    for depart in departures(cfg["pedestrian_rate_per_hour"]):
        route = rng.choices(walks, walk_weights)[0]
        kind = "slow" if rng.random() < cfg.get("slow_fraction", 0) else "normal"
        people.append({"id": f"p{len(people):06d}", "depart": depart, "route": route, "type": kind})
    burst_at = cfg.get("burst_start_seconds", end / 2)
    for i in range(cfg.get("burst_count", 0)):
        depart = round(burst_at + rng.random() * min(40, end - burst_at), 3)
        if not 0 <= depart < end:
            raise ValueError("Burst must lie inside the demand interval")
        people.append({"id": f"b{i:06d}", "depart": depart, "route": walks[0], "type": "normal"})
    for record, kind in sorted([(v, "vehicle") for v in vehicles] + [(p, "person") for p in people],
                               key=lambda item: (item[0]["depart"], item[0]["id"])):
        attrs = {key: str(record[key]) for key in ("id", "depart", "type")}
        if kind == "vehicle":
            element = ET.SubElement(root, kind, **attrs, departLane="best", departSpeed="0")
            ET.SubElement(element, "route", edges=" ".join(record["route"]))
        else:
            element = ET.SubElement(root, kind, **attrs)
            ET.SubElement(element, "walk", **{"from": record["route"][0], "to": record["route"][1]})
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "vehicles": vehicles, "people": people, "seed": seed,
            "basis": "synthetic; development load labels have not been capacity-calibrated"}
