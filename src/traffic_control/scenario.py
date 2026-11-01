"""Build the plain-XML network and deterministic, explicit SUMO demand."""

from __future__ import annotations

import hashlib
import math
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .connected import compatible_turn_routes, demand_routes, write_network


ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "scenario"


def executable(name: str) -> str:
    local = ROOT / ".local" / "sumo-1.27.1" / "bin" / (name + ".exe")
    path = shutil.which(name) or shutil.which(name + ".exe")
    if path is None and local.exists():
        path = str(local)
    if path is None:
        raise RuntimeError(f"{name} is not on PATH. Install SUMO and add its bin directory to PATH.")
    return path


def build_network(output: Path, junctions: int = 1,
                  spacing_m: int | list[int] = 140,
                  left_hand: bool = False,
                  include_compatible_turns: bool = True) -> dict[str, float]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if junctions == 1:
        node_file, edge_file, connection_file = (SCENARIO / "single.nod.xml",
                                                 SCENARIO / "single.edg.xml",
                                                 SCENARIO / "single.con.xml")
    elif junctions == 3:
        node_file, edge_file, connection_file = write_network(
            output.parent, spacing_m, left_hand, include_compatible_turns)
    else:
        raise ValueError("Only one or three junctions are implemented")
    command = [executable("netconvert"), "--node-files", str(node_file),
               "--edge-files", str(edge_file),
               "--connection-files", str(connection_file),
               "--output-file", str(output)]
    if left_hand:
        command.extend(("--lefthand", "true"))
    subprocess.run(command, check=True)
    tree = ET.parse(output)
    crossings: dict[str, float] = {}
    for edge in tree.findall(".//edge"):
        if edge.get("function") != "crossing":
            continue
        lane = edge.find("lane")
        if lane is not None:
            crossings[edge.attrib["id"]] = float(lane.attrib["length"])
    if len(crossings) < 1:
        raise RuntimeError("No protected crossing was generated; inspect the plain XML network.")
    return crossings


def make_demand(output: Path, seed: int, config: dict, junctions: int = 1,
                slow_pedestrians: bool = False) -> dict:
    """Same seed and config produce byte-identical demand for either controller."""
    rng = random.Random(seed)
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", vClass="passenger", accel="2.6", decel="4.5",
                  sigma="0", length="5", maxSpeed="13.9", speedFactor="1")
    walking_speed = config["minimum_walking_speed_mps"] if slow_pedestrians else 1.3
    ET.SubElement(root, "vType", id="walker", vClass="pedestrian", maxSpeed=str(walking_speed))
    vehicle_routes, person_od = demand_routes(junctions, bool(config.get("left_hand", False)))
    trips = []
    for route_index, route in enumerate(vehicle_routes):
        for i in range(config["vehicle_count_per_axis"]):
            depart = rng.randint(0, config["vehicle_depart_last_second"])
            trips.append((depart, f"v{route_index:01d}_{i:03d}", route))
    turn_count = int(config.get("turning_vehicle_count_per_route", 0))
    if turn_count < 0:
        raise ValueError("turning_vehicle_count_per_route must be nonnegative")
    for route_index, route in enumerate(
            compatible_turn_routes(junctions, bool(config.get("left_hand", False)))):
        for i in range(turn_count):
            depart = rng.randint(0, config["vehicle_depart_last_second"])
            trips.append((depart, f"vt{route_index:02d}_{i:04d}", route))
    people = []
    for i in range(config["pedestrian_count"]):
        start, finish = person_od[i % len(person_od)]
        depart = rng.randint(0, config["pedestrian_depart_last_second"])
        people.append((depart, f"p{i:03d}", start, finish))
    burst_count = int(config.get("pedestrian_burst_count", 0))
    burst_route = int(config.get("pedestrian_burst_route_index", 0))
    burst_start = int(config.get("pedestrian_burst_start_second", 120))
    burst_end = int(config.get("pedestrian_burst_end_second", 135))
    if burst_count < 0 or not 0 <= burst_route < len(person_od) or burst_start < 0 or burst_end < burst_start:
        raise ValueError("Invalid pedestrian burst parameters")
    if burst_end >= config["end_seconds"]:
        raise ValueError("Pedestrian burst must depart before simulation end")
    start, finish = person_od[burst_route]
    for i in range(burst_count):
        depart = rng.randint(burst_start, burst_end)
        people.append((depart, f"pb{i:04d}", start, finish))
    for kind, depart, ident, start, finish in sorted(
        [("vehicle", depart, ident, route, None) for depart, ident, route in trips]
        + [("person", *person) for person in people],
        key=lambda row: (row[1], row[2]),
    ):
        if kind == "vehicle":
            vehicle = ET.SubElement(root, "vehicle", id=ident, type="car", depart=str(depart),
                                    departLane="best", departSpeed="0")
            ET.SubElement(vehicle, "route", edges=" ".join(start))
        else:
            person = ET.SubElement(root, "person", id=ident, type="walker", depart=str(depart))
            ET.SubElement(person, "walk", **{"from": start, "to": finish})
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"vehicle_ids": sorted(t[1] for t in trips), "person_ids": sorted(p[1] for p in people),
            "sha256": digest}


def clearance_seconds(crossings: dict[str, float], config: dict) -> int:
    return math.ceil(max(crossings.values()) / config["minimum_walking_speed_mps"]
                     + config["pedestrian_clearance_buffer_seconds"])


def crossing_edges(network_file: Path) -> dict[str, str]:
    root = ET.parse(network_file).getroot()
    return {edge.attrib["id"]: edge.attrib.get("crossingEdges", "")
            for edge in root.findall("edge") if edge.get("function") == "crossing"}
