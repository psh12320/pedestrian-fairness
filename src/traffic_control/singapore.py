"""Prepare and audit a small Singapore-shaped SUMO case study.

This module does not train a controller or claim to reproduce an LTA signal plan.
It records source hashes and exposes gaps that require local observation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import random
import stat
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from pathlib import PurePosixPath

from .scenario import ROOT, executable


OSM_API = "https://api.openstreetmap.org/api/0.6/map"
LTA_SIGNAL_DATASET = "d_44cc2bba6131f9057fb92f367eba69ed"
LTA_POLL = ("https://api-open.data.gov.sg/v1/public/api/datasets/"
            + LTA_SIGNAL_DATASET + "/poll-download")
COUNT_FIELDS = ("collection_date", "site_id", "start_s", "end_s", "mode",
                "vehicle_class", "movement_id", "count", "source",
                "source_reference")
OBSERVED_SOURCES = {"field_observation", "lta_traffic_count"}
COUNT_MEMBER_LIMIT = 20_000_000


def load_site(path: Path) -> dict:
    site = json.loads(path.read_text(encoding="utf-8"))
    bounds = site["bbox_west_south_east_north"]
    if len(bounds) != 4 or not all(isinstance(x, (float, int)) for x in bounds):
        raise ValueError("bbox_west_south_east_north needs four numeric values")
    west, south, east, north = bounds
    if not (103.5 < west < east < 104.2 and 1.1 < south < north < 1.6):
        raise ValueError("The bounding box must be a small area within Singapore")
    if (east - west) * (north - south) > 0.0003:
        raise ValueError("Keep the pilot bounding box under 0.0003 square degrees")
    if site.get("study_junctions") != 3:
        raise ValueError("This pilot expects three connected junctions")
    if not 0 < site.get("max_osm_bytes", 0) <= 12_000_000:
        raise ValueError("max_osm_bytes must be at most 12 MB")
    if not 0 < site.get("max_lta_geojson_bytes", 0) <= 25_000_000:
        raise ValueError("max_lta_geojson_bytes must be at most 25 MB")
    if not 0 < site.get("max_signal_link_metres", 0) <= 800:
        raise ValueError("max_signal_link_metres must be within 1–800 m")
    return site


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_limited(url: str, limit: int) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS data sources are accepted")
    request = urllib.request.Request(url, headers={"User-Agent": "cs4246-team38-sumo-case/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Source exceeds the {limit / 1_000_000:g} MB download cap")
    return data


def fetch_osm(bounds: list[float], limit: int) -> tuple[bytes, str]:
    query = urllib.parse.urlencode({"bbox": ",".join(str(x) for x in bounds)})
    url = OSM_API + "?" + query
    data = get_limited(url, limit)
    if ET.fromstring(data).tag != "osm":
        raise ValueError("OSM did not return an <osm> document")
    return data, url


def fetch_lta_signal_aspects(limit: int) -> tuple[dict, str]:
    """Fetch the public full GeoJSON in memory, keeping only the clipped output."""
    reply = json.loads(get_limited(LTA_POLL, 100_000))
    if reply.get("code") != 0 or not reply.get("data", {}).get("url"):
        raise RuntimeError(f"LTA signal-aspect download unavailable: {reply.get('errMsg', reply)}")
    url = reply["data"]["url"]
    full_data = get_limited(url, limit)
    geojson = json.loads(full_data)
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Expected LTA FeatureCollection")
    return geojson, sha256(full_data)


def clip_points(geojson: dict, bounds: list[float]) -> dict:
    west, south, east, north = bounds
    chosen = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        coords = geometry.get("coordinates", [])
        if len(coords) >= 2 and west <= coords[0] <= east and south <= coords[1] <= north:
            chosen.append(feature)
    return {"type": "FeatureCollection", "features": chosen}


def osm_audit(osm_file: Path) -> dict:
    data = osm_file.read_bytes()
    root = ET.fromstring(gzip.decompress(data) if osm_file.suffix == ".gz" else data)
    if root.tag != "osm":
        raise ValueError("Not an OSM XML document")
    traffic_signals = []
    crossing_ids = []
    footway_ids = []
    for item in root:
        if item.tag not in {"node", "way"}:
            continue
        tags = {t.get("k"): t.get("v") for t in item.findall("tag")}
        if item.tag == "node" and tags.get("highway") == "traffic_signals":
            traffic_signals.append(item.get("id"))
        if (tags.get("highway") == "crossing" or tags.get("footway") == "crossing"
                or tags.get("crossing") == "traffic_signals"):
            crossing_ids.append(item.get("id"))
        if tags.get("highway") in {"footway", "path", "pedestrian"}:
            footway_ids.append(item.get("id"))
    return {"osm_signal_nodes": sorted(traffic_signals),
            "osm_crossing_features": len(crossing_ids),
            "osm_footway_features": len(footway_ids)}


def build_sumo_network(osm_file: Path, net_file: Path) -> dict:
    net_file.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        executable("netconvert"), "--osm-files", str(osm_file),
        "--output-file", str(net_file), "--lefthand", "true",
        "--osm.sidewalks", "true", "--osm.crossings", "true",
    ], capture_output=True, text=True)
    log = net_file.parent / "netconvert.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"netconvert failed; inspect {log}")
    warnings = [line for line in result.stderr.splitlines() if line.startswith("Warning:")]
    return {"warning_count": len(warnings), "warning_log": log.name}


def _vehicle_edge(edge: ET.Element) -> bool:
    lanes = edge.findall("lane")
    for lane in lanes:
        allowed = lane.get("allow", "").split()
        forbidden = lane.get("disallow", "").split()
        if (not allowed or "passenger" in allowed) and "passenger" not in forbidden:
            return True
    return False


def candidate_chains(net_root: ET.Element, max_link_m: float = 450) -> list[dict]:
    """Suggest three *separate* signal sites, not three heads at one intersection."""
    junctions = {j.get("id"): j for j in net_root.findall("junction")
                 if j.get("type") != "internal"}
    controlled = {j_id for j_id, j in junctions.items()
                  if j.get("type", "").startswith("traffic_light")}
    parent = {j_id: j_id for j_id in controlled}

    def find(j_id: str) -> str:
        while parent[j_id] != j_id:
            parent[j_id] = parent[parent[j_id]]
            j_id = parent[j_id]
        return j_id

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    ids = sorted(controlled)
    for index, left in enumerate(ids):
        x1, y1 = float(junctions[left].get("x", "nan")), float(junctions[left].get("y", "nan"))
        for right in ids[index + 1:]:
            x2, y2 = float(junctions[right].get("x", "nan")), float(junctions[right].get("y", "nan"))
            if math.hypot(x1 - x2, y1 - y2) <= 45:
                union(left, right)
    groups: dict[str, list[str]] = defaultdict(list)
    for j_id in ids:
        groups[find(j_id)].append(j_id)
    member_group = {member: group for group, members in groups.items() for member in members}

    graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in net_root.findall("edge"):
        if edge.get("function") or not _vehicle_edge(edge):
            continue
        start, end = edge.get("from"), edge.get("to")
        lane = edge.find("lane")
        if start not in junctions or end not in junctions or lane is None:
            continue
        length = float(lane.get("length", "0"))
        graph[start].append((end, length))
        graph[end].append((start, length))

    adjacent: dict[str, dict[str, float]] = defaultdict(dict)
    for source, members in groups.items():
        heap = [(0.0, member) for member in members]
        heapq.heapify(heap)
        best = {member: 0.0 for member in members}
        while heap:
            distance, here = heapq.heappop(heap)
            if distance != best[here] or distance > max_link_m:
                continue
            other_group = member_group.get(here)
            if other_group is not None and other_group != source:
                adjacent[source][other_group] = min(
                    distance, adjacent[source].get(other_group, math.inf))
                continue
            for nxt, length in graph[here]:
                new_distance = distance + length
                if new_distance <= max_link_m and new_distance < best.get(nxt, math.inf):
                    best[nxt] = new_distance
                    heapq.heappush(heap, (new_distance, nxt))

    proposals = {}
    for middle, neighbors in adjacent.items():
        ordered = sorted(neighbors)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                route = (left, middle, right)
                reverse = tuple(reversed(route))
                key = min(route, reverse)
                total = neighbors[left] + neighbors[right]
                if total < proposals.get(key, math.inf):
                    proposals[key] = total
    output = []
    for group_ids, road_distance in sorted(proposals.items(), key=lambda x: (x[1], x[0]))[:10]:
        points = [
            (sum(float(junctions[member].get("x")) for member in groups[group]) / len(groups[group]),
             sum(float(junctions[member].get("y")) for member in groups[group]) / len(groups[group]))
            for group in group_ids
        ]
        output.append({"signal_groups": [groups[group] for group in group_ids],
                       "road_path_metres": round(road_distance, 1),
                       "site_centroids_m": [[round(x, 2), round(y, 2)] for x, y in points],
                       "centroid_spacings_m": [round(math.dist(points[0], points[1]), 1),
                                               round(math.dist(points[1], points[2]), 1)]})
    return output


def network_audit(net_file: Path, max_link_m: float) -> dict:
    root = ET.parse(net_file).getroot()
    if root.tag != "net":
        raise ValueError("Not a SUMO network")
    crossings = [e for e in root.findall("edge") if e.get("function") == "crossing"]
    crossing_ids = {e.get("id") for e in crossings}
    signalised_crossings = {e_id for connection in root.findall("connection")
                            if connection.get("tl")
                            for e_id in (connection.get("from"), connection.get("to"))
                            if e_id in crossing_ids}
    sidewalks = sum(1 for e in root.findall("edge") for lane in e.findall("lane")
                    if lane.get("allow") == "pedestrian")
    return {"sumo_signal_junctions": sum(1 for j in root.findall("junction")
                                           if j.get("type", "").startswith("traffic_light")),
            "sumo_crossings": len(crossings),
            "sumo_signalised_crossings": len(signalised_crossings),
            "sumo_pedestrian_only_lanes": sidewalks,
            "candidate_three_signal_chains": candidate_chains(root, max_link_m)}


def audit_counts(path: Path) -> dict:
    """Validate optional local observations without inventing demand."""
    counts = {"vehicle": 0, "pedestrian": 0}
    intervals = set()
    sources = set()
    sites = set()
    dates = set()
    classes = set()
    site_intervals = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or []) != COUNT_FIELDS:
            raise ValueError("Counts CSV header must be " + ",".join(COUNT_FIELDS))
        for line, row in enumerate(reader, 2):
            try:
                start, end, count = (int(row[k]) for k in ("start_s", "end_s", "count"))
            except (ValueError, TypeError) as error:
                raise ValueError(f"Line {line}: start_s, end_s, count must be integers") from error
            if start < 0 or end <= start or count < 0:
                raise ValueError(f"Line {line}: invalid interval or negative count")
            mode = row["mode"].strip()
            if mode not in counts:
                raise ValueError(f"Line {line}: mode must be vehicle or pedestrian")
            try:
                collected = date.fromisoformat(row["collection_date"].strip())
            except ValueError as error:
                raise ValueError(f"Line {line}: collection_date must be YYYY-MM-DD") from error
            site = row["site_id"].strip()
            vehicle_class = row["vehicle_class"].strip()
            if not all(row[name].strip() for name in
                       ("site_id", "movement_id", "source", "source_reference")):
                raise ValueError(f"Line {line}: site, movement, source, and reference are required")
            if mode == "vehicle" and not vehicle_class:
                raise ValueError(f"Line {line}: vehicle_class is required for vehicle counts")
            if mode == "pedestrian" and vehicle_class:
                raise ValueError(f"Line {line}: vehicle_class must be empty for pedestrians")
            counts[mode] += count
            intervals.add((start, end))
            sources.add(row["source"].strip())
            sites.add(site)
            dates.add(collected.isoformat())
            site_intervals[site].add((collected.isoformat(), start, end))
            if vehicle_class:
                classes.add(vehicle_class)
    if not intervals:
        raise ValueError("Counts CSV has no observations")
    return {"rows": reader.line_num - 1, "intervals": len(intervals),
            "totals": counts, "sources": sorted(sources),
            "site_ids": sorted(sites), "collection_dates": sorted(dates),
            "vehicle_classes": sorted(classes),
            "synchronised_site_intervals": sorted(
                set.intersection(*site_intervals.values())) if site_intervals else [],
            "observed_count_data": sources <= OBSERVED_SOURCES,
            "sha256": sha256(path.read_bytes())}


def inspect_count_zip(path: Path, name_filter: str = "") -> dict:
    """Read central-directory metadata; never expand the large LTA archive."""
    with zipfile.ZipFile(path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        matched = [info for info in files if name_filter.lower() in info.filename.lower()]
        candidates = [info for info in matched if PurePosixPath(info.filename).suffix.lower()
                      in {".csv", ".txt", ".xls", ".xlsx"}]
        return {"archive_bytes": path.stat().st_size,
                "archive_sha256": file_sha256(path),
                "member_count": len(files),
                "total_uncompressed_bytes": sum(info.file_size for info in files),
                "matching_count": len(matched), "extractable_tabular_count": len(candidates),
                "matched_members_preview": [
                    {"name": info.filename, "uncompressed_bytes": info.file_size,
                     "within_20_mb_extract_cap": info.file_size <= COUNT_MEMBER_LIMIT}
                    for info in matched[:30]],
                "note": "Inspection only; .gdb requires separate GIS conversion on a machine with enough space"}


def extract_count_member(zip_path: Path, member_name: str, output: Path) -> Path:
    """Extract one small tabular member by exact name, with a 20 MB hard cap."""
    member = PurePosixPath(member_name)
    if (member.is_absolute() or ".." in member.parts or "\\" in member_name
            or ":" in member_name or member.suffix.lower() not in {".csv", ".txt", ".xls", ".xlsx"}):
        raise ValueError("Select one CSV/TXT/XLS/XLSX member by its exact safe ZIP name")
    with zipfile.ZipFile(zip_path) as archive:
        info = archive.getinfo(member_name)
        if info.is_dir() or info.file_size > COUNT_MEMBER_LIMIT:
            raise ValueError("Selected member is a directory or exceeds the 20 MB cap")
        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
            raise ValueError("ZIP symlink members are not allowed")
        output.mkdir(parents=True, exist_ok=True)
        destination = output / member.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        with archive.open(info) as source:
            contents = source.read(COUNT_MEMBER_LIMIT + 1)
        if len(contents) > COUNT_MEMBER_LIMIT:
            raise ValueError("Decompressed member exceeded the 20 MB cap")
        with destination.open("xb") as target:
            target.write(contents)
    return destination


def _midpoint(edge) -> tuple[float, float]:
    shape = edge.getShape()
    return ((shape[0][0] + shape[-1][0]) / 2,
            (shape[0][1] + shape[-1][1]) / 2)


def _pedestrian_pairs(net) -> list[tuple[object, object, object]]:
    """Find walks whose shortest route explicitly includes a mapped crossing."""
    pairs = []
    for crossing in net.getEdges(withInternal=True):
        if crossing.getFunction() != "crossing":
            continue
        starts = set()
        ends = set()
        for before in crossing.getIncoming():
            earlier = before.getIncoming() if before.getFunction() == "walkingarea" else {before: None}
            starts.update(e for e in earlier if e.getFunction() == "" and e.allows("pedestrian"))
        for after in crossing.getOutgoing():
            later = after.getOutgoing() if after.getFunction() == "walkingarea" else {after: None}
            ends.update(e for e in later if e.getFunction() == "" and e.allows("pedestrian"))
        for start in sorted(starts, key=lambda e: e.getID()):
            for end in sorted(ends, key=lambda e: e.getID()):
                if start == end:
                    continue
                route, _ = net.getShortestPath(start, end, vClass="pedestrian", withInternal=True)
                if route and crossing in route:
                    pairs.append((crossing, start, end))
                    break
            else:
                continue
            break
    return pairs


def _vehicle_routes(net, focal, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    centre = _midpoint(focal)
    nearby = [e for e in net.getEdges(withInternal=False)
              if e.getFunction() == "" and e.allows("passenger")
              and math.dist(_midpoint(e), centre) < 350 and e.getLength() > 12]
    nearby.sort(key=lambda e: e.getID())
    if len(nearby) < 2:
        raise RuntimeError("Too few car edges near the candidate crossing")
    routes = []
    seen = set()
    for _ in range(500):
        start, end = rng.sample(nearby, 2)
        route, distance = net.getShortestPath(start, end, vClass="passenger")
        if route is None or not 120 <= distance <= 900 or len(route) < 3:
            continue
        if min(math.dist(_midpoint(e), centre) for e in route) > 90:
            continue
        ids = tuple(e.getID() for e in route if e.getFunction() == "")
        if ids not in seen:
            routes.append(list(ids))
            seen.add(ids)
        if len(routes) == 4:
            break
    if len(routes) < 2:
        raise RuntimeError("Could not find two plausible car routes near a crossing")
    return routes


def run_geometry_smoke(net_file: Path, output: Path, seed: int = 38) -> dict:
    """Move synthetic users on imported geometry; this is not an RL evaluation."""
    try:
        import sumolib
        import traci
    except ImportError as error:
        raise RuntimeError("Install requirements.txt or put SUMO tools on PYTHONPATH") from error
    net = sumolib.net.readNet(str(net_file), withInternal=True)
    pairs = _pedestrian_pairs(net)
    if not pairs:
        raise RuntimeError("No routable pedestrian crossing in imported network")
    focal, _, _ = pairs[0]
    routes = _vehicle_routes(net, focal, seed)
    rng = random.Random(seed)
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="smoke_car", vClass="passenger", sigma="0")
    ET.SubElement(root, "vType", id="smoke_walker", vClass="pedestrian", maxSpeed="1.3")
    departures = []
    for index in range(24):
        departures.append((rng.randint(0, 180), f"sgv{index:02d}", "vehicle",
                           routes[index % len(routes)]))
    for index in range(12):
        _, start, end = pairs[index % len(pairs)]
        departures.append((rng.randint(0, 180), f"sgp{index:02d}", "person",
                           (start.getID(), end.getID())))
    for depart, ident, kind, path in sorted(departures):
        if kind == "vehicle":
            item = ET.SubElement(root, "vehicle", id=ident, type="smoke_car", depart=str(depart))
            ET.SubElement(item, "route", edges=" ".join(path))
        else:
            item = ET.SubElement(root, "person", id=ident, type="smoke_walker", depart=str(depart))
            ET.SubElement(item, "walk", **{"from": path[0], "to": path[1]})
    route_file = output / "geometry-smoke.rou.xml"
    ET.indent(root)
    ET.ElementTree(root).write(route_file, encoding="utf-8", xml_declaration=True)
    config = ET.Element("configuration")
    inputs = ET.SubElement(config, "input")
    ET.SubElement(inputs, "net-file", value=net_file.name)
    ET.SubElement(inputs, "route-files", value=route_file.name)
    time = ET.SubElement(config, "time")
    ET.SubElement(time, "begin", value="0")
    ET.SubElement(time, "end", value="450")
    config_file = output / "geometry-smoke.sumocfg"
    ET.indent(config)
    ET.ElementTree(config).write(config_file, encoding="utf-8", xml_declaration=True)
    crossing_ids = {e.getID() for e in net.getEdges(withInternal=True)
                    if e.getFunction() == "crossing"}
    completed = set()
    departed = set()
    crossing_people = set()
    persons_seen = set()
    trip_file = output / "geometry-smoke.tripinfo.xml"
    traci.start([executable("sumo"), "-c", str(config_file), "--seed", str(seed),
                 "--no-step-log", "true", "--time-to-teleport", "-1",
                 "--tripinfo-output", str(trip_file)], label="singapore_geometry_smoke")
    connection = traci.getConnection("singapore_geometry_smoke")
    try:
        for _ in range(450):
            connection.simulationStep()
            departed.update(connection.simulation.getDepartedIDList())
            completed.update(connection.simulation.getArrivedIDList())
            people = connection.person.getIDList()
            persons_seen.update(people)
            crossing_people.update(p for p in people
                                   if connection.person.getRoadID(p) in crossing_ids)
    finally:
        connection.close()
    result = {"purpose": "geometry-only synthetic-demand smoke test",
              "seed": seed, "vehicle_routes": len(routes), "pedestrian_crossing_routes": len(pairs),
              "scheduled_vehicles": 24, "departed_vehicles": len(departed),
              "completed_vehicles": len(completed), "scheduled_pedestrians": 12,
              "pedestrians_seen": len(persons_seen),
              "pedestrians_entering_crossings": len(crossing_people),
              "focal_crossing": focal.getID(),
              "route_sha256": sha256(route_file.read_bytes()),
              "controller": "SUMO imported default; not a validated study controller"}
    (output / "geometry-smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not departed or not completed or not crossing_people:
        raise RuntimeError("Geometry smoke did not move both modes; inspect geometry-smoke.json")
    return result


def load_transfer_models(model_dir: Path, legacy_models: bool = False,
                         all_replicas: bool = False) -> list[dict]:
    """Load the full trained set, or replica zero for a short illustrative pilot."""
    from .ppo import PPOModel

    selected = []
    if legacy_models and all_replicas:
        raise ValueError("Legacy policies cannot be used for the full-replica transfer")
    if legacy_models:
        for method in ("ippo", "mappo"):
            for objective in ("vehicle", "pedestrian"):
                label = f"legacy-{method}-{objective}"
                path = model_dir / f"{method}-{objective}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing legacy pilot policy: {path}")
                selected.append((label, path, method, objective,
                                 0.0 if objective == "vehicle" else 3.0, None))
    else:
        manifest_path = model_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No training manifest at {manifest_path}; train the selected replicas first, "
                "or explicitly select --legacy-models --model-dir models for an exploratory pilot")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 2:
            raise ValueError("Transfer test requires the version-2 manifest schema")
        entries = [item for item in manifest.get("models", [])
                   if item.get("method") in ("ippo", "mappo")]
        if all_replicas:
            settings = manifest.get("settings", {})
            expected = {(method, float(weight), index)
                        for method in ("ippo", "mappo")
                        for weight in settings.get("reward_weights", [])
                        for index in range(len(settings.get("training_run_seeds", [])))}
            actual = {(item["method"], float(item["reward_weight"]),
                       int(item["training_run"])) for item in entries}
            if (not manifest.get("complete_protocol") or not expected
                    or actual != expected or len(entries) != len(expected)):
                raise ValueError("Full transfer requires every trained method, weight, and replica")
        else:
            entries = [item for item in entries if item.get("training_run") == 0]
            positive = [float(item["reward_weight"]) for item in entries
                        if float(item["reward_weight"]) > 0]
            if not positive:
                raise ValueError("No replica-zero pedestrian-reward model is available")
            pedestrian_weight = min(positive)
        for item in entries:
            weight = float(item["reward_weight"])
            if not all_replicas and weight not in (0.0, pedestrian_weight):
                continue
            path = model_dir / Path(item["model_path"]).name
            label = f"{item['method']}-w{weight:g}-r{item['training_run']}"
            if not path.is_file():
                raise FileNotFoundError(f"Manifest policy is missing: {path}")
            selected.append((label, path, item["method"], item["objective"],
                             weight, item.get("training_seed")))
        if (not all_replicas and
                (len(selected) != 4 or len({item[0] for item in selected}) != 4)):
            raise ValueError("Need replica zero for IPPO and MAPPO at both reward weights")
    output = []
    for label, path, method, objective, weight, training_seed in selected:
        model = PPOModel.load(path)
        if (model.method != method or model.objective != objective
                or model.junction_ids != ["C0", "C1", "C2"]
                or model.episodes_trained < 1):
            raise ValueError(f"Incompatible or untrained policy: {path}")
        output.append({"label": label, "model": model, "model_path": str(path.resolve()),
                       "model_sha256": file_sha256(path), "method": method,
                       "objective": objective, "reward_weight": weight,
                       "training_seed": training_seed,
                       "episodes_trained": model.episodes_trained})
    return sorted(output, key=lambda item: item["label"])


def derived_case_spec(site_audit: dict,
                      derived_config: Path = ROOT / "config" / "singapore_derived.json"
                      ) -> tuple[dict, dict, dict]:
    """Validate the saved approximate geometry against the OSM candidate audit."""
    case = json.loads(derived_config.read_text(encoding="utf-8"))
    candidates = site_audit["network"]["candidate_three_signal_chains"]
    selected = candidates[case["candidate_chain_index"]]
    observed = selected["centroid_spacings_m"]
    declared = case["junction_spacings_m"]
    if len(declared) != 2 or any(abs(round(value) - target) > 1
                                 for value, target in zip(observed, declared)):
        raise ValueError("Saved site spacings disagree with the selected OSM candidate")
    if not case["left_hand"] or case["include_compatible_turns"]:
        raise ValueError("This pilot requires left-hand, straight-only vehicle connections")
    site_centroids = selected.get("site_centroids_m")
    declared_centroids = case.get("junction_site_centroids_m")
    if (not site_centroids or not declared_centroids or len(site_centroids) != 3
            or len(declared_centroids) != 3
            or any(len(actual) != 2 or len(saved) != 2 or
                   math.dist(actual, saved) > 0.01
                   for actual, saved in zip(site_centroids, declared_centroids))):
        raise ValueError("Saved candidate-site centroids disagree with the selected OSM audit")
    overrides = {"junction_spacings_m": declared, "left_hand": True,
                 "junction_site_centroids_m": declared_centroids,
                 "include_compatible_turns": False,
                 "pedestrian_count": int(case["pedestrian_count"])}
    return case, selected, overrides


def run_derived_baselines(site_audit: dict, output: Path,
                          derived_config: Path = ROOT / "config" / "singapore_derived.json",
                          model_dir: Path | None = None,
                          legacy_models: bool = False) -> dict:
    """Use approximate candidate-site positions in a protected synthetic corridor."""
    from .cli import run
    from .viewer import write_replay

    case, selected, overrides = derived_case_spec(site_audit, derived_config)
    observed = selected["centroid_spacings_m"]
    declared = case["junction_spacings_m"]
    site_centroids = selected["site_centroids_m"]
    base_config = ROOT / case["base_experiment_config"]
    policies = load_transfer_models(model_dir, legacy_models) if model_dir else []
    summaries = {}
    replays = {}
    hashes = set()
    runs = [(name, None) for name in ("fixed", "actuated", "pressure")]
    runs.extend((item["label"], item) for item in policies)
    for controller, policy_item in runs:
        policy_callback = (None if policy_item is None else
                           lambda tls, obs, global_obs, eligible, model=policy_item["model"]:
                           model.select(tls, obs, global_obs, eligible, training=False))
        metrics = run("policy" if policy_item else controller, case["seed"],
                      output / "derived" / controller, base_config, junctions=3,
                      config_overrides=overrides, policy_callback=policy_callback,
                      objective=policy_item["objective"] if policy_item else "vehicle",
                      pedestrian_reward_weight=(policy_item["reward_weight"]
                                                if policy_item else 0.0))
        _, replay = write_replay(output / "derived" / controller,
                                 output / "derived" / controller / "replay")
        replays[controller] = str(replay.relative_to(output))
        hashes.add(metrics["demand_sha256"])
        summaries[controller] = {
            "kind": "learned" if policy_item else "conventional",
            "metrics_file": str((output / "derived" / controller / "metrics.json").relative_to(output)),
            "completed_vehicles": metrics["vehicles"]["completed_count"],
            "unfinished_vehicles": metrics["vehicles"]["unfinished_count"],
            "uninserted_vehicles": metrics["vehicles"]["uninserted_count"],
            "completed_mean_vehicle_time_loss_seconds":
                metrics["vehicles"]["completed_mean_time_loss_seconds"],
            "protected_crossings": len(metrics["pedestrians_by_crossing"]),
            "crossing_entries": sum(value["entered_count"]
                                    for value in metrics["pedestrians_by_crossing"].values()),
            "crossings_without_entries": sorted(crossing for crossing, value in
                                                metrics["pedestrians_by_crossing"].items()
                                                if value["entered_count"] == 0),
            "pending_crossing_requests": sum(value["unserved_count"]
                                             for value in metrics["pedestrians_by_crossing"].values()),
            "per_crossing_wait_seconds": {
                crossing: {"entries": value["entered_count"],
                           "mean": value["mean_wait_seconds"],
                           "p95": value["p95_wait_seconds"],
                           "unserved": value["unserved_count"]}
                for crossing, value in sorted(metrics["pedestrians_by_crossing"].items())},
        }
        if policy_item:
            summaries[controller]["policy"] = {
                key: policy_item[key] for key in ("model_path", "model_sha256", "method",
                                                  "objective", "reward_weight", "training_seed",
                                                  "episodes_trained")}
    if len(hashes) != 1:
        raise AssertionError("Controller demand was not identical")
    if any(row["completed_vehicles"] == 0 or row["crossing_entries"] == 0
           for row in summaries.values()):
        raise AssertionError("Derived corridor did not move both modes")
    result = {"purpose": "NUS candidate-centroid simplified corridor transfer pilot"
                         if policies else "NUS candidate-centroid simplified corridor baseline smoke",
              "simulation_status": "approximate OSM-derived candidate-site centroids; straight synthetic connectors, synthetic side arms, synthetic crossings, synthetic demand; uncalibrated",
              "site_groups": selected["signal_groups"],
              "candidate_site_centroids_m": site_centroids,
              "observed_centroid_spacings_m": observed,
              "modeled_junction_spacings_m": declared,
              "source_osm_xml_sha256": site_audit["osm"]["xml_sha256"],
              "demand_sha256": hashes.pop(), "seed": case["seed"],
              "assumed_geometry": ["straight line segments between approximate signal-site centroids",
                                   "synthetic perpendicular 110 m side arms",
                                   "one vehicle lane per approach", "2 m sidewalks",
                                   "4 m crossing width", "straight vehicle movements only"],
              "controllers": summaries, "phase_replays": replays,
              "safety": "shared legal phase machine and runtime conflict/clearance assertions"}
    if policies:
        result["transfer_status"] = (
            "EXPLORATORY LEGACY POLICY PILOT: older policies, not v2 research evaluation"
            if legacy_models else
            "UNCALIBRATED GEOMETRY TRANSFER: replica-zero policies from "
            f"{model_dir.name}, one synthetic seed; not a Singapore performance estimate")
        result["model_dir"] = str(model_dir.resolve())
    (output / "derived-baselines.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def prepare(site_file: Path, output: Path, osm_file: Path | None = None,
            fetch_osm_data: bool = False, signal_geojson: Path | None = None,
            fetch_lta: bool = False, counts_file: Path | None = None,
            smoke: bool = False, seed: int = 38,
            derived_baselines: bool = False, model_dir: Path | None = None,
            legacy_models: bool = False) -> dict:
    site = load_site(site_file)
    if fetch_osm_data and osm_file is not None:
        raise ValueError("Choose --fetch-osm or --osm, not both")
    if fetch_lta and signal_geojson is not None:
        raise ValueError("Choose --fetch-lta-signals or --signal-geojson, not both")
    output.mkdir(parents=True, exist_ok=True)
    source_urls = {}
    if fetch_osm_data:
        data, url = fetch_osm(site["bbox_west_south_east_north"], site["max_osm_bytes"])
        osm_file = output / "source.osm.xml"
        osm_file.write_bytes(data)
        source_urls["osm"] = url
    elif osm_file is None:
        osm_file = ROOT / site["osm_snapshot"]
        source_urls["osm"] = site["osm_source_url"]
    if osm_file is None or not osm_file.is_file():
        raise FileNotFoundError("No OSM snapshot is available; provide --osm or --fetch-osm")
    net_file = output / "singapore.net.xml"
    build_details = build_sumo_network(osm_file, net_file)
    osm_bytes = osm_file.read_bytes()
    report = {"site": site["name"], "bbox_west_south_east_north": site["bbox_west_south_east_north"],
              "osm": {"file_sha256": sha256(osm_bytes),
                      "xml_sha256": sha256(gzip.decompress(osm_bytes)
                                            if osm_file.suffix == ".gz" else osm_bytes),
                      **osm_audit(osm_file)},
              "network": {"sha256": sha256(net_file.read_bytes()), **build_details,
                          **network_audit(net_file, site["max_signal_link_metres"])},
              "source_urls": source_urls,
              "control_status": "NOT VALIDATED: imported signals and crossings require a manual conflict audit",
              "demand_status": "UNCALIBRATED: no vehicle or pedestrian counts supplied"}
    if report["network"]["sumo_signalised_crossings"] == 0:
        report["control_status"] += "; no protected pedestrian crossing was imported"
    if fetch_lta:
        full, full_hash = fetch_lta_signal_aspects(site["max_lta_geojson_bytes"])
        clipped = clip_points(full, site["bbox_west_south_east_north"])
        signal_geojson = output / "lta-signal-aspects-clipped.geojson"
        signal_geojson.write_text(json.dumps(clipped, indent=2), encoding="utf-8")
        report["lta_full_geojson_sha256"] = full_hash
        source_urls["lta_signal_aspects"] = site["lta_signal_aspects_dataset_url"]
    elif signal_geojson is None:
        signal_geojson = ROOT / site["lta_signal_aspects_snapshot"]
        report["lta_full_geojson_sha256"] = site["lta_full_geojson_sha256"]
        source_urls["lta_signal_aspects"] = site["lta_signal_aspects_dataset_url"]
    if signal_geojson is not None:
        signal_data = json.loads(signal_geojson.read_text(encoding="utf-8"))
        clipped = clip_points(signal_data, site["bbox_west_south_east_north"])
        report["lta_signal_aspects"] = {"sha256": sha256(signal_geojson.read_bytes()),
                                         "features_in_window": len(clipped["features"]),
                                         "pedestrian_signal_aspects": sum(
                                             "pedestrian" in f.get("properties", {}).get("TYP_NAM", "").lower()
                                             for f in clipped["features"])}
    if counts_file is not None:
        report["counts"] = audit_counts(counts_file)
        report["demand_status"] = ("OBSERVED COUNTS INTAKE ONLY: route generation and calibration pending"
                                   if report["counts"]["observed_count_data"] else
                                   "MIXED OR SYNTHETIC COUNTS INTAKE ONLY: not field-calibrated")
    if smoke:
        report["geometry_smoke"] = run_geometry_smoke(net_file, output, seed)
    if derived_baselines:
        report["derived_baselines"] = run_derived_baselines(
            report, output, model_dir=model_dir, legacy_models=legacy_models)
    report_file = output / "case-audit.json"
    report_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "config" / "singapore.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "singapore-case")
    parser.add_argument("--osm", type=Path, help="Previously downloaded, versioned OSM XML")
    parser.add_argument("--fetch-osm", action="store_true", help="Fetch only the configured small OSM box")
    parser.add_argument("--signal-geojson", type=Path, help="Local LTA signal-aspect GeoJSON")
    parser.add_argument("--fetch-lta-signals", action="store_true",
                        help="Fetch the public signal-aspect GeoJSON, then retain only the local clip")
    parser.add_argument("--counts", type=Path, help="Optional locally observed count CSV")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a synthetic-demand geometry smoke, not a controller comparison")
    parser.add_argument("--seed", type=int, default=38, help="Geometry-smoke demand seed")
    parser.add_argument("--derived-baselines", action="store_true",
                        help="Run safe baselines on an approximate candidate-centroid corridor")
    parser.add_argument("--model-dir", type=Path,
                        help="Add replica-zero IPPO/MAPPO transfer pilot from this model directory")
    parser.add_argument("--legacy-models", action="store_true",
                        help="Explicitly label --model-dir models policies as exploratory legacy models")
    parser.add_argument("--count-zip", type=Path, help="Inspect a locally supplied LTA count ZIP only")
    parser.add_argument("--count-filter", default="", help="ZIP member-name substring for inspection")
    parser.add_argument("--count-member", help="Extract one exact tabular ZIP member, capped at 20 MB")
    args = parser.parse_args()
    if args.count_member and not args.count_zip:
        parser.error("--count-member requires --count-zip")
    if args.model_dir and not args.derived_baselines:
        parser.error("--model-dir requires --derived-baselines")
    if args.legacy_models and not args.model_dir:
        parser.error("--legacy-models requires --model-dir")
    if args.count_zip:
        index = inspect_count_zip(args.count_zip, args.count_filter)
        if args.count_member:
            extracted = extract_count_member(args.count_zip, args.count_member,
                                             args.output / "count-intake")
            index["extracted_member"] = str(extracted)
        print(json.dumps(index, indent=2))
        return
    result = prepare(args.site, args.output, args.osm, args.fetch_osm,
                     args.signal_geojson, args.fetch_lta_signals, args.counts,
                     args.smoke, args.seed, args.derived_baselines,
                     args.model_dir, args.legacy_models)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
