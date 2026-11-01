"""New three-junction corridor in SUMO plain XML."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path


def _candidate_layout(points: list[list[float]]) -> tuple[list[tuple[float, float]],
                                                           tuple[float, float],
                                                           tuple[float, float],
                                                           list[tuple[float, float]]]:
    """Place synthetic arms around three approximate signal-site centroids."""
    if len(points) != 3 or any(len(point) != 2 for point in points):
        raise ValueError("Provide three 2D signal-site centroids")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) for point in points for value in point):
        raise ValueError("Signal-site centroids must contain finite coordinates")
    centers = [(float(x), float(y)) for x, y in points]
    vectors = [(b[0] - a[0], b[1] - a[1])
               for a, b in zip(centers, centers[1:])]
    distances = [math.hypot(x, y) for x, y in vectors]
    if any(length < 60 for length in distances):
        raise ValueError("Candidate signal-site centroids must be at least 60 m apart")
    directions = [(x / length, y / length)
                  for (x, y), length in zip(vectors, distances)]
    middle = (directions[0][0] + directions[1][0],
              directions[0][1] + directions[1][1])
    middle_length = math.hypot(*middle)
    if middle_length < 0.5:
        raise ValueError("A sharp reversal cannot use this simplified corridor layout")
    tangents = [directions[0],
                (middle[0] / middle_length, middle[1] / middle_length),
                directions[1]]
    west = (centers[0][0] - directions[0][0] * distances[0],
            centers[0][1] - directions[0][1] * distances[0])
    east = (centers[2][0] + directions[1][0] * distances[1],
            centers[2][1] + directions[1][1] * distances[1])
    return centers, west, east, tangents


def write_network(directory: Path, spacing_m: int | list[int] | list[list[float]] = 140,
                  left_hand: bool = False,
                  include_compatible_turns: bool = True) -> tuple[Path, Path, Path]:
    """Write the default corridor or an optional three-centroid derived layout.

    Three coordinate pairs are used only by the Singapore transfer pilot.
    The generated road segments, arms, crossings and traffic are synthetic.
    """
    if (isinstance(spacing_m, list) and len(spacing_m) == 3
            and all(isinstance(point, (list, tuple)) for point in spacing_m)):
        centers_xy, west, east, tangents = _candidate_layout(spacing_m)
    else:
        spacings = [spacing_m, spacing_m] if isinstance(spacing_m, int) else list(spacing_m)
        if len(spacings) != 2 or any(s < 60 for s in spacings):
            raise ValueError("Provide two junction spacings of at least 60 m")
        # Keep the original 140 m synthetic corridor byte-identical to the
        # geometry used by the long-running training process.
        centers = [-spacings[0], 0, spacings[1]]
        centers_xy = [(x, 0) for x in centers]
        west = (-2 * spacings[0], 0)
        east = (2 * spacings[1], 0)
        tangents = [(1, 0)] * 3
    directory.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    edges = ET.Element("edges")
    connections = ET.Element("connections")
    for i in range(3):
        x, y = centers_xy[i]
        tangent_x, tangent_y = tangents[i]
        arm_x, arm_y = -110 * tangent_y, 110 * tangent_x
        ET.SubElement(nodes, "node", id=f"C{i}", x=str(x), y=str(y), type="traffic_light")
        ET.SubElement(nodes, "node", id=f"N{i}", x=str(x + arm_x), y=str(y + arm_y))
        ET.SubElement(nodes, "node", id=f"S{i}", x=str(x - arm_x), y=str(y - arm_y))
        for direction, source, target in (("ni", f"N{i}", f"C{i}"),
                                           ("no", f"C{i}", f"N{i}"),
                                           ("si", f"S{i}", f"C{i}"),
                                           ("so", f"C{i}", f"S{i}")):
            ET.SubElement(edges, "edge", id=f"{direction}{i}", **{"from": source, "to": target},
                          numLanes="1", speed="13.9", priority="2", sidewalkWidth="2")
    ET.SubElement(nodes, "node", id="W", x=str(west[0]), y=str(west[1]))
    ET.SubElement(nodes, "node", id="E", x=str(east[0]), y=str(east[1]))
    spine = ["W", "C0", "C1", "C2", "E"]
    for i in range(4):
        for suffix, source, target in (("e", spine[i], spine[i + 1]),
                                       ("w", spine[i + 1], spine[i])):
            ET.SubElement(edges, "edge", id=f"h{i}{suffix}", **{"from": source, "to": target},
                          numLanes="1", speed="13.9", priority="2", sidewalkWidth="2")
    for i in range(3):
        for source, target in ((f"h{i}e", f"h{i+1}e"),
                               (f"h{i+1}w", f"h{i}w"),
                               (f"ni{i}", f"so{i}"),
                               (f"si{i}", f"no{i}")):
            ET.SubElement(connections, "connection", **{"from": source, "to": target})
        if include_compatible_turns:
            # Only near-side turns are included; opposing-side turns need a
            # separate protected phase. Which turn is near-side follows the
            # network's driving side.
            turns = ((f"h{i}e", f"no{i}"), (f"h{i+1}w", f"so{i}"),
                     (f"ni{i}", f"h{i+1}e"), (f"si{i}", f"h{i}w")) if left_hand else (
                     (f"h{i}e", f"so{i}"), (f"h{i+1}w", f"no{i}"),
                     (f"ni{i}", f"h{i}w"), (f"si{i}", f"h{i+1}e"))
            for source, target in turns:
                ET.SubElement(connections, "connection", **{"from": source, "to": target})
        for pair in ((f"h{i}e", f"h{i}w"), (f"h{i+1}e", f"h{i+1}w"),
                     (f"ni{i}", f"no{i}"), (f"si{i}", f"so{i}")):
            ET.SubElement(connections, "crossing", node=f"C{i}", edges=" ".join(pair), width="4")
    result = []
    for name, root in (("connected.nod.xml", nodes), ("connected.edg.xml", edges),
                       ("connected.con.xml", connections)):
        ET.indent(root)
        path = directory / name
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        result.append(path)
    return tuple(result)


def incoming_groups(junctions: int) -> dict[str, dict[str, set[str]]]:
    if junctions == 1:
        return {"C": {"EW": {"wi", "ei"}, "NS": {"ni", "si"}}}
    if junctions == 3:
        return {f"C{i}": {"EW": {f"h{i}e", f"h{i+1}w"},
                          "NS": {f"ni{i}", f"si{i}"}} for i in range(3)}
    raise ValueError("Only one or three junctions are implemented")


def outgoing_groups(junctions: int) -> dict[str, dict[str, set[str]]]:
    if junctions == 1:
        return {"C": {"EW": {"wo", "eo"}, "NS": {"no", "so"}}}
    if junctions == 3:
        return {f"C{i}": {"EW": {f"h{i}w", f"h{i+1}e"},
                          "NS": {f"no{i}", f"so{i}"}} for i in range(3)}
    raise ValueError("Only one or three junctions are implemented")


def demand_routes(junctions: int, left_hand: bool = False) -> tuple[list[tuple[str, ...]], list[tuple[str, str]]]:
    if junctions == 1:
        return ([('wi', 'eo'), ('ei', 'wo'), ('ni', 'so'), ('si', 'no')],
                [('wi', 'no'), ('ni', 'eo'), ('ei', 'so'), ('si', 'wo')])
    if junctions == 3:
        vehicles = [tuple(f"h{i}e" for i in range(4)),
                    tuple(f"h{i}w" for i in range(3, -1, -1))]
        people = []
        for i in range(3):
            vehicles.extend([(f"ni{i}", f"so{i}"), (f"si{i}", f"no{i}")])
            if left_hand:
                people.extend([(f"h{i}e", f"so{i}"), (f"ni{i}", f"h{i}w"),
                               (f"h{i+1}w", f"no{i}"), (f"si{i}", f"h{i+1}e")])
            else:
                people.extend([(f"h{i}e", f"no{i}"), (f"ni{i}", f"h{i+1}e"),
                               (f"h{i+1}w", f"so{i}"), (f"si{i}", f"h{i}w")])
        return vehicles, people
    raise ValueError("Only one or three junctions are implemented")


def compatible_turn_routes(junctions: int, left_hand: bool = False) -> list[tuple[str, ...]]:
    """Optional near-side vehicle turns with no opposing-through conflict."""
    if junctions == 1:
        return []
    if junctions == 3:
        routes: list[tuple[str, ...]] = []
        for i in range(3):
            routes.extend(((f"h{i}e", f"no{i}"),
                           (f"h{i+1}w", f"so{i}"),
                           (f"ni{i}", f"h{i+1}e"),
                           (f"si{i}", f"h{i}w")) if left_hand else (
                           (f"h{i}e", f"so{i}"),
                           (f"h{i+1}w", f"no{i}"),
                           (f"ni{i}", f"h{i}w"),
                           (f"si{i}", f"h{i+1}e")))
        return routes
    raise ValueError("Only one or three junctions are implemented")
