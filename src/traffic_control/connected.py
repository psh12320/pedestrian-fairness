"""New three-junction corridor in SUMO plain XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def write_network(directory: Path) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    edges = ET.Element("edges")
    connections = ET.Element("connections")
    for i in range(3):
        x = (i - 1) * 140
        ET.SubElement(nodes, "node", id=f"C{i}", x=str(x), y="0", type="traffic_light")
        ET.SubElement(nodes, "node", id=f"N{i}", x=str(x), y="110")
        ET.SubElement(nodes, "node", id=f"S{i}", x=str(x), y="-110")
        for direction, source, target in (("ni", f"N{i}", f"C{i}"),
                                           ("no", f"C{i}", f"N{i}"),
                                           ("si", f"S{i}", f"C{i}"),
                                           ("so", f"C{i}", f"S{i}")):
            ET.SubElement(edges, "edge", id=f"{direction}{i}", **{"from": source, "to": target},
                          numLanes="1", speed="13.9", priority="2", sidewalkWidth="2")
    ET.SubElement(nodes, "node", id="W", x="-280", y="0")
    ET.SubElement(nodes, "node", id="E", x="280", y="0")
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


def demand_routes(junctions: int) -> tuple[list[tuple[str, ...]], list[tuple[str, str]]]:
    if junctions == 1:
        return ([('wi', 'eo'), ('ei', 'wo'), ('ni', 'so'), ('si', 'no')],
                [('wi', 'no'), ('ni', 'eo'), ('ei', 'so'), ('si', 'wo')])
    if junctions == 3:
        vehicles = [tuple(f"h{i}e" for i in range(4)),
                    tuple(f"h{i}w" for i in range(3, -1, -1))]
        people = []
        for i in range(3):
            vehicles.extend([(f"ni{i}", f"so{i}"), (f"si{i}", f"no{i}")])
            people.extend([(f"h{i}e", f"no{i}"), (f"ni{i}", f"h{i+1}e"),
                           (f"h{i+1}w", f"so{i}"), (f"si{i}", f"h{i}w")])
        return vehicles, people
    raise ValueError("Only one or three junctions are implemented")
