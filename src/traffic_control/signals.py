"""Shared legal phases and transition timing for every controller."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


GROUPS = ("EW", "NS", "PED")


def load_signal_conflicts(network_file: Path, tls_id: str) -> set[tuple[int, int]]:
    """Translate SUMO junction foes into pairs of TLS-controlled link indices.

    The two indices usually agree for an isolated signal, but are different
    concepts in SUMO. In particular, ``foes`` bits are right-to-left whereas
    the signal-state string is left-to-right. We explicitly map through the
    junction's incoming-lane and connection order and fail on an incomplete
    mapping instead of certifying an unexamined phase as safe.
    """
    root = ET.parse(network_file).getroot()
    junction = next((node for node in root.findall("junction")
                     if node.get("id") == tls_id), None)
    if junction is None:
        raise ValueError(f"Cannot inspect signal {tls_id}: no matching junction")
    edge_function = {edge.get("id"): edge.get("function") for edge in root.findall("edge")}
    connections = root.findall("connection")
    ordered = []
    for incoming_lane in junction.get("incLanes", "").split():
        for connection in connections:
            if f"{connection.get('from')}_{connection.get('fromLane')}" != incoming_lane:
                continue
            from_function = edge_function.get(connection.get("from"))
            to_function = edge_function.get(connection.get("to"))
            if to_function == "walkingarea" or (from_function == "walkingarea"
                                                   and to_function != "crossing"):
                continue
            ordered.append(connection)
    requests = {int(request.attrib["index"]): request.attrib["foes"]
                for request in junction.findall("request")}
    if len(ordered) != len(requests) or set(requests) != set(range(len(ordered))):
        raise ValueError(f"Cannot map all junction conflicts for {tls_id}: "
                         f"{len(ordered)} connections, {len(requests)} requests")
    tls_indices = {}
    for request_index, connection in enumerate(ordered):
        if connection.get("tl") != tls_id:
            raise ValueError(f"Junction {tls_id} has a movement outside its signal plan")
        indices = {int(connection.attrib["linkIndex"])}
        if "linkIndex2" in connection.attrib:
            secondary = int(connection.attrib["linkIndex2"])
            if secondary >= 0:
                indices.add(secondary)
        tls_indices[request_index] = indices
    conflicts = set()
    for request_index, foes in requests.items():
        if len(foes) != len(ordered):
            raise ValueError(f"Malformed SUMO foes bitset at {tls_id} request {request_index}")
        for foe_index, flag in enumerate(reversed(foes)):
            if flag != "1":
                continue
            for left in tls_indices[request_index]:
                for right in tls_indices[foe_index]:
                    if left == right:
                        raise ValueError(f"Conflicting movements share signal index {left}")
                    conflicts.add(tuple(sorted((left, right))))
    return conflicts


def classify_links(controlled_links, crossing_ids: set[str],
                   incoming_groups: dict[str, set[str]] | None = None) -> dict[str, set[int]]:
    incoming_groups = incoming_groups or {"EW": {"wi", "ei"}, "NS": {"ni", "si"}}
    groups = {name: set() for name in GROUPS}
    for index, links in enumerate(controlled_links):
        if not links:
            continue
        labels = set()
        for incoming, outgoing, via in links:
            lanes = (incoming, outgoing, via)
            if any(lane.startswith(crossing + "_") or lane == crossing
                   for lane in lanes for crossing in crossing_ids):
                labels.add("PED")
            elif incoming.rsplit("_", 1)[0] in incoming_groups["EW"]:
                labels.add("EW")
            elif incoming.rsplit("_", 1)[0] in incoming_groups["NS"]:
                labels.add("NS")
            else:
                raise ValueError(f"Unclassified controlled link {index}: {lanes}")
        if len(labels) != 1:
            raise ValueError(f"Signal index {index} controls mixed movements: {labels}")
        groups[labels.pop()].add(index)
    if any(not indices for indices in groups.values()):
        raise ValueError(f"Expected EW, NS and pedestrian-controlled links: {groups}")
    return groups


@dataclass
class SignalPlan:
    groups: dict[str, set[int]]
    link_count: int
    clearance: int
    config: dict
    conflicts: set[tuple[int, int]] | None = None

    def state(self, stage: str) -> str:
        chars = ["r"] * self.link_count
        if stage in GROUPS:
            for index in self.groups[stage]:
                chars[index] = "G"
        elif stage.endswith("_YELLOW"):
            source = stage.split("_")[0]
            for index in self.groups[source]:
                chars[index] = "y"
        elif stage not in ("ALL_RED", "PED_CLEARANCE"):
            raise ValueError(stage)
        state = "".join(chars)
        self.assert_safe(state)
        return state

    def assert_safe(self, state: str) -> None:
        if len(state) != self.link_count:
            raise AssertionError("Signal state length differs from controlled links")
        active = [name for name, indices in self.groups.items()
                  if any(state[i] in "Ggy" for i in indices)]
        if len(active) > 1:
            raise AssertionError(f"Conflicting moving groups: {active}")
        for left, right in self.conflicts or ():
            if not (0 <= left < self.link_count and 0 <= right < self.link_count):
                raise AssertionError("SUMO foe index lies outside the signal state")
            if state[left] in "Ggy" and state[right] in "Ggy":
                raise AssertionError(f"SUMO junction foes {left} and {right} both moving")


class Controller:
    """Shared phase machine; callers can request early vehicle-phase changes."""

    def __init__(self, plan: SignalPlan, kind: str):
        if kind not in ("fixed", "actuated", "policy", "pressure"):
            raise ValueError(kind)
        self.plan = plan
        self.kind = kind
        self.stage = "EW"
        self.elapsed = 0
        self.next_green = "NS"
        self.clearance_empty_steps = 0
        self.history = [(0, self.stage)]

    def _enter(self, stage: str, time: int) -> None:
        self.stage = stage
        self.elapsed = 0
        if stage == "PED_CLEARANCE":
            self.clearance_empty_steps = 0
        self.history.append((time, stage))

    def tick(self, time: int, pending_pedestrian: bool,
             crossing_occupied: bool = False, action: int = 0) -> str:
        """Call after each one-second SUMO step; returns state for the next step."""
        self.elapsed += 1
        cfg = self.plan.config
        if self.stage in ("EW", "NS"):
            green_limit = (cfg["vehicle_green_fixed_seconds"] if self.kind == "fixed"
                           else cfg["vehicle_green_actuated_max_seconds"])
            should_advance = self.elapsed >= green_limit
            if self.kind == "actuated" and pending_pedestrian:
                should_advance = self.elapsed >= cfg["vehicle_green_min_seconds"]
            if self.kind in ("policy", "pressure") and action == 1:
                should_advance = self.elapsed >= cfg["vehicle_green_min_seconds"]
            if should_advance:
                self._enter(self.stage + "_YELLOW", time)
        elif self.stage.endswith("_YELLOW"):
            if self.elapsed >= cfg["yellow_seconds"]:
                self._enter("ALL_RED", time)
        elif self.stage == "ALL_RED":
            if self.elapsed >= cfg["all_red_seconds"]:
                self._enter(self.next_green, time)
        elif self.stage == "PED":
            if self.elapsed >= cfg["pedestrian_green_seconds"]:
                self._enter("PED_CLEARANCE", time)
        elif self.stage == "PED_CLEARANCE":
            self.clearance_empty_steps = (0 if crossing_occupied
                                          else self.clearance_empty_steps + 1)
            if (self.elapsed >= self.plan.clearance
                    and self.clearance_empty_steps >= cfg["pedestrian_clearance_empty_seconds"]):
                self._enter("EW", time)
        else:
            raise AssertionError(self.stage)

        if self.stage == "EW_YELLOW":
            self.next_green = "NS"
        elif self.stage == "NS_YELLOW":
            self.next_green = "PED" if self.kind == "fixed" or pending_pedestrian else "EW"
        return self.plan.state(self.stage)
