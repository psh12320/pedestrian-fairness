"""Shared legal phases and transition timing for every controller."""

from __future__ import annotations

from dataclasses import dataclass


GROUPS = ("EW", "NS", "PED")


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
        active = [name for name, indices in self.groups.items()
                  if any(state[i] in "Gg" for i in indices)]
        if len(active) > 1:
            raise AssertionError(f"Conflicting green groups: {active}")
        if len(state) != self.link_count:
            raise AssertionError("Signal state length differs from controlled links")


class Controller:
    """Only this state machine may change lights; policy selects when to serve PED."""

    def __init__(self, plan: SignalPlan, kind: str):
        if kind not in ("fixed", "actuated", "policy"):
            raise ValueError(kind)
        self.plan = plan
        self.kind = kind
        self.stage = "EW"
        self.elapsed = 0
        self.next_green = "NS"
        self.history = [(0, self.stage)]

    def _enter(self, stage: str, time: int) -> None:
        self.stage = stage
        self.elapsed = 0
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
            if self.kind == "policy" and action == 1:
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
            if self.elapsed >= self.plan.clearance and not crossing_occupied:
                self._enter("EW", time)
        else:
            raise AssertionError(self.stage)

        if self.stage == "EW_YELLOW":
            self.next_green = "NS"
        elif self.stage == "NS_YELLOW":
            self.next_green = "PED" if self.kind == "fixed" or pending_pedestrian else "EW"
        return self.plan.state(self.stage)
