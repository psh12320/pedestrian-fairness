"""Event-based crossing waits and vehicle service accounting."""

from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CrossingEvent:
    person_id: str
    crossing_id: str
    reached_at: float
    entered_at: float | None = None
    arrival_source: str = "unknown"

    @property
    def wait(self) -> float | None:
        return None if self.entered_at is None else self.entered_at - self.reached_at


class CrossingTracker:
    """Track each person's arrival at a crossing entrance and crossing entry.

    The fourth person observation value is the geometric distance in metres to
    the nearest endpoint of the next crossing lane. On SUMO's generated network,
    a pedestrian stops about one metre from that endpoint because the walking
    lane has a lateral offset. The 1.2 m entrance zone includes this stop line.
    A stopped pedestrian farther away on the immediate crossing approach is a
    queued request; the first stopped interval begins one step before observation.
    SUMO waitingTime is only a queue trigger, not the reported crossing wait.
    Both entrance and entry timestamps have one simulation step resolution.
    """

    def __init__(self, crossing_ids: set[str], step_seconds: float = 1.0,
                 approach_distance_metres: float = 1.2):
        if step_seconds <= 0 or approach_distance_metres <= 0:
            raise ValueError("Step and crossing-entrance distance must be positive")
        self.crossing_ids = crossing_ids
        self.step_seconds = step_seconds
        self.approach_distance_metres = approach_distance_metres
        self.events: dict[tuple[str, str], CrossingEvent] = {}

    def observe(self, time: float,
                persons: dict[str, tuple[str, str, float] | tuple[str, str, float, float | None]]) -> None:
        for person_id, observation in persons.items():
            road, next_edge, stopped_seconds = observation[:3]
            distance = observation[3] if len(observation) == 4 else None
            if next_edge in self.crossing_ids and road != next_edge:
                key = (person_id, next_edge)
                if key not in self.events:
                    at_entrance = (distance is not None and math.isfinite(distance)
                                   and 0 <= distance <= self.approach_distance_metres)
                    if at_entrance or stopped_seconds > 0:
                        reached = max(0.0, time - self.step_seconds) if stopped_seconds > 0 else time
                        source = ("entrance_and_stopped" if at_entrance and stopped_seconds > 0 else
                                  "entrance" if at_entrance else "queued_approach")
                        self.events[key] = CrossingEvent(person_id, next_edge, reached,
                                                         arrival_source=source)
            if road in self.crossing_ids:
                key = (person_id, road)
                if key not in self.events:
                    self.events[key] = CrossingEvent(person_id, road, time, time,
                                                     "entry_without_approach")
                elif self.events[key].entered_at is None:
                    self.events[key].entered_at = time

    def pending(self) -> bool:
        return any(event.entered_at is None for event in self.events.values())

    def pending_for(self, crossing_ids: set[str]) -> bool:
        return any(event.entered_at is None and event.crossing_id in crossing_ids
                   for event in self.events.values())

    def summary(self, end_time: float) -> dict:
        out = {}
        for crossing in sorted(self.crossing_ids):
            events = [e for e in self.events.values() if e.crossing_id == crossing]
            waits = sorted(e.wait for e in events if e.wait is not None)
            outstanding = [e for e in events if e.entered_at is None]
            rank = max(0, math.ceil(0.95 * len(waits)) - 1)
            out[crossing] = {
                "entered_count": len(waits),
                "mean_wait_seconds": statistics.mean(waits) if waits else None,
                "p95_wait_seconds": waits[rank] if waits else None,
                "unserved_count": len(outstanding),
                "unserved_elapsed_seconds": [end_time - e.reached_at for e in outstanding],
                "entry_without_approach_count": sum(
                    e.arrival_source == "entry_without_approach" for e in events),
            }
        return out

    def event_rows(self) -> list[dict]:
        return [dict(asdict(e), wait_seconds=e.wait) for e in sorted(
            self.events.values(), key=lambda e: (e.crossing_id, e.person_id))]


def vehicle_summary(tripinfo_file: Path, scheduled_ids: set[str],
                    departed_ids: set[str], arrived_ids: set[str]) -> dict:
    if arrived_ids - departed_ids or departed_ids - scheduled_ids:
        raise AssertionError("Inconsistent vehicle accounting")
    tripinfos = {}
    for item in ET.parse(tripinfo_file).iter("tripinfo"):
        tripinfos[item.attrib["id"]] = item.attrib
    completed = departed_ids & arrived_ids
    unfinished = departed_ids - arrived_ids
    uninserted = scheduled_ids - departed_ids
    completed_losses = [float(tripinfos[ident]["timeLoss"]) for ident in completed
                        if ident in tripinfos and "timeLoss" in tripinfos[ident]]
    if len(completed_losses) != len(completed):
        raise RuntimeError("Missing tripinfo timeLoss for a completed vehicle")
    return {
        "scheduled_count": len(scheduled_ids),
        "inserted_count": len(departed_ids),
        "completed_count": len(completed),
        "unfinished_count": len(unfinished),
        "uninserted_count": len(uninserted),
        "completed_mean_time_loss_seconds": statistics.mean(completed_losses) if completed_losses else None,
        "completed_total_time_loss_seconds": sum(completed_losses),
        "unfinished_ids": sorted(unfinished),
        "uninserted_ids": sorted(uninserted),
    }
