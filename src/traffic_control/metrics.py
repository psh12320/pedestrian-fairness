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

    @property
    def wait(self) -> float | None:
        return None if self.entered_at is None else self.entered_at - self.reached_at


class CrossingTracker:
    """A stopped approach or direct crossing entry is a crossing request.

    SUMO's person waitingTime is used only to detect the first stopped step, never
    as the measured wait. The request timestamp is the beginning of that step.
    The observed entry event is when road ID changes to the crossing edge.
    Time resolution is one simulation step.
    """

    def __init__(self, crossing_ids: set[str], step_seconds: float = 1.0):
        self.crossing_ids = crossing_ids
        self.step_seconds = step_seconds
        self.events: dict[tuple[str, str], CrossingEvent] = {}

    def observe(self, time: float, persons: dict[str, tuple[str, str, float]]) -> None:
        for person_id, (road, next_edge, stopped_seconds) in persons.items():
            if next_edge in self.crossing_ids and stopped_seconds > 0 and road != next_edge:
                key = (person_id, next_edge)
                if key not in self.events:
                    self.events[key] = CrossingEvent(person_id, next_edge,
                                                     max(0.0, time - self.step_seconds))
            if road in self.crossing_ids:
                key = (person_id, road)
                if key not in self.events:
                    self.events[key] = CrossingEvent(person_id, road, time, time)
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
