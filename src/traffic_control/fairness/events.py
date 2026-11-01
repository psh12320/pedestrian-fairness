"""Observed crossing visits with timestamp bounds, never zero-imputed waits.

Queue membership requires a stopped chain connected to an entrance, on the
same immediate approach and towards the same crossing endpoint. It does not
follow from SUMO person waitingTime alone. Entrance membership is not monotone:
lateral movement can leave/re-enter the zone between observations. Arrival lower
bounds therefore use the first observed approach, not just the preceding tick.
These conservative bounds still do not bound geometric-definition error.
"""

from dataclasses import asdict, dataclass
import math
import statistics


@dataclass(frozen=True)
class PersonSample:
    road: str
    next_edge: str
    position: tuple[float, float]
    speed: float
    speed_group: str = "normal"


@dataclass
class Visit:
    person_id: str
    crossing_id: str
    visit: int
    direction: int | None
    speed_group: str
    reached_at: float | None
    reached_lower: float
    entered_at: float | None = None
    entered_lower: float | None = None
    exited_at: float | None = None
    exit_lower: float | None = None
    source: str = "entrance"
    lost: bool = False

    def row(self, now):
        row = asdict(self)
        observed = self.entered_at is not None and self.reached_at is not None
        row.update(wait_seconds=(self.entered_at - self.reached_at if observed else None),
                   wait_lower_seconds=(max(0.0, self.entered_lower - self.reached_at)
                                       if observed else (0.0 if self.entered_at is not None else None)),
                   wait_upper_seconds=(self.entered_at - self.reached_lower
                                       if self.entered_at is not None else None),
                   pending_age_seconds=(now - self.reached_at
                                        if self.entered_at is None and self.reached_at is not None else None),
                   traversal_seconds=(self.exited_at - self.entered_at
                                      if self.exited_at is not None else None))
        return row


class EventTracker:
    def __init__(self, endpoints, entrance_m=1.2, queue_gap_m=1.0, crossing_half_width_m=2.0):
        self.endpoints = endpoints
        self.entrance_m, self.queue_gap_m = entrance_m, queue_gap_m
        self.crossing_half_width_m = crossing_half_width_m
        self.events = []
        self.active = {}
        self.visits = {}
        self.previous = {}
        self.approach_started = {}
        self.now = 0.0

    def _new(self, ident, crossing, sample, reached, lower, source, direction):
        key = (ident, crossing)
        self.visits[key] = self.visits.get(key, 0) + 1
        event = Visit(ident, crossing, self.visits[key], direction, sample.speed_group,
                      reached, lower, source=source)
        self.events.append(event)
        self.active[ident] = event
        return event

    def observe(self, now, samples):
        if now <= self.now:
            raise ValueError("Observations must have increasing timestamps")
        # Connected queues are discovered from entrance anchors, never from
        # an arbitrary stopped person far back on a sidewalk.
        candidates, reached = {}, {}
        for ident, sample in samples.items():
            if sample.next_edge in self.endpoints and sample.road not in self.endpoints:
                key = (ident, sample.next_edge)
                self.approach_started.setdefault(key, self.previous.get(ident, (0., None))[0])
                ends = self.endpoints[sample.next_edge]
                direction = min(range(2), key=lambda i: math.dist(sample.position, ends[i]))
                group = (sample.next_edge, sample.road, direction)
                candidates.setdefault(group, []).append((ident, sample))
                # A crossing entrance spans its full width, not a disk around
                # the lane centre. A disk misses legitimate outer walking strips.
                origin, opposite = ends[direction], ends[1 - direction]
                length = math.dist(origin, opposite)
                dx, dy = ((opposite[i] - origin[i]) / length for i in range(2))
                px, py = (sample.position[i] - origin[i] for i in range(2))
                along, lateral = px * dx + py * dy, abs(px * dy - py * dx)
                if abs(along) <= self.entrance_m and lateral <= self.crossing_half_width_m + .05:
                    reached[ident] = (sample.next_edge, "entrance", direction)
        for (crossing, _, direction), people in candidates.items():
            chain = [s.position for p, s in people if p in reached and s.speed <= 0.1]
            remaining = [(p, s) for p, s in people if p not in reached and s.speed <= 0.1]
            while remaining:
                joined = [(p, s) for p, s in remaining
                          if any(math.dist(s.position, anchor) <= self.queue_gap_m for anchor in chain)]
                if not joined:
                    break
                for p, s in joined:
                    reached[p] = (crossing, "connected_queue", direction)
                    chain.append(s.position)
                joined_ids = {p for p, _ in joined}
                remaining = [(p, s) for p, s in remaining if p not in joined_ids]
        for ident, sample in samples.items():
            prior = self.previous.get(ident)
            lower = prior[0] if prior else 0.0
            event = self.active.get(ident)
            if event and event.entered_at is not None and sample.road != event.crossing_id:
                event.exited_at, event.exit_lower = now, lower
                self.approach_started.pop((ident, event.crossing_id), None)
                del self.active[ident]
                event = None
            if event and event.entered_at is None and sample.road not in self.endpoints:
                if sample.next_edge != event.crossing_id:
                    event.lost = True  # diverted/disappeared requests remain in accounting
                    del self.active[ident]
                    event = None
            if ident in reached and event is None:
                crossing, source, direction = reached[ident]
                event = self._new(ident, crossing, sample, now,
                                  self.approach_started[(ident, crossing)], source, direction)
            if sample.road in self.endpoints:
                if event is not None and event.crossing_id != sample.road:
                    event.lost = True
                    event = None
                if event is None:
                    # A missed approach has unknown point wait; use the full
                    # simulation start as its conservative arrival lower bound.
                    event = self._new(ident, sample.road, sample, None, 0.0,
                                      "entry_without_approach", None)
                if event.entered_at is None:
                    event.entered_at, event.entered_lower = now, lower
        for ident in set(self.active) - set(samples):
            self.active[ident].lost = True
            del self.active[ident]
        self.previous = {p: (now, sample) for p, sample in samples.items()}
        self.now = now

    def pending(self, crossing):
        return [e for e in self.active.values()
                if e.crossing_id == crossing and e.entered_at is None]

    def rows(self):
        return [event.row(self.now) for event in self.events]

    def summary(self):
        result = {}
        for crossing in sorted(self.endpoints):
            rows = [r for r in self.rows() if r["crossing_id"] == crossing]
            entered = [r for r in rows if r["entered_at"] is not None]
            waits = sorted(r["wait_seconds"] for r in entered if r["wait_seconds"] is not None)
            def p95(values):
                return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None
            result[crossing] = {
                "request_count": len(rows), "entered_count": len(entered),
                "observed_wait_count": len(waits),
                "unknown_wait_count": len(entered) - len(waits),
                "mean_observed_wait_seconds": statistics.mean(waits) if waits else None,
                "p95_observed_wait_seconds": p95(waits),
                "p95_wait_lower_seconds": p95([r["wait_lower_seconds"] for r in entered]),
                "p95_wait_upper_seconds": p95([r["wait_upper_seconds"] for r in entered]),
                "unserved_count": sum(r["entered_at"] is None for r in rows),
                "lost_count": sum(r["lost"] for r in rows),
                "still_crossing_count": sum(r["entered_at"] is not None and r["exited_at"] is None for r in rows),
            }
        return result
