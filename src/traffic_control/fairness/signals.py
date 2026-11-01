"""Time-based, common safety machine with explicit service-phase requests."""

from ..signals import SignalPlan

PHASES = ("EW", "NS", "PED")


class SafeSignal:
    def __init__(self, plan: SignalPlan):
        self.plan = plan
        self.stage = "EW"
        self.since = 0.0
        self.target = None
        self.empty_since = None
        self.last_time = 0.0
        self.history = [(0.0, "EW")]
        self.decisions = []

    def enter(self, stage, now):
        self.stage, self.since = stage, now
        self.empty_since = None
        self.history.append((now, stage))

    def age(self, now):
        return now - self.since

    def advance(self, now, occupied):
        if now < self.last_time:
            raise ValueError("Signal time moved backwards")
        self.last_time = now
        cfg, age = self.plan.config, self.age(now)
        if self.stage in ("EW", "NS") and occupied:
            raise AssertionError("Pedestrian crossing occupied during vehicle green")
        if self.stage.endswith("_YELLOW") and age + 1e-8 >= cfg["yellow_seconds"]:
            self.enter("ALL_RED", now)
        elif self.stage == "ALL_RED" and age + 1e-8 >= cfg["all_red_seconds"]:
            if self.target == "PED" or not occupied:
                self.enter(self.target, now)
                self.target = None
        elif self.stage == "PED" and age + 1e-8 >= cfg["pedestrian_green_seconds"]:
            self.enter("PED_CLEARANCE", now)
        elif self.stage == "PED_CLEARANCE":
            if occupied:
                self.empty_since = None
            elif self.empty_since is None:
                self.empty_since = now
            if (age + 1e-8 >= self.plan.clearance and self.empty_since is not None
                    and now - self.empty_since + 1e-8 >= cfg["pedestrian_clearance_empty_seconds"]):
                self.enter("READY", now)

    def mask(self, now, demand):
        if self.stage == "READY":
            # A new pedestrian admission immediately after clearance is unnecessary
            # unless pedestrians remain and there is no waiting vehicle movement.
            return (True, True, bool(demand[2] and not any(demand[:2])))
        if self.stage not in ("EW", "NS"):
            return (False, False, False)
        current = PHASES.index(self.stage)
        if self.age(now) + 1e-8 < self.plan.config["vehicle_green_min_seconds"]:
            return tuple(i == current for i in range(3))
        mask = [True, True, True]
        if (self.age(now) + 1e-8 >= self.plan.config["vehicle_green_actuated_max_seconds"]
                and any(demand[i] for i in range(3) if i != current)):
            mask[current] = False
        return tuple(mask)

    def request(self, now, action, demand, occupied=False):
        mask = self.mask(now, demand)
        if not isinstance(action, int) or action not in range(3) or not mask[action]:
            raise ValueError(f"Illegal requested phase {action}, mask={mask}")
        selected = PHASES[action]
        before = self.stage
        if before == "READY":
            if occupied and selected != "PED":
                raise AssertionError("Cannot release vehicles onto occupied crossings")
            self.enter(selected, now)
        elif selected != before:
            self.target = selected
            self.enter(before + "_YELLOW", now)
        self.decisions.append({"time": now, "before": before, "requested": selected,
                               "executed_stage": self.stage, "mask": list(mask)})

    def state(self):
        return self.plan.state("ALL_RED" if self.stage == "READY" else self.stage)


def baseline_action(kind, signal, now, queues, downstream, pedestrian_count):
    """Every baseline uses the same available service phases and transitions."""
    demand = (queues[0] > 0, queues[1] > 0, pedestrian_count > 0)
    mask = signal.mask(now, demand)
    available = [i for i, legal in enumerate(mask) if legal]
    if not available:
        return None
    if kind == "fixed":
        if signal.stage == "READY":
            desired = 0
        elif signal.age(now) < signal.plan.config["vehicle_green_fixed_seconds"]:
            desired = PHASES.index(signal.stage)
        else:
            desired = 1 if signal.stage == "EW" else 2
    elif kind == "actuated":
        if demand[2] and mask[2]:
            desired = 2
        elif signal.stage in ("EW", "NS"):
            current = PHASES.index(signal.stage)
            desired = current if queues[current] > 0 else 1 - current
        else:
            desired = max((0, 1), key=lambda i: (queues[i], -i))
    elif kind == "pressure":
        scores = [queues[0] - downstream[0], queues[1] - downstream[1],
                  2.0 * pedestrian_count]
        desired = max(available, key=lambda i: (scores[i], -i))
    else:
        raise ValueError(kind)
    return desired if desired in available else available[0]
