import json
from pathlib import Path
import tempfile
import unittest

from src.traffic_control.fairness.demand import make_demand, scenarios
from src.traffic_control.fairness.events import EventTracker, PersonSample
from src.traffic_control.fairness.signals import SafeSignal, baseline_action
from src.traffic_control.signals import SignalPlan

CONFIG = json.loads(Path("config/fairness_v2.json").read_text())


class FairnessTests(unittest.TestCase):
    def signal(self):
        return SafeSignal(SignalPlan({"EW": {0}, "NS": {1}, "PED": {2}}, 3, 18, CONFIG,
                                     {(0, 1), (0, 2), (1, 2)}))

    def test_direct_pedestrian_service_preserves_timing(self):
        signal = self.signal()
        self.assertEqual(signal.mask(11.8, (True, True, True)), (True, False, False))
        with self.assertRaises(ValueError):
            signal.request(11.8, 2, (True, True, True))
        signal.request(12, 2, (True, True, True))
        self.assertEqual(signal.stage, "EW_YELLOW")
        signal.advance(14.8, False)
        self.assertEqual(signal.stage, "EW_YELLOW")
        signal.advance(15, False)
        self.assertEqual(signal.stage, "ALL_RED")
        signal.advance(17, False)
        self.assertEqual(signal.stage, "PED")
        self.assertEqual(signal.history, [(0, "EW"), (12, "EW_YELLOW"), (15, "ALL_RED"), (17, "PED")])

    def test_clearance_uses_seconds_not_number_of_samples(self):
        signal = self.signal()
        signal.enter("PED_CLEARANCE", 0)
        for tick in range(1, 101):
            signal.advance(tick / 5, True)
        for tick in range(101, 116):
            signal.advance(tick / 5, False)
            self.assertEqual(signal.stage, "PED_CLEARANCE")
        signal.advance(23.2, False)
        self.assertEqual(signal.stage, "READY")
        signal.request(24, 1, (True, True, False))
        self.assertEqual(signal.state(), "rGr")

    def test_max_green_and_every_baseline_share_masks(self):
        for kind in ("fixed", "actuated", "pressure"):
            signal = self.signal()
            signal.advance(30, False)
            action = baseline_action(kind, signal, 30, [5, 3], [0, 0], 2)
            self.assertNotEqual(action, 0)
            signal.request(30, action, (True, True, True))
            self.assertEqual(signal.state(), "yrr")
        with self.assertRaises(AssertionError):
            self.signal().advance(.2, True)

    def test_timestamp_bounds_entry_exit_and_repeat_visit(self):
        tracker = EventTracker({"cross": ((0, 0), (5, 0))})
        def sample(road, next_edge, x, speed=1):
            return PersonSample(road, next_edge, (x, 0), speed)
        tracker.observe(.2, {"p": sample("approach", "cross", -2)})
        tracker.observe(.4, {"p": sample("approach", "cross", -1)})
        tracker.observe(.6, {"p": sample("cross", "exit", .1)})
        tracker.observe(.8, {"p": sample("exit", "", 6)})
        tracker.observe(1, {"p": sample("approach", "cross", -1)})
        tracker.observe(1.2, {"p": sample("cross", "exit", .1)})
        self.assertEqual(len(tracker.events), 2)
        self.assertEqual(tracker.events[1].visit, 2)
        row = tracker.rows()[0]
        self.assertAlmostEqual(row["wait_seconds"], .2)
        self.assertEqual(row["wait_lower_seconds"], 0)
        self.assertAlmostEqual(row["wait_upper_seconds"], .6)
        self.assertAlmostEqual(row["traversal_seconds"], .2)

    def test_transient_entrance_has_conservative_bound(self):
        trackers = [EventTracker({"cross": ((0, 0), (5, 0))}) for _ in range(2)]
        # Fine observations see a brief entrance visit that coarse sampling misses.
        for tick, x in enumerate((-2, -2, -1, -2, -1, .1), 1):
            s = {"p": PersonSample("cross" if tick == 6 else "approach", "exit" if tick == 6 else "cross", (x, 0), 1)}
            trackers[0].observe(tick / 10, s)
            if tick % 2 == 0:
                trackers[1].observe(tick / 10, s)
        fine, coarse = [t.rows()[0] for t in trackers]
        self.assertIsNone(coarse["wait_seconds"])
        self.assertLessEqual(coarse["wait_lower_seconds"], fine["wait_seconds"])
        self.assertGreaterEqual(coarse["wait_upper_seconds"], fine["wait_seconds"])

    def test_stopped_people_require_a_connected_queue(self):
        tracker = EventTracker({"cross": ((0, 0), (5, 0))})
        samples = {name: PersonSample("approach", "cross", (x, 0), 0)
                   for name, x in (("front", -1), ("queue", -1.8), ("unrelated", -5))}
        tracker.observe(.2, samples)
        self.assertEqual({e.person_id for e in tracker.events}, {"front", "queue"})
        tracker.observe(.4, samples)
        self.assertEqual(tracker.summary()["cross"]["unserved_count"], 2)

    def test_entrance_covers_outer_walking_strip(self):
        tracker = EventTracker({"cross": ((0, 0), (5, 0))})
        tracker.observe(.2, {"outer": PersonSample("approach", "cross", (-1, 1.8), 1)})
        tracker.observe(.4, {"outer": PersonSample("cross", "exit", (.1, 1.8), 1)})
        self.assertAlmostEqual(tracker.rows()[0]["wait_seconds"], .2)

    def test_missing_approach_is_unknown_not_zero(self):
        tracker = EventTracker({"cross": ((0, 0), (5, 0))})
        tracker.observe(.2, {"direct": PersonSample("cross", "exit", (2, 0), 1)})
        row = tracker.rows()[0]
        self.assertIsNone(row["wait_seconds"])
        summary = tracker.summary()["cross"]
        self.assertEqual(summary["unknown_wait_count"], 1)
        self.assertIsNone(summary["p95_observed_wait_seconds"])
        self.assertEqual(summary["p95_wait_lower_seconds"], 0)
        self.assertEqual(summary["p95_wait_upper_seconds"], .2)

    def test_scenario_count_and_reproducible_demand(self):
        self.assertEqual(len(scenarios()), 21)
        with tempfile.TemporaryDirectory() as directory:
            cfg = {**CONFIG, "demand_end_seconds": 120, "slow_fraction": .2}
            a = make_demand(Path(directory) / "a.xml", 21001, cfg, 3)
            b = make_demand(Path(directory) / "b.xml", 21001, cfg, 3)
            c = make_demand(Path(directory) / "c.xml", 21002, cfg, 3)
        self.assertEqual(a, b)
        self.assertNotEqual(a["sha256"], c["sha256"])
        self.assertTrue(a["vehicles"] and a["people"])
        self.assertTrue(any(p["type"] == "slow" for p in a["people"]))


class FairnessSUMOTests(unittest.TestCase):
    def test_three_baselines_and_repeated_live_metrics(self):
        try:
            import traci
            from src.traffic_control.scenario import executable
            executable("sumo")
        except (ImportError, RuntimeError):
            self.skipTest("SUMO and TraCI required")
        from src.traffic_control.fairness.environment import run
        cfg = {"end_seconds": 360, "warmup_seconds": 0, "measurement_end_seconds": 240,
               "demand_end_seconds": 90, "vehicle_rate_per_hour": 720, "pedestrian_rate_per_hour": 720,
               "subscription_crosscheck": True}
        with tempfile.TemporaryDirectory() as directory:
            results = [run(kind, 21001, Path(directory) / str(i), 3, cfg)
                       for i, kind in enumerate(("fixed", "actuated", "pressure", "actuated"))]
        self.assertEqual(results[1], results[3])
        self.assertEqual(len({r["demand_sha256"] for r in results}), 1)
        for result in results:
            self.assertGreater(result["vehicles_end"]["completed_count"], 0)
            self.assertGreater(result["persons"]["completed"], 0)
            self.assertEqual(len(result["pedestrians_by_crossing_end"]), 12)
            self.assertGreater(sum(c["entered_count"] for c in result["pedestrians_by_crossing_end"].values()), 0)
            self.assertTrue(all(any(stage == "PED" for _, stage in history)
                                for history in result["phase_history"].values()))


if __name__ == "__main__":
    unittest.main()
