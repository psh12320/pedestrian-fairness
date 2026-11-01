"""Live interaction tests beyond the light-demand reproducibility smoke."""
import json
from pathlib import Path
import tempfile
import unittest


class StressTests(unittest.TestCase):
    def test_slow_walkers_burst_short_links_and_turns(self):
        try:
            import traci
            from src.traffic_control.scenario import executable
            executable("sumo")
        except (ImportError, RuntimeError):
            self.skipTest("SUMO/TraCI required")
        from src.traffic_control.fairness.environment import run
        cases = [("actuated", {"slow_fraction": 1.}),
                 ("pressure", {"burst_count": 80, "burst_start_seconds": 20, "junction_spacing_m": 65}),
                 ("fixed", {"turn_fraction": .3})]
        common = {"end_seconds": 360, "warmup_seconds": 0, "measurement_end_seconds": 240,
                  "demand_end_seconds": 90, "vehicle_rate_per_hour": 720, "pedestrian_rate_per_hour": 720}
        with tempfile.TemporaryDirectory() as scratch:
            for i, (controller, cfg) in enumerate(cases):
                with self.subTest(controller=controller, cfg=cfg):
                    directory = Path(scratch) / str(i)
                    observed_speeds = []
                    def observe(now, samples, endpoints):
                        observed_speeds.extend(s.speed for s in samples.values())
                    metrics = run(controller, 21031, directory, overrides={**common, **cfg}, sample_sink=observe)
                    self.assertLessEqual(max(observed_speeds), (.8 if cfg.get("slow_fraction") else 1.3) + 1e-5)
                    self.assertGreater(metrics["vehicles_end"]["completed_count"], 0)
                    self.assertGreater(metrics["persons"]["completed"], 0)
                    events = json.loads((directory / "events.json").read_text())
                    self.assertTrue(events)
                    self.assertFalse(any(e["wait_seconds"] is None and e["entered_at"] is not None for e in events))
                    vehicles = metrics["vehicles_end"]
                    self.assertEqual(vehicles["scheduled_count"], vehicles["completed_count"] +
                                     vehicles["unfinished_count"] + vehicles["uninserted_count"])
                    if cfg.get("slow_fraction"):
                        self.assertEqual({e["speed_group"] for e in events}, {"slow"})
