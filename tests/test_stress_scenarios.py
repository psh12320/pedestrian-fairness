"""Live checks that the stress design exercises real traffic and preserves pairing."""

import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.traffic_control.cli import run
from src.traffic_control.scenario import executable, make_demand


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "milestone.json").read_text())
try:
    SUMO_READY = bool(executable("sumo") and executable("netconvert")
                      and importlib.util.find_spec("traci"))
except RuntimeError:
    SUMO_READY = False


class StressScenarioTests(unittest.TestCase):
    def test_turns_and_burst_have_deterministic_scheduled_demand(self):
        burst_count = json.loads((ROOT / "config" / "evaluation.json").read_text())[
            "extra_scenarios"]["pedestrian-burst"]["pedestrian_burst_count"]
        config = {**CONFIG, "end_seconds": CONFIG["end_seconds_three"],
                  "turning_vehicle_count_per_route": 5,
                  "pedestrian_burst_count": burst_count,
                  "pedestrian_burst_route_index": 4}
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.rou.xml"
            second = Path(folder) / "second.rou.xml"
            a = make_demand(first, 91, config, junctions=3)
            b = make_demand(second, 91, config, junctions=3)
            root = ET.parse(first).getroot()
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertEqual(len([v for v in root.findall("vehicle")
                              if v.get("id", "").startswith("vt")]), 60)
        self.assertEqual(len([p for p in root.findall("person")
                              if p.get("id", "").startswith("pb")]), burst_count)

    @unittest.skipUnless(SUMO_READY, "SUMO and TraCI are needed for overload check")
    def test_overload_leaves_traffic_unserved_at_fixed_horizon(self):
        overrides = {
            "vehicle_count_per_axis": 100, "pedestrian_count": 120,
            "turning_vehicle_count_per_route": 5, "junction_spacing_m": 70,
            "vehicle_depart_last_second": 360,
            "pedestrian_depart_last_second": 360,
        }
        with tempfile.TemporaryDirectory() as folder:
            metrics = run("fixed", 81, Path(folder), ROOT / "config" / "milestone.json",
                          junctions=3, config_overrides=overrides)
        vehicle = metrics["vehicles"]
        self.assertEqual(vehicle["scheduled_count"], 860)
        self.assertGreaterEqual(vehicle["unfinished_count"] +
                                vehicle["uninserted_count"],
                                0.05 * vehicle["scheduled_count"])
        self.assertEqual(vehicle["scheduled_count"],
                         vehicle["completed_count"] + vehicle["unfinished_count"]
                         + vehicle["uninserted_count"])
        self.assertGreater(sum(c["entered_count"] for c in
                               metrics["pedestrians_by_crossing"].values()), 0)


if __name__ == "__main__":
    unittest.main()
