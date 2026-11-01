"""Checks for the new experimental protocol without running a long SUMO study."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.traffic_control.cli import prepare_inputs, run
from src.traffic_control.evaluate import (
    crossing_comparison, hierarchical_interval, primary_criterion,
    scenario_specs,
)
from src.traffic_control.scenario import executable
from src.traffic_control.train import validation_score


ROOT = Path(__file__).resolve().parents[1]


class ExperimentalProtocolTests(unittest.TestCase):
    def test_seed_sets_and_scenario_knobs_are_distinct(self):
        training = json.loads((ROOT / "config" / "training.json").read_text())
        evaluation = json.loads((ROOT / "config" / "evaluation.json").read_text())
        self.assertEqual(len(training["training_run_seeds"]), 5)
        self.assertEqual(training["reward_weights"], [0, 3, 10])
        self.assertEqual(len(evaluation["seeds"]), 10)
        self.assertFalse(set(evaluation["seeds"]) & set(training["validation_seeds"]))
        self.assertFalse(set(evaluation["seeds"]) & set(evaluation["exploratory_seeds"]))
        specs = scenario_specs(evaluation)
        self.assertIn("vmedium-pmedium-slow", specs)
        self.assertIn("pedestrian-burst", specs)
        self.assertIn("vehicle-overload", specs)
        self.assertIn("short-spacing", specs)
        self.assertEqual(specs["pedestrian-burst"][0]["pedestrian_burst_route_index"], 4)

    def test_validation_penalises_unserved_vehicles_and_pedestrians(self):
        settings = {"unfinished_vehicle_penalty_seconds": 300,
                    "unserved_request_penalty_seconds": 60}
        item = {"vehicles": {
            "scheduled_count": 10, "completed_mean_time_loss_seconds": 10,
            "unfinished_count": 0, "uninserted_count": 0},
            "pedestrians_by_crossing": {"c": {
                "entered_count": 10, "mean_wait_seconds": 5, "unserved_count": 0}}}
        self.assertEqual(validation_score([item], 3, settings), 25)
        neglected = json.loads(json.dumps(item))
        neglected["vehicles"]["unfinished_count"] = 2
        neglected["pedestrians_by_crossing"]["c"]["unserved_count"] = 2
        self.assertGreater(validation_score([neglected], 3, settings),
                           validation_score([item], 3, settings))

    def test_hierarchical_interval_uses_matched_seed_differences(self):
        reference = [{"training_run": "", "seed": 5001, "loss": 10},
                     {"training_run": "", "seed": 5002, "loss": 20}]
        candidate = [
            {"training_run": 0, "seed": 5001, "loss": 9},
            {"training_run": 0, "seed": 5002, "loss": 19},
            {"training_run": 1, "seed": 5001, "loss": 8},
            {"training_run": 1, "seed": 5002, "loss": 18},
        ]
        result = hierarchical_interval(candidate, "loss", 200, 42, reference)
        self.assertEqual(result["mean"], -1.5)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["training_runs"], 2)
        self.assertLessEqual(result["ci95_high"], -1)

    def test_crossing_comparison_detects_worse_tail_and_missing_crossing(self):
        events = []
        for seed, baseline, candidate in (
                (5001, [1, 2], [1, 2]), (5002, [3, 4], [5, 6])):
            for label, training_run, waits in (
                    ("actuated", "", baseline), ("ippo-w3", 0, candidate)):
                for wait in waits:
                    events.append({"scenario": "vmedium-pmedium", "controller": label,
                                   "training_run": training_run, "seed": seed,
                                   "crossing_id": ":C0_c0", "wait_seconds": wait})
        rows, maximum = crossing_comparison(
            events, "vmedium-pmedium", "ippo-w3", [5001, 5002], [0],
            200, 42, 4, {":C0_c0", ":C0_c1"})
        by_crossing = {row["crossing"]: row for row in rows}
        self.assertEqual(by_crossing[":C0_c0"]["p95_difference_seconds"], 2)
        self.assertTrue(by_crossing[":C0_c0"]["adequate_entries"])
        self.assertIsNone(by_crossing[":C0_c1"]["p95_difference_seconds"])
        self.assertFalse(maximum["all_crossings_measured"])
        self.assertFalse(maximum["all_crossings_adequate"])

    def test_crossing_comparison_accepts_baseline_without_training_run(self):
        events = [
            {"scenario": "s", "controller": label, "training_run": "",
             "seed": 5001, "crossing_id": "c", "wait_seconds": wait}
            for label, wait in (("actuated", 4), ("fixed", 5))]
        rows, maximum = crossing_comparison(
            events, "s", "fixed", [5001], [-1], 20, 42, 1, {"c"})
        self.assertEqual(rows[0]["p95_difference_seconds"], 1)
        self.assertTrue(maximum["all_crossings_measured"])

    def test_primary_criterion_rejects_shifted_pending_requests(self):
        # Total pending requests do not rise, but one crossing loses service.
        differences = {
            name: {"mean": value, "ci95_high": value}
            for name, value in {
                "vehicle_time_loss": -2, "completed": 0,
                "unfinished": 0, "uninserted": 0, "unserved_requests": 0,
            }.items()
        }
        maximum = {
            "all_crossings_measured": True, "all_crossings_adequate": True,
            "complete_bootstrap_fraction": 1, "mean": -1,
            "ci95_high": -0.5,
        }
        crossings = [
            {"actuated_unserved": 0, "mean_model_unserved": 1},
            {"actuated_unserved": 1, "mean_model_unserved": 0},
        ]
        self.assertEqual(primary_criterion(differences, maximum, crossings),
                         (False, False))
        crossings[0]["mean_model_unserved"] = 0
        self.assertEqual(primary_criterion(differences, maximum, crossings),
                         (True, True))

    def test_prepared_sumo_inputs_are_byte_equivalent_to_rebuilding(self):
        try:
            executable("sumo")
            executable("netconvert")
            import traci  # noqa: F401
        except (RuntimeError, ImportError) as exc:
            self.skipTest(str(exc))
        overrides = {
            "end_seconds": 200, "vehicle_count_per_axis": 3,
            "pedestrian_count": 12, "vehicle_depart_last_second": 15,
            "pedestrian_depart_last_second": 15,
            "include_compatible_turns": False,
        }
        config = ROOT / "config" / "milestone.json"

        def policy(_tls, obs, _global_obs, eligible):
            return int(eligible and obs[2] > 0), 0.0, 0.0

        with tempfile.TemporaryDirectory(prefix="prepared-equivalence-") as scratch:
            root = Path(scratch)
            prepared = prepare_inputs(root / "inputs", 38, config, junctions=3,
                                      config_overrides=overrides)
            for kind in ("fixed", "policy"):
                legacy = root / f"{kind}-rebuilt"
                shared = root / f"{kind}-prepared"
                old_trace, new_trace = {}, {}
                kwargs = ({"policy_callback": policy} if kind == "policy" else {})
                old_metrics = run(kind, 38, legacy, config, junctions=3,
                                  config_overrides=overrides,
                                  trace_sink=old_trace, **kwargs)
                new_metrics = run(kind, 38, shared, config, junctions=3,
                                  config_overrides=overrides,
                                  trace_sink=new_trace,
                                  prepared_inputs=prepared, **kwargs)
                self.assertEqual(old_metrics["demand_sha256"],
                                 new_metrics["demand_sha256"])
                # netconvert writes its invocation time in an XML comment;
                # compare the actual network elements rather than that header.
                self.assertEqual(
                    ET.tostring(ET.parse(legacy / "connected.net.xml").getroot()),
                    ET.tostring(ET.parse(prepared.network_file).getroot()))
                self.assertEqual((legacy / "demand.rou.xml").read_bytes(),
                                 prepared.routes_file.read_bytes())
                self.assertEqual((legacy / "metrics.json").read_bytes(),
                                 (shared / "metrics.json").read_bytes())
                self.assertEqual((legacy / "crossing_events.json").read_bytes(),
                                 (shared / "crossing_events.json").read_bytes())
                self.assertEqual(old_metrics["vehicles"], new_metrics["vehicles"])
                self.assertEqual(old_metrics["pedestrians_by_crossing"],
                                 new_metrics["pedestrians_by_crossing"])
                self.assertEqual(old_metrics["signal_history"],
                                 new_metrics["signal_history"])
                self.assertEqual(old_trace, new_trace)
                self.assertGreater(old_metrics["vehicles"]["scheduled_count"], 0)
                self.assertGreater(old_metrics["persons_seen_at_crossing"], 0)


if __name__ == "__main__":
    unittest.main()
