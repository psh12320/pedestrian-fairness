import copy
import unittest
from src.traffic_control.fairness.analysis import summarise, selection_score, primary_inference


def vehicle():
    return {"scheduled_count": 10, "completed_count": 9, "unfinished_count": 1,
            "uninserted_count": 0, "completed_mean_time_loss_seconds": 20}


def result():
    return {"vehicles": vehicle(), "crossings": {"c": {"waits": [10, 20], "entries": 2,
        "p95": 20, "pending_at_cutoff": 0, "unknown_cohort_entries": 0,
        "lost": 0, "unresolved_after_followup": 0}}}


class AnalysisTests(unittest.TestCase):
    def inference_fixture(self, crossing_difference):
        baseline, candidate = result(), result()
        baseline["crossings"]["c"].update(waits=[20] * 30, entries=30)
        candidate["crossings"]["c"].update(waits=[20 + crossing_difference] * 30,
                                            entries=30, p95=20 + crossing_difference)
        candidate["vehicles"]["completed_mean_time_loss_seconds"] = 10
        return [{"scenario": "vmedium-pmedium", "seed": seed, "records": [
            {"controller": "actuated", "replica": None, "result": copy.deepcopy(baseline)},
            *[{"controller": "mappo-constrained", "replica": r, "result": copy.deepcopy(candidate)}
              for r in range(5)]]} for seed in range(20)]

    def test_joint_bootstrap_passes_degenerate_known_improvement(self):
        value = primary_inference(self.inference_fixture(-5), bootstrap=30)["mappo-constrained"]
        self.assertTrue(value["statistical_support"])
        self.assertEqual(value["vehicle_upper_97_5"], -10)
        self.assertEqual(value["simultaneous_crossing_upper_97_5"], -5)

    def test_joint_bootstrap_rejects_worse_crossing_despite_vehicle_gain(self):
        value = primary_inference(self.inference_fixture(1), bootstrap=30)["mappo-constrained"]
        self.assertFalse(value["descriptive_pass"])
        self.assertFalse(value["statistical_support"])
        self.assertEqual(value["simultaneous_crossing_upper_97_5"], 1)

    def test_service_cannot_be_improved_by_hiding_uninserted_traffic(self):
        baseline, candidate = result(), result()
        candidate["vehicles"].update(uninserted_count=1, unfinished_count=0,
                                     completed_mean_time_loss_seconds=1)
        self.assertGreater(selection_score(candidate, baseline)[0], 0)

    def test_missing_baseline_measurements_block_claim(self):
        baseline = result()
        baseline["crossings"]["c"]["unknown_cohort_entries"] = 1
        self.assertGreater(selection_score(result(), baseline)[0], 0)

    def test_sampling_is_not_statistical_support(self):
        baseline, candidate = result(), result()
        candidate["vehicles"]["completed_mean_time_loss_seconds"] = 10
        groups = [{"scenario": "vmedium-pmedium", "seed": 1, "records": [
            {"controller": "actuated", "replica": None, "result": baseline},
            {"controller": "mappo-constrained", "replica": 0, "result": candidate}]}]
        inference = primary_inference(groups)["mappo-constrained"]
        self.assertTrue(inference["descriptive_pass"])
        self.assertFalse(inference["statistical_support"])
        self.assertEqual(inference["minimum_crossing_entries"], 2)

    def test_cohort_includes_followup_entry_and_retains_pending(self):
        metrics = {"config": {"warmup_seconds": 10, "measurement_end_seconds": 20},
            "pedestrians_by_crossing_end": {"c": {}}, "vehicles_at_measurement_cutoff": vehicle(),
            "persons": {}, "demand_sha256": "x"}
        event = {"crossing_id": "c", "reached_at": 12, "entered_at": 22, "wait_seconds": 10,
            "wait_lower_seconds": 9.8, "wait_upper_seconds": 10.2, "lost": False, "speed_group": "slow"}
        pending = {**event, "entered_at": None, "wait_seconds": None}
        warmup = {**event, "reached_at": 9}
        summary = summarise(metrics, [event, pending, warmup])["crossings"]["c"]
        self.assertEqual(summary["waits"], [10])
        self.assertEqual(summary["pending_at_cutoff"], 2)
        self.assertEqual(summary["unresolved_after_followup"], 1)
        self.assertEqual(summary["speed_groups"]["slow"]["waits"], [10])
