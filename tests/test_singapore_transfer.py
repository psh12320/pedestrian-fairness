"""Protocol and crossing-denominator checks for the NUS-derived transfer."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.traffic_control.ppo import PPOModel
from src.traffic_control.singapore import load_transfer_models
from src.traffic_control.singapore_transfer import (
    aggregate_crossings, load_settings, selected_model_budget)
from src.traffic_control.scenario import ROOT


class SingaporeTransferTests(unittest.TestCase):
    def test_transfer_seeds_are_ten_new_seeds(self) -> None:
        config = ROOT / "config" / "singapore_transfer.json"
        settings, seeds = load_settings(config)
        self.assertEqual(seeds, list(range(9101, 9111)))
        self.assertEqual(load_settings(config, smoke=True)[1], [61])
        with TemporaryDirectory() as temporary:
            bad = Path(temporary) / "bad.json"
            settings["seeds"][0] = 9001  # Main locked seed must stay separate.
            bad.write_text(json.dumps(settings), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(bad)

    def test_full_transfer_requires_complete_replica_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entries = []
            for method in ("ippo", "mappo"):
                for weight in (0, 3, 10):
                    for replica in range(2):
                        objective = "vehicle" if weight == 0 else "pedestrian"
                        model = PPOModel(method, objective, ["C0", "C1", "C2"], {}, 1)
                        model.episodes_trained = 150
                        path = directory / f"{method}-w{weight}-r{replica}.json"
                        model.save(path)
                        entries.append({"method": method, "objective": objective,
                                        "reward_weight": weight, "training_run": replica,
                                        "training_seed": 1101 + replica,
                                        "model_path": str(path)})
            manifest = {"version": 2, "complete_protocol": True,
                        "settings": {"reward_weights": [0, 3, 10],
                                     "training_run_seeds": [1101, 1102]},
                        "models": entries}
            file = directory / "training_manifest.json"
            file.write_text(json.dumps(manifest), encoding="utf-8")
            selected = load_transfer_models(directory, all_replicas=True)
            self.assertEqual(len(selected), 12)
            self.assertEqual(len({item["label"] for item in selected}), 12)
            manifest["models"].pop()
            file.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every trained"):
                load_transfer_models(directory, all_replicas=True)

    def test_crossing_summary_keeps_zero_entries_and_pending_requests(self) -> None:
        rows = [{"controller": "actuated", "training_run": ""},
                {"controller": "ippo-w3", "training_run": 0},
                {"controller": "ippo-w3", "training_run": 1}]
        events = [
            {"controller": "actuated", "training_run": "", "crossing_id": ":C0_c0",
             "wait_seconds": 2},
            {"controller": "ippo-w3", "training_run": 0, "crossing_id": ":C0_c0",
             "wait_seconds": 8},
            {"controller": "ippo-w3", "training_run": 1, "crossing_id": ":C0_c0",
             "wait_seconds": None},
        ]
        result = aggregate_crossings(events, rows, {":C0_c0", ":C0_c1"})
        by_key = {(item["controller"], item["crossing"]): item for item in result}
        self.assertEqual(by_key[("ippo-w3", ":C0_c0")][
            "minimum_entries_per_replica"], 0)
        self.assertIsNone(by_key[("ippo-w3", ":C0_c0")]["mean_p95_wait_seconds"])
        self.assertEqual(by_key[("ippo-w3", ":C0_c0")][
            "mean_pending_requests_per_replica"], 0.5)
        self.assertEqual(by_key[("actuated", ":C0_c1")][
            "mean_entries_per_replica"], 0)

    def test_selected_budget_gate_rejects_150_models_when_300_selected(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_file = directory / "training_manifest.json"
            decision_file = directory / "extension_decision.json"
            decision_file.write_text(json.dumps({"extend_all_to_300": True,
                                                 "selected_training_episodes": 300}),
                                     encoding="utf-8")
            manifest = {"complete_protocol": True, "episodes_requested": 150,
                        "models": [{"episodes_run": 150}]}
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "300-episode budget"):
                selected_model_budget(directory, decision_file)
            manifest["episodes_requested"] = 300
            manifest["models"][0]["episodes_run"] = 300
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(selected_model_budget(directory, decision_file)[0], 300)


if __name__ == "__main__":
    unittest.main()
