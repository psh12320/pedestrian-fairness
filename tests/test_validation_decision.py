"""Check the predeclared extension gate on complete validation records."""

import unittest

from src.traffic_control.validation_decision import check_selected_budget, decide


class ValidationDecisionTests(unittest.TestCase):
    def test_final_artifact_budget_must_match_validation_decision(self):
        manifest = {"complete_protocol": True, "episodes_requested": 150,
                    "models": [{"episodes_run": 150} for _ in range(30)]}
        decision = {"extend_all_to_300": False,
                    "selected_training_episodes": 150}
        self.assertEqual(check_selected_budget(manifest, decision), 150)
        decision = {"extend_all_to_300": True,
                    "selected_training_episodes": 300}
        with self.assertRaisesRegex(ValueError, "validation-selected"):
            check_selected_budget(manifest, decision)
        with self.assertRaisesRegex(ValueError, "malformed"):
            check_selected_budget(manifest, {"extend_all_to_300": True,
                                             "selected_training_episodes": 150})

    def test_one_consistent_improving_configuration_extends_everyone(self):
        labels = [f"{method}-w{weight}" for method in ("ippo", "mappo")
                  for weight in (0, 3, 10)]
        manifest = {
            "complete_protocol": True, "episodes_requested": 150,
            "settings": {"reward_weights": [0, 3, 10],
                         "training_run_seeds": [1101, 1102, 1103, 1104, 1105]},
            "models": [{"controller": label, "training_run": replica}
                       for label in labels for replica in range(5)],
        }
        rows = []
        for label in labels:
            for replica in range(5):
                scores = (100, 98 if label == "mappo-w3" and replica < 4 else 100)
                for episode, score in zip((100, 150), scores):
                    rows.append({"controller": label, "training_run": replica,
                                 "episode": episode, "score": score})
        result = decide(manifest, rows)
        self.assertFalse(result["extend_all_to_300"])
        for row in rows:
            if row["controller"] == "mappo-w3" and row["episode"] == 150:
                row["score"] = 96 if row["training_run"] < 4 else 100
        result = decide(manifest, rows)
        self.assertTrue(result["extend_all_to_300"])
        self.assertEqual(result["selected_training_episodes"], 300)


if __name__ == "__main__":
    unittest.main()
