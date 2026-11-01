import unittest
from src.traffic_control.fairness.learning import advantages


class ReturnsTests(unittest.TestCase):
    def test_finite_horizon_hand_return(self):
        adv, target = advantages([1, 2, 3], [0, 0, 0], 1, 1)
        self.assertEqual(adv, [6, 5, 3])
        self.assertEqual(target, [6, 5, 3])

    def test_value_baseline_and_discount(self):
        adv, targets = advantages([1, 2], [4, 5], .5, 1)
        self.assertEqual(targets, [2, 2])
        self.assertEqual(adv, [-2, -3])

    def test_masked_rollout_update_is_finite(self):
        import torch
        from src.traffic_control.fairness.networks import JunctionNetworks
        from src.traffic_control.fairness.learning import Learner
        torch.set_num_threads(1)
        model = JunctionNetworks("ippo", {"C0": ["queue", "phase"]})
        learner = Learner(model, thresholds={f":C0_c{i}": 20 for i in range(4)})
        trace = [{"junction": "C0", "time": i + 1, "observation": [1, 0], "global_state": [1, 0],
                  "mask": [False, False, False] if i % 2 else [True, True, True],
                  "action": None if i % 2 else 0, "vehicle_queue": 1, "uninserted_due": 0,
                  "pending_count": 1} for i in range(10)]
        before = [p.detach().clone() for p in model.parameters()]
        learner.update(trace, [{"crossing_id": ":C0_c0", "wait_seconds": None, "entered_at": None}])
        self.assertGreater(learner.multipliers[":C0_c0"], 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
