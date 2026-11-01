import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch dependency missing")
class NeuralInterfaceTests(unittest.TestCase):
    def test_actors_local_masking_and_global_critic(self):
        import torch
        from src.traffic_control.fairness.networks import JunctionNetworks
        torch.set_num_threads(1)
        features = {"C0": ["queue", "age"], "C1": ["queue", "age"]}
        model = JunctionNetworks("mappo", features)
        before = model.act("C0", [1, 0], [True, False, True], True)
        _, costs = model.values("C0", [1, 0], [1, 0, 100, 100])
        after = model.act("C0", [1, 0], [True, False, True], True)
        self.assertEqual(before, after)
        self.assertEqual(tuple(costs.shape), (4,))
        for _ in range(100):
            action, logp = model.act("C0", [1, 0], [False, True, False])
            self.assertEqual(action, 1)
            self.assertAlmostEqual(logp, 0, places=5)
        with self.assertRaises(ValueError):
            model.act("C0", [1, 0], [False, False, False])

    def test_checkpoint_schema_and_roundtrip(self):
        import torch
        from src.traffic_control.fairness.networks import JunctionNetworks
        torch.set_num_threads(1)
        features = {"C0": ["queue", "age"]}
        model = JunctionNetworks("ippo", features)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            loaded = JunctionNetworks.load(path, features)
            self.assertEqual(model.act("C0", [1, 2], [True] * 3, True),
                             loaded.act("C0", [1, 2], [True] * 3, True))
            with self.assertRaisesRegex(ValueError, "feature"):
                JunctionNetworks.load(path, {"C0": ["age", "queue"]})
