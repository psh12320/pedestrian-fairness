import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.traffic_control.fairness.demand import make_demand
from src.traffic_control.scenario import ROOT


class DemandBytesTest(unittest.TestCase):
    def test_windows_and_linux_match_frozen_cluster_demand(self):
        cfg = json.loads((ROOT / 'config/fairness_v2.json').read_text())
        (ROOT / 'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as folder:
            path = Path(folder) / 'demand.xml'
            manifest = make_demand(path, 830001, cfg, 3)
            data = path.read_bytes()
        self.assertNotIn(b'\r\n', data)
        self.assertEqual(manifest['sha256'], hashlib.sha256(data).hexdigest())
        # Verified independently against the unchanged Linux cluster baseline
        # vmedium-pmedium-830001 in ppo-repair-results.zip. No archive is required
        # when collaborators run this regression.
        self.assertEqual(manifest['sha256'],
                         '6f6805164f67c4a70071ee6bd490e7cbe55c7689ed7e5ffab0d3b9ca536c4474')


if __name__ == '__main__':
    unittest.main()
