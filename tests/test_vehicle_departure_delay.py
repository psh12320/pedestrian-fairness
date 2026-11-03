import tempfile
from pathlib import Path
import unittest

from src.traffic_control.metrics import vehicle_summary
from src.traffic_control.scenario import ROOT


class DepartureDelayTest(unittest.TestCase):
    def result(self, xml):
        (ROOT / 'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as folder:
            path = Path(folder) / 'tripinfo.xml'
            path.write_text(xml)
            return vehicle_summary(path, {'a', 'b', 'c'}, {'a', 'b'}, {'a', 'b'})

    def test_outside_network_delay_is_separate_and_added(self):
        r = self.result('<tripinfos><tripinfo id="a" timeLoss="5" departDelay="25"/>'
                        '<tripinfo id="b" timeLoss="15" departDelay="5"/></tripinfos>')
        self.assertEqual(r['completed_mean_time_loss_seconds'], 10)
        self.assertEqual(r['completed_mean_depart_delay_seconds'], 15)
        self.assertEqual(r['completed_mean_time_loss_plus_depart_delay_seconds'], 25)
        self.assertEqual(r['completed_total_depart_delay_seconds'], 30)
        self.assertEqual(r['completed_depart_delay_observed_count'], 2)
        self.assertEqual(r['uninserted_count'], 1)

    def test_missing_departure_delay_is_not_zero_imputed(self):
        r = self.result('<tripinfos><tripinfo id="a" timeLoss="5" departDelay="25"/>'
                        '<tripinfo id="b" timeLoss="15"/></tripinfos>')
        self.assertIsNone(r['completed_mean_depart_delay_seconds'])
        self.assertIsNone(r['completed_mean_time_loss_plus_depart_delay_seconds'])
        self.assertEqual(r['completed_depart_delay_observed_count'], 1)
        self.assertEqual(r['completed_mean_time_loss_seconds'], 10)


if __name__ == '__main__':
    unittest.main()
