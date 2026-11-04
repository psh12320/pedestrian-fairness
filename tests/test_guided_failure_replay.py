"""Do not excuse outcome drift as cross-platform diagnostic roundoff."""
import unittest
from scripts.replay_guided_failures import differences, split_diagnostic_roundoff


class ReplayVerificationTests(unittest.TestCase):
    def test_entropy_roundoff_retained_but_not_material(self):
        d = {'/diagnostics/C0/mean_entropy': [.04519037902355194, .04519038274884224]}
        material, roundoff = split_diagnostic_roundoff(d)
        self.assertEqual(material, {})
        self.assertEqual(roundoff, d)

    def test_even_tiny_traffic_difference_is_material(self):
        d = {'/vehicles/completed_mean_time_loss_seconds': [40.0, 40.000000001]}
        self.assertEqual(split_diagnostic_roundoff(d), (d, {}))

    def test_missing_metric_and_integer_counts_are_material(self):
        d = differences({'x': 4, 'y': 0}, {'x': 4})
        d['/diagnostics/C0/free_decisions'] = [100, 101]
        self.assertEqual(split_diagnostic_roundoff(d), (d, {}))

    def test_large_diagnostic_difference_is_material(self):
        d = {'/diagnostics/C0/mean_entropy': [.04, .05]}
        self.assertEqual(split_diagnostic_roundoff(d), (d, {}))


if __name__ == '__main__':
    unittest.main()
