"""Reporting regressions for partial fairness, censoring and paired uncertainty."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.pedestrian_finetune.reporting import (
    aggregate_scenario, detailed_comparison, write_report,
)


CROSSINGS = [f':C{i // 4}_c{i % 4}' for i in range(12)]


def result(loss=10, tail=20, seed=0):
    return {
        'demand_sha256': f'demand-{seed}',
        'vehicles': {'scheduled_count': 10, 'completed_count': 10,
                     'unfinished_count': 0, 'uninserted_count': 0,
                     'completed_mean_time_loss_seconds': loss,
                     'completed_mean_depart_delay_seconds': 2,
                     'completed_mean_time_loss_plus_depart_delay_seconds': loss + 2,
                     'completed_depart_delay_observed_count': 10},
        'persons': {'scheduled': 10, 'completed': 10, 'unfinished': 0, 'uninserted': 0},
        'crossings': {c: {'waits': [tail] * 100, 'p95': tail, 'entries': 100,
                         'pending_at_cutoff': 0, 'unresolved_after_followup': 0,
                         'unknown_cohort_entries': 0, 'lost': 0,
                         'speed_groups': {'normal': {'waits': [tail] * 100, 'pending': 0}}}
                      for c in CROSSINGS},
    }


def matrix(loss=10, tail=20):
    return [[result(loss, tail, s) for s in range(2)] for _ in range(3)]


def records(loss=10, tail=20):
    return [[{'result': row, 'network_sha256': 'network',
              'vehicles_end': deepcopy(row['vehicles']),
              'crossing_completion': {c: {'entered_without_observed_exit': 0, 'completed_visits': 100}
                                      for c in CROSSINGS}}
             for row in replica] for replica in matrix(loss, tail)]


class PedestrianFineTuneReportingTests(unittest.TestCase):
    def test_six_crossings_with_gain_are_partial_not_original_success(self):
        left, right = matrix(8, 15), matrix()
        for replica in left:
            for row in replica:
                for c in CROSSINGS[6:]:
                    row['crossings'][c].update(waits=[25] * 100, p95=25)
        report = detailed_comparison(left, right, draws=40)
        self.assertEqual(report['point']['crossings_no_worse'], 6)
        self.assertEqual(len(report['crossings']), 12)
        self.assertEqual(report['point']['worst_crossing_delta_seconds'], 5)
        self.assertFalse(report['descriptive_pass'])
        self.assertFalse(report['original_target_descriptively_met'])
        self.assertTrue(report['partial_evidence_with_no_strict_service_deterioration'])
        self.assertTrue(report['exploratory_partial_evidence'])
        self.assertEqual(report['pointwise_interval_supported_crossings'], CROSSINGS[:6])
        self.assertIn('6/12', report['evidence_label'])
        self.assertIn('original target unmet', report['evidence_label'])
        self.assertEqual(report['exploratory_intervals']['vehicle_delta_seconds'], [-2, -2])
        self.assertEqual(report['crossings'][CROSSINGS[0]]['pointwise_exploratory_95_interval_seconds'], [-5, -5])
        self.assertFalse(report['confirmatory_statistical_claim'])

    def test_cutoff_only_service_failure_cannot_be_hidden_by_good_waits(self):
        left, right = matrix(8, 15), matrix()
        left[0][0]['crossings'][CROSSINGS[0]]['pending_at_cutoff'] = 1
        report = detailed_comparison(left, right, draws=0)
        self.assertEqual(report['point']['crossings_no_worse'], 12)
        self.assertEqual(report['service_failure_pairs'], 1)
        self.assertEqual(report['cutoff_only_failure_pairs'], 1)
        self.assertFalse(report['descriptive_pass'])
        self.assertFalse(report['partial_evidence_with_no_strict_service_deterioration'])
        self.assertIn('strict service deterioration', report['evidence_label'])

    def test_no_entries_remain_missing_not_zero_wait_success(self):
        left, right = matrix(8, 15), matrix()
        for replica in left:
            for row in replica:
                row['crossings'][CROSSINGS[0]].update(waits=[], p95=None, entries=0,
                                                    pending_at_cutoff=10, unresolved_after_followup=10)
        report = detailed_comparison(left, right, draws=20)
        self.assertIsNone(report['crossings'][CROSSINGS[0]]['p95_delta_seconds'])
        self.assertIsNone(report['crossings'][CROSSINGS[0]]['pointwise_exploratory_95_interval_seconds'])
        self.assertEqual(report['point']['crossings_no_worse'], 11)
        self.assertFalse(report['descriptive_pass'])
        self.assertTrue(report['sparse'])
        self.assertIn('inconclusive', report['evidence_label'])

    def test_zero_entry_run_is_visible_even_with_enough_pooled_entries(self):
        left, right = matrix(8, 15), matrix()
        left[0][0]['crossings'][CROSSINGS[0]].update(waits=[], p95=None, entries=0)
        report = detailed_comparison(left, right, draws=0)
        crossing = report['crossings'][CROSSINGS[0]]
        self.assertFalse(crossing['sparse'])
        self.assertEqual(crossing['model_entries_per_replica'], [100, 200, 200])
        self.assertEqual(crossing['model_zero_entry_runs_replica_seed_index'], [[0, 0]])

    def test_reference_original_is_not_original_proposal_criterion(self):
        report = detailed_comparison(matrix(8, 15), matrix(), draws=0,
                                     reference_name='mappo-original')
        self.assertTrue(report['descriptive_pass'])
        self.assertFalse(report['original_target_descriptively_met'])
        self.assertIn('original target is assessed against actuated only', report['evidence_label'])

    def test_pair_bootstrap_preserves_matched_demand_cancellation(self):
        left, right = matrix(8, 15), matrix()
        for r in range(3):
            for s in range(2):
                for m in (left, right):
                    m[r][s]['vehicles']['completed_mean_time_loss_seconds'] += 1000 * s + 100 * r
        a = detailed_comparison(left, right, draws=40)
        b = detailed_comparison(left, right, draws=40)
        self.assertEqual(a['exploratory_intervals']['vehicle_delta_seconds'], [-2, -2])
        self.assertEqual(a, b)

    def test_aggregate_preserves_service_insertion_and_completion_diagnostics(self):
        baseline, original, fine = records(), records(9), records(8, 15)
        fine[0][0]['crossing_completion'][CROSSINGS[0]]['entered_without_observed_exit'] = 6
        report = aggregate_scenario({'actuated': baseline[0], 'fixed': baseline[0]},
                                    {'mappo-original': original, 'mappo-tail': fine}, draws=10,
                                    contrasts=[('mappo-tail', 'mappo-original')])
        row = report['learned']['mappo-tail']
        self.assertEqual(row['absolute']['vehicle_mean_insertion_delay'], 2)
        self.assertEqual(row['absolute']['persons']['completed'], 10)
        self.assertEqual(row['vehicles_end']['completed_count'], 10)
        self.assertEqual(row['crossing_completion']['network_means']['entered_without_observed_exit'], 1)
        self.assertIn('including warmup', row['crossing_completion']['scope'])
        self.assertEqual(row['comparisons']['mappo-original']['point']['vehicle_delta_seconds'], -1)
        self.assertEqual(row['comparisons']['actuated']['point']['vehicle_delta_seconds'], -2)
        self.assertEqual(len(report['paired_contrasts']), 1)

    def test_missing_diagnostics_stay_unavailable(self):
        fine, baseline = records(8, 15), records()
        fine[0][0].pop('crossing_completion')
        fine[0][0].pop('vehicles_end')
        report = aggregate_scenario({'actuated': baseline[0]}, {'mappo-tail': fine}, draws=0)
        self.assertFalse(report['learned']['mappo-tail']['crossing_completion']['available'])
        self.assertFalse(report['learned']['mappo-tail']['vehicles_end']['available'])

    def test_rejects_unmatched_demand_geometry_or_replica_coverage(self):
        baseline = records()
        for alteration, text in [('network', 'network'), ('demand', 'demand'), ('replica', 'replica')]:
            with self.subTest(alteration=alteration):
                fine, original = records(), records()
                if alteration == 'network':
                    fine[0][0]['network_sha256'] = 'other'
                elif alteration == 'demand':
                    fine[0][0]['result']['demand_sha256'] = 'other'
                else:
                    fine.pop()
                with self.assertRaisesRegex(ValueError, text):
                    aggregate_scenario({'actuated': baseline[0]},
                                       {'mappo-original': original, 'mappo-tail': fine}, draws=0)

    def test_final_report_requires_complete_coverage_and_lists_both_references(self):
        protocol = {'fingerprint': 'test', 'evaluation_scenarios': ['medium']}
        data = aggregate_scenario({'actuated': records()[0]},
                                  {'mappo-original': records(), 'mappo-tail': records(8, 15)}, draws=0)
        workspace_runs = Path(__file__).resolve().parents[1] / 'runs'
        workspace_runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='finetune-report-test-', dir=workspace_runs) as directory:
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                write_report(directory, protocol, {'medium': data}, {'complete': False})
            with self.assertRaisesRegex(ValueError, 'Missing planned'):
                write_report(directory, protocol, {}, {'complete': True})
            write_report(directory, protocol, {'medium': data}, {'complete': True})
            content = (Path(directory) / 'REPORT.md').read_text(encoding='utf-8')
            summary = json.loads((Path(directory) / 'summary.json').read_text(encoding='utf-8'))
            self.assertIn('all crossings versus actuated', content)
            self.assertIn('all crossings versus mappo-original', content)
            self.assertIn('6/12', content)
            self.assertIn('Persons unfinished', content)
            self.assertIn('Entered crossing without observed exit', content)
            self.assertIn('pedestrian reward weight 3', content)
            self.assertEqual(summary['fingerprint'], 'test')
            protocol['quick'] = True
            write_report(directory, protocol, {'medium': data}, {'complete': True})
            self.assertIn('Quick smoke test only', (Path(directory) / 'REPORT.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
