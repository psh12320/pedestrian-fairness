import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import tempfile
import unittest

from src.traffic_control.audit_trial.reporting import (
    absolute_metrics, comparison, end_vehicle_metrics, labels, paired_estimator, report, validate_result,
)
from src.traffic_control.fairness.analysis import quantile
from src.traffic_control.scenario import ROOT


def result(loss=10, tail=20, insertion=2):
    return {'demand_sha256': 'same', 'vehicles': {
        'scheduled_count': 10, 'completed_count': 10, 'unfinished_count': 0, 'uninserted_count': 0,
        'completed_mean_time_loss_seconds': loss, 'completed_mean_depart_delay_seconds': insertion,
        'completed_mean_time_loss_plus_depart_delay_seconds': loss + insertion,
        'completed_depart_delay_observed_count': 10},
        'persons': {'scheduled': 10, 'completed': 10, 'unfinished': 0, 'uninserted': 0},
        'crossings': {f':C{i//4}_c{i%4}': {'waits': [tail] * 100, 'p95': tail, 'entries': 100,
            'pending_at_cutoff': 0, 'unresolved_after_followup': 0, 'unknown_cohort_entries': 0, 'lost': 0,
            'speed_groups': {'normal': {'waits': [tail] * 100, 'pending': 0}}} for i in range(12)}}


class AuditTrialComparisonTests(unittest.TestCase):
    def test_two_replica_paired_intervals_and_insertion_delay_visible(self):
        references = [[result() for _ in range(4)] for _ in range(2)]
        models = [[result(loss=8, tail=18, insertion=12) for _ in range(4)] for _ in range(2)]
        c = comparison(models, references, draws=40)
        self.assertEqual(c['exploratory_intervals']['vehicle_delta_seconds'], [-2, -2])
        self.assertEqual(c['exploratory_intervals']['worst_crossing_delta_seconds'], [-2, -2])
        self.assertEqual(c['point']['time_loss_plus_insertion_delay_delta_seconds'], 8)
        self.assertTrue(c['descriptive_pass'])  # Original timeLoss criterion; added delay is separately transparent.
        self.assertFalse(c['confirmatory_statistical_claim'])
        self.assertEqual(c['bootstrap_valid_draws'], 40)

    def test_cutoff_only_failure_is_visible_without_changing_gate(self):
        a, b = result(loss=8), result()
        a['crossings'][':C0_c0']['pending_at_cutoff'] = 1
        c = comparison([[a]], [[b]], draws=0)
        self.assertEqual(c['service_failure_pairs'], 1)
        self.assertEqual(c['cutoff_only_failure_pairs'], 1)
        self.assertFalse(c['descriptive_pass'])
        a['vehicles'].update(completed_count=9, unfinished_count=1)
        c = comparison([[a]], [[b]], draws=0)
        self.assertEqual(c['cutoff_only_failure_pairs'], 0)

    def test_weighted_tails_match_explicit_paired_resampling(self):
        rng = random.Random(52)
        groups = [[[result(loss=7 + r + s + k) for s in range(4)] for r in range(2)] for k in range(2)]
        for group in groups:
            for replica in group:
                for row in replica:
                    for crossing in row['crossings'].values():
                        waits = [rng.randrange(80) for _ in range(rng.randrange(1, 12))]
                        crossing.update(waits=waits, entries=len(waits), p95=quantile(waits))
        estimate = paired_estimator(*groups)
        for _ in range(20):
            replicas, seeds = rng.choices(range(2), k=2), rng.choices(range(4), k=4)
            actual = estimate(replicas, seeds)
            for crossing in groups[0][0][0]['crossings']:
                differences = [quantile([w for s in seeds for w in groups[0][r][s]['crossings'][crossing]['waits']])
                    - quantile([w for s in seeds for w in groups[1][r][s]['crossings'][crossing]['waits']])
                    for r in replicas]
                self.assertEqual(actual['crossing_p95_delta_seconds'][crossing], statistics.mean(differences))

    def test_pool_events_within_replica_not_average_run_tails(self):
        a, b = result(tail=0), result(tail=100)
        for c in b['crossings'].values():
            c.update(waits=[100], entries=1)
        out = absolute_metrics([[a, b]])
        self.assertEqual(out['crossings'][':C0_c0']['mean_replica_pooled_p95'], 0)
        self.assertEqual(out['crossings'][':C0_c0']['entries_per_replica'], [101])

    def test_missing_measurements_never_become_a_win(self):
        a, b = result(loss=8), result()
        a['crossings'][':C0_c0'].update(waits=[], p95=None, entries=0)
        a['vehicles'].pop('completed_mean_depart_delay_seconds')
        c = comparison([[a]], [[b]], draws=0)
        self.assertFalse(c['descriptive_pass'])
        self.assertTrue(c['sparse'])
        self.assertIsNone(c['point']['worst_crossing_delta_seconds'])
        self.assertIsNone(c['point']['insertion_delay_delta_seconds'])
        a['demand_sha256'] = 'different'
        with self.assertRaises(ValueError):
            comparison([[a]], [[b]])
        b['crossings'][':C0_c0']['waits'][0] = float('nan')
        with self.assertRaises(ValueError):
            validate_result(b)

    def test_whole_horizon_service_keeps_its_distinct_window(self):
        a = result()['vehicles']
        b = {**a, 'scheduled_count': 12, 'completed_count': 11, 'unfinished_count': 1}
        metrics = end_vehicle_metrics([[a, b]])
        self.assertTrue(metrics['available'])
        self.assertEqual(metrics['scheduled_count'], 11)
        self.assertEqual(metrics['completed_count'], 10.5)
        self.assertFalse(end_vehicle_metrics([[a, None]])['available'])


class AuditTrialCoverageTests(unittest.TestCase):
    def test_report_requires_every_final_model_and_evaluation(self):
        (ROOT / 'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='audit-report-test-', dir=ROOT / 'runs') as folder:
            root = Path(folder)
            p = {'fingerprint': 'unit', 'methods': ['ippo'], 'variants': ['legacy', 'memory', 'separate', 'memory-separate'],
                 'replicas': 1, 'episodes': 2, 'training_scenarios': ['medium'], 'training_seed_base': 100,
                 'evaluation_scenarios': ['medium'], 'evaluation_seeds': [11], 'baselines': ['actuated', 'fixed'], 'quick': True}
            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'fingerprint': 'unit', **value}))
            save(root / 'baselines/medium-11.json', {'records': [
                {'controller': c, 'result': result(), 'network_sha256': 'net'} for c in p['baselines']]})
            missing = None
            for method, variant, replica, label in labels(p):
                directory = root / 'models' / label
                directory.mkdir(parents=True)
                (directory / 'final.pt').write_bytes(label.encode())
                sha = hashlib.sha256(label.encode()).hexdigest()
                save(directory / 'complete.json', {'episodes': 2, 'checkpoint_sha256': sha})
                history = [{'rollouts': [{'episode': i + 1, 'scenario': 'medium', 'seed': 100 + i,
                    'demand_sha256': f'demand-{i}', 'network_sha256': 'net',
                    'diagnostics': {t: {'free_decisions': 2} for t in ('C0', 'C1', 'C2')}}]} for i in range(2)]
                save(directory / 'learning.json', {'history': history})
                path = root / 'evaluation' / f'{label}-medium-11.json'
                value = {'checkpoint_sha256': sha, 'result': result(loss=8), 'network_sha256': 'net'}
                if variant == 'memory-separate':
                    missing = path, value
                else:
                    save(path, value)
            with self.assertRaisesRegex(ValueError, 'Incomplete experiment'):
                report(root, p, draws=0)
            self.assertFalse((root / 'REPORT.md').exists())
            save(*missing)
            summary = report(root, p, draws=0)
            self.assertEqual(summary['completed_models'], 4)
            self.assertEqual(len(summary['scenarios']['medium']['paired_ablation']), 5)
            self.assertTrue((root / 'REPORT.md').is_file())
            p.update(validation_episodes=[2], validation_seeds=[12])
            with self.assertRaisesRegex(ValueError, 'Validation coverage mismatch'):
                report(root, p, draws=0)
            p.pop('validation_episodes')
            p.pop('validation_seeds')
            value = json.loads(missing[0].read_text())
            value['network_sha256'] = 'wrong'
            missing[0].write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Unmatched evaluation'):
                report(root, p, draws=0)


if __name__ == '__main__':
    unittest.main()
