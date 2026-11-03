"""Independent live checks for a matched initial-policy SUMO comparison.

These short simulations are engineering fixtures, not evidence of trained gains.
"""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.local/sumo-1.27.1/tools'))

import torch

from src.traffic_control.audit_trial.learning import TrialNetworks
from src.traffic_control.audit_trial.learning import TrialLearner
from src.traffic_control.audit_trial.learning import prepare_trial_batch
from src.traffic_control.audit_trial.pilot import (make_protocol, labels, exclusive_lock,
    train_job, verify_source, atomic_torch, episode)
from src.traffic_control.fairness.control_observation import FEATURES as MEMORY_FEATURES
from src.traffic_control.fairness.environment import run
from src.traffic_control.scenario import executable


try:
    SUMO_READY = bool(importlib.util.find_spec('traci') and executable('sumo') and executable('netconvert'))
except RuntimeError:
    SUMO_READY = False


@unittest.skipUnless(SUMO_READY, 'SUMO and TraCI required')
class AuditTrialIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (ROOT / 'runs').mkdir(exist_ok=True)
        cls.scratch = tempfile.TemporaryDirectory(prefix='audit-trial-integration-', dir=ROOT / 'runs')
        cls.folder = Path(cls.scratch.name)
        cls.cfg = dict(end_seconds=180, warmup_seconds=0, measurement_end_seconds=120,
                       demand_end_seconds=90)
        metrics = run('actuated', 990001, cls.folder / 'reference', overrides=cls.cfg)
        cls.features = metrics['observation_features']

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def evaluate(self, model, name):
        streams = {t: random.Random(991001 + i * 1000003) for i, t in enumerate(model.features)}
        trace = []
        probabilities = []

        def policy(tls, observation, state, mask):
            legal = [i for i, allowed in enumerate(mask) if allowed]
            with torch.no_grad():
                probs = model.distribution(tls, observation, mask).probs.tolist()
            probabilities.append((tls, probs))
            return legal[0] if len(legal) == 1 else streams[tls].choices(
                legal, weights=[probs[i] for i in legal])[0]

        before = copy.deepcopy(model.state_dict())
        metrics = run('policy', 990001, self.folder / name,
                      overrides={**self.cfg, 'observation_schema': model.observation_schema},
                      policy=policy, trace_sink=trace)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))
        return metrics, trace, probabilities

    def test_initial_functions_and_traffic_match_and_evaluation_is_read_only(self):
        keys = ('vehicles_end', 'vehicles_measurement_cohort', 'persons',
                'pedestrians_by_crossing_end', 'phase_history', 'demand_sha256', 'network_sha256')
        for method in ('ippo', 'mappo'):
            old = TrialNetworks(method, self.features, 'legacy', 94001)
            new = TrialNetworks(method, self.features, 'memory', 94001)
            left, lt, lp = self.evaluate(old, method + '-legacy')
            right, rt, rp = self.evaluate(new, method + '-memory')
            for key in keys:
                self.assertEqual(left[key], right[key], (method, key))
            self.assertEqual([(r['time'], r['junction'], r['action'], r['mask']) for r in lt],
                             [(r['time'], r['junction'], r['action'], r['mask']) for r in rt])
            self.assertEqual(len(lp), len(rp))
            for (a, p), (b, q) in zip(lp, rp):
                self.assertEqual(a, b)
                self.assertTrue(torch.allclose(torch.tensor(p), torch.tensor(q), atol=1e-7, rtol=1e-6))
            exposed = 0
            for row in rt:
                observed = dict(zip(right['observation_features'][row['junction']], row['observation']))
                memory = [observed[name] for name in MEMORY_FEATURES]
                exposed += any(memory)
                if row['action'] is not None:
                    self.assertFalse(any(memory), 'Controller memory unexpectedly affects an actor choice')
            self.assertGreater(exposed, 0, 'No forced transition was exercised')

    def test_runner_replay_log_probabilities_and_actual_actor_decision_count(self):
        p = make_protocol(quick=True)
        model = TrialNetworks('mappo', self.features, 'memory-separate', 94001)
        model.eval()
        before = copy.deepcopy(model.state_dict())
        left = episode(self.folder, p, 'vmedium-pmedium', 990002, 'policy', model)
        right = episode(self.folder, p, 'vmedium-pmedium', 990002, 'policy', model)
        for key in ('result', 'diagnostics', 'network_sha256', 'vehicles_end', 'phase_history', 'events', 'rollout'):
            self.assertEqual(left[key], right[key], key)
        for tls in model.features:
            batch = prepare_trial_batch(model, tls, [left['rollout']])
            self.assertEqual(left['diagnostics'][tls]['free_decisions'], int(batch['valid'].sum()))
            self.assertEqual(batch['old_logp'].shape[0], 179)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))


class AuditTrialRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (ROOT / 'runs').mkdir(exist_ok=True)

    def test_protocol_counts_seed_isolation_and_fixed_final_checkpoint(self):
        p = make_protocol(False)
        models = len(labels(p))
        self.assertEqual(models, 16)
        self.assertEqual(p['episodes'], 64)
        train = {p['training_seed_base'] + r * p['training_seed_stride'] + i
                 for r in range(p['replicas']) for i in range(p['episodes'])}
        domains = [train, set(p['validation_seeds']), set(p['evaluation_seeds']),
                   set(p['reserved_final_seeds_not_used'])]
        for i, a in enumerate(domains):
            for b in domains[i + 1:]:
                self.assertFalse(a & b)
        groups = ({(s, d) for s in p['evaluation_scenarios'] for d in p['evaluation_seeds']}
                  | {('vmedium-pmedium', d) for d in p['validation_seeds']})
        counts = (models * p['episodes'], models * len(p['validation_episodes']) * len(p['validation_seeds']),
                  models * len(p['evaluation_scenarios']) * len(p['evaluation_seeds']), len(groups) * len(p['baselines']))
        self.assertEqual(counts, (1024, 48, 256, 68))
        self.assertEqual(sum(counts), 1396)
        self.assertEqual(p['reward_weight'], 3)
        self.assertFalse(p['backlog_input'])
        self.assertEqual(p['checkpoint_stages'][-1], p['episodes'])

    def test_duplicate_writer_and_changed_source_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='audit-lock-', dir=ROOT / 'runs') as folder:
            with exclusive_lock(Path(folder)):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(Path(folder)):
                        self.fail('Duplicate writer acquired the same study')
        with self.assertRaises(RuntimeError):
            verify_source({'source': {'src/traffic_control/nonexistent-audit-file.py': 'missing'}})

    def test_completed_atomic_checkpoint_recovers_exports_without_training(self):
        features = {t: ['queue', 'phase', 'legal_EW', 'legal_NS', 'legal_PED'] for t in ('C0', 'C1', 'C2')}
        with tempfile.TemporaryDirectory(prefix='audit-resume-', dir=ROOT / 'runs') as folder:
            root = Path(folder)
            (root / 'baselines').mkdir()
            (root / 'baselines/one.json').write_text(json.dumps({'records': [{'features': features}]}))
            label = 'mappo-memory-separate-r0'
            directory = root / 'models' / label
            directory.mkdir(parents=True)
            model = TrialNetworks('mappo', features, 'memory-separate', 94001)
            learner = TrialLearner(model)
            history = [{'episodes': 4, 'rollouts': []}]
            atomic_torch(directory / 'latest.pt', {'fingerprint': 'unit', 'label': label, 'episodes': 4,
                'model': model.snapshot(), 'learner': learner.state(), 'rng': torch.get_rng_state(),
                'history': history, 'validations': []})
            p = {'fingerprint': 'unit', 'initial_seeds': [94001], 'reward_weight': 3, 'episodes': 4}
            with patch('src.traffic_control.audit_trial.pilot.episode',
                       side_effect=AssertionError('Must not retrain an already completed checkpoint')):
                train_job((str(root), p, 'mappo', 'memory-separate', 0, label, 4))
            self.assertEqual(json.loads((directory / 'learning.json').read_text())['history'], history)
            self.assertEqual(json.loads((directory / 'complete.json').read_text())['episodes'], 4)
            saved = torch.load(directory / 'final.pt', weights_only=True)
            self.assertTrue(all(torch.equal(v, saved['model']['weights'][k]) for k, v in model.state_dict().items()))

    def test_episode_boundary_resume_matches_uninterrupted_training(self):
        features = {t: ['queue', 'phase', 'legal_EW', 'legal_NS', 'legal_PED'] for t in ('C0', 'C1', 'C2')}
        p = {'fingerprint': 'unit', 'initial_seeds': [94001], 'reward_weight': 3, 'episodes': 4,
             'training_scenarios': ['vmedium-pmedium', 'vmedium-plow', 'pedestrian-burst', 'short-spacing'],
             'training_seed_base': 930001, 'training_seed_stride': 1000, 'validation_episodes': [],
             'validation_seeds': []}
        label = 'mappo-memory-separate-r0'

        def fake_episode(root, protocol, scenario, seed, controller, model, training=False):
            self.assertTrue(training)
            rng = random.Random(seed)
            trace = []
            for tick in range(1, 7):
                local = {}
                for i, tls in enumerate(model.features):
                    values = {'queue': float(i + tick % 3 + seed % 2), 'phase': float(tick % 2),
                              'legal_EW': 1., 'legal_NS': 1., 'legal_PED': 0.}
                    local[tls] = [values.get(name, 0.) for name in model.features[tls]]
                state = [x for tls in model.features for x in local[tls]]
                for i, tls in enumerate(model.features):
                    with torch.no_grad():
                        dist = model.distribution(tls, local[tls], [True, True, False])
                        action = rng.choices([0, 1], weights=dist.probs[:2].tolist())[0]
                        logp = float(dist.log_prob(torch.tensor(action)))
                    trace.append({'junction': tls, 'time': tick, 'observation': local[tls], 'global_state': state,
                                  'mask': [True, True, False], 'action': action, 'behavior_logp': logp,
                                  'vehicle_queue': tick + i, 'pending_count': tick % 2, 'uninserted_due': i})
            return {'rollout': {'trace': trace, 'horizon': 6},
                    'result': {'demand_sha256': f'{scenario}-{seed}', 'vehicles': {'fixture': True}},
                    'network_sha256': 'fixture-network',
                    'diagnostics': {t: {'free_decisions': 5} for t in features}}

        with tempfile.TemporaryDirectory(prefix='audit-resume-learning-', dir=ROOT / 'runs') as folder:
            roots = [Path(folder) / suffix for suffix in ('full', 'resumed')]
            for root in roots:
                (root / 'baselines').mkdir(parents=True)
                (root / 'baselines/one.json').write_text(json.dumps({'records': [{'features': features}]}))
            with patch('src.traffic_control.audit_trial.pilot.episode', side_effect=fake_episode):
                train_job((str(roots[0]), p, 'mappo', 'memory-separate', 0, label, 4))
                train_job((str(roots[1]), p, 'mappo', 'memory-separate', 0, label, 2))
                train_job((str(roots[1]), p, 'mappo', 'memory-separate', 0, label, 4))
            a, b = [torch.load(root / 'models' / label / 'latest.pt', weights_only=True) for root in roots]
            self.assertEqual(a['episodes'], b['episodes'])
            self.assertTrue(all(torch.equal(v, b['model']['weights'][k]) for k, v in a['model']['weights'].items()))
            for left, right in zip(a['history'], b['history']):
                self.assertEqual(left['rollouts'], right['rollouts'])
                self.assertEqual(left['update'], right['update'])
            for param, values in a['learner']['optimizer']['state'].items():
                for name, value in values.items():
                    other = b['learner']['optimizer']['state'][param][name]
                    self.assertTrue(torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other)


if __name__ == '__main__':
    unittest.main()
