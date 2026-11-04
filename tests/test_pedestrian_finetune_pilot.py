"""Exercise study orchestration with real networks/Adam and a tiny fake SUMO trace.

These checks establish reproducibility and provenance, not traffic performance.
"""
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from scripts.pedestrian_finetune import pilot
from src.traffic_control.connected import incoming_groups
from src.traffic_control.ppo_repair.learning import RepairNetworks


FEATURES = {tls: [edge + '/queue' for edge in sorted(set.union(*groups.values()))]
            + [f':{tls}_c{i}/{suffix}' for i in range(4) for suffix in ('count', 'oldest')]
            + ['EW', 'NS'] for tls, groups in incoming_groups(3).items()}


def parent_bytes():
    model = RepairNetworks('mappo', FEATURES, 'normalized-clock', 12)
    for tls in FEATURES:
        model.scales[tls].update(torch.tensor([-5., -40., -12.]), model.critics[tls][-1])
    stream = io.BytesIO()
    torch.save(dict(fingerprint='parent-study', label='mappo-control-r0',
                    episodes=128, model=model.snapshot()), stream)
    return stream.getvalue()


def setup_study(root, raw):
    parent_sha = hashlib.sha256(raw).hexdigest()
    pilot.atomic_bytes(root / 'models/mappo-original-r0/final.pt', raw)
    # The runtime is mocked in these fixtures, so its frozen-source list is
    # deliberately empty; report orchestration still verifies the archive.
    with zipfile.ZipFile(root / 'source.zip', 'w'):
        pass
    return dict(fingerprint='test-pilot', replicas=1, episodes=3,
                arms=list(pilot.ARMS), baselines=list(pilot.BASELINES),
                learner=dict(pilot.LEARNER_CFG, ppo_epochs=1),
                starting_models=dict(fingerprint='parent-study', episodes=128,
                    checkpoints={'mappo-original-r0': dict(sha256=parent_sha,
                                                          parent_label='mappo-control-r0')}),
                training_scenarios=['vmedium-pmedium', 'vhigh-phigh'],
                training_seed_base=990001, training_seed_stride=1000,
                validation_episodes=[2, 3], validation_seeds=[970001],
                evaluation_scenarios=['vmedium-pmedium'], evaluation_seeds=[980001],
                bootstrap_draws=0, contrasts=[], source={}, simulation_totals={})


class TinySimulator:
    """Generate on-policy actions; consume Torch RNG to expose bad resume logic."""
    def __init__(self):
        self.calls = []

    def __call__(self, root, protocol, scenario, seed, controller, model=None, training=False):
        self.calls.append((scenario, seed, controller, training))
        rows = []
        jitter = float(torch.rand(()))
        for i in range(6):
            observations = {tls: [(.2 + jitter / 10 if n.endswith('/queue') else
                                  .2 if n.endswith('/count') else
                                  (10. + i * 10.) / 60 if n.endswith('/oldest') else
                                  float(n == 'EW')) for n in names]
                            for tls, names in FEATURES.items()}
            state = sum(observations.values(), [])
            for tls, obs in observations.items():
                mask = [False, True, False] if i == 1 else [False] * 3 if i == 2 else [True] * 3
                action = None
                if any(mask):
                    action = 1 if i == 1 else int(model.distribution(tls, obs, mask).sample()) if model else 0
                row = dict(junction=tls, time=i, observation=obs, global_state=state,
                           mask=mask, action=action, vehicle_queue=2 + i + jitter,
                           pending_count=4, uninserted_due=1)
                if action is not None and model is not None:
                    with torch.no_grad():
                        row['behavior_logp'] = float(model.distribution(tls, obs, mask).log_prob(torch.tensor(action)))
                rows.append(row)
        crossings = {f':{tls}_c{i}': {} for tls in FEATURES for i in range(4)}
        events = [dict(crossing_id=':C0_c0', requested_at=0, entered_at=1, exited_at=3),
                  dict(crossing_id=':C0_c0', requested_at=1, entered_at=4, exited_at=None),
                  dict(crossing_id=':C0_c1', requested_at=2, entered_at=None, exited_at=None)]
        return dict(result=dict(demand_sha256=f'demand-{scenario}-{seed}', crossings=crossings,
                                vehicles=dict(completed=4, unfinished=1, uninserted=1)),
                    features=copy.deepcopy(FEATURES), diagnostics={}, network_sha256=f'network-{scenario}',
                    sumo_version='mock-only', vehicles_end={}, phase_history={}, events=events,
                    rollout=dict(trace=rows, horizon=5))


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assertNestedEqual(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assertNestedEqual(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertNestedEqual(a, b)
        else:
            self.assertEqual(left, right)

    def baselines(self, root, protocol, simulator):
        with patch.object(pilot, 'episode', simulator):
            for seed in protocol['validation_seeds'] + protocol['evaluation_seeds']:
                pilot.baseline_job((root, protocol, 'vmedium-pmedium', seed))

    def test_interrupted_resume_matches_uninterrupted_weights_optimizer_rng_and_data(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='.finetune-pilot-') as folder:
            a, b = Path(folder) / 'full', Path(folder) / 'resumed'
            raw = parent_bytes()
            pa, pb = setup_study(a, raw), setup_study(b, raw)
            full, interrupted = TinySimulator(), TinySimulator()
            self.baselines(a, pa, full)
            self.baselines(b, pb, interrupted)
            with patch.object(pilot, 'episode', full), patch('builtins.print'):
                pilot.train_job((a, pa, 'guided', 0, 'mappo-guided-r0', 3))
            original_atomic = pilot.atomic_torch
            saves = 0

            def fail_second_checkpoint(path, data):
                nonlocal saves
                if Path(path).name == 'latest.pt':
                    saves += 1
                    if saves == 2:
                        raise InterruptedError('simulated failure after update, before checkpoint')
                original_atomic(path, data)

            with patch.object(pilot, 'episode', interrupted), patch('builtins.print'):
                with patch.object(pilot, 'atomic_torch', fail_second_checkpoint):
                    with self.assertRaises(InterruptedError):
                        pilot.train_job((b, pb, 'guided', 0, 'mappo-guided-r0', 3))
                latest = b / 'models/mappo-guided-r0/latest.pt'
                self.assertEqual(torch.load(latest, weights_only=True)['episodes'], 1)
                torch.manual_seed(937)  # unrelated work between interruption and resume
                pilot.train_job((b, pb, 'guided', 0, 'mappo-guided-r0', 3))
            left = torch.load(a / 'models/mappo-guided-r0/latest.pt', weights_only=True)
            right = torch.load(b / 'models/mappo-guided-r0/latest.pt', weights_only=True)
            for key in ('model', 'learner', 'rng', 'validations'):
                self.assertNestedEqual(left[key], right[key])
            for row in left['history'] + right['history']:
                row.pop('seconds')  # wall time is expected to differ
            self.assertNestedEqual(left['history'], right['history'])
            for index in range(1, 4):
                relative = f'teacher-data/mappo-guided-r0/{index:03}.json.gz'
                self.assertEqual((a / relative).read_bytes(), (b / relative).read_bytes())
            self.assertEqual([s for _, s, _, train in interrupted.calls if train],
                             [990001, 990002, 990002, 990003])
            self.assertEqual([s for _, s, _, train in full.calls if train], [990001, 990002, 990003])

    def test_matched_arm_demands_all_free_labels_and_no_evaluation_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='.finetune-pilot-') as folder:
            root = Path(folder)
            p = setup_study(root, parent_bytes())
            simulator = TinySimulator()
            self.baselines(root, p, simulator)
            histories = []
            with patch.object(pilot, 'episode', simulator), patch('builtins.print'):
                for arm in p['arms']:
                    label = f'mappo-{arm}-r0'
                    pilot.train_job((root, p, arm, 0, label, 3))
                    histories.append(json.loads((root / 'models' / label / 'learning.json').read_text())['history'])
                before = {f.relative_to(root): pilot.digest(f) for f in (root / 'teacher-data').rglob('*.gz')}
                for arm in p['arms']:
                    pilot.evaluation_job((root, p, arm, 0, f'mappo-{arm}-r0', 'vmedium-pmedium', 980001))
                after = {f.relative_to(root): pilot.digest(f) for f in (root / 'teacher-data').rglob('*.gz')}
                self.assertEqual(before, after)
            schedules = [[(h['seed'], h['scenario'], h['demand_sha256'], h['network_sha256']) for h in history]
                         for history in histories]
            self.assertEqual(schedules[0], schedules[1])
            self.assertEqual(schedules[1], schedules[2])
            self.assertEqual(len(before), 9)
            for path in before:
                data = json.loads(gzip.decompress((root / path).read_bytes()))
                self.assertTrue(data['seed'] in (990001, 990002, 990003))
                for local in data['data'].values():
                    self.assertEqual(local['times'], [0., 3., 4.])
                    self.assertEqual(local['labels'], [2, 2, 2])
                    self.assertEqual(local['guidance_weights'], [0., 1., 1.])
                self.assertEqual(data['counts']['terminal_rows_excluded'], 3)
            self.assertTrue(all(not train for _, s, _, train in simulator.calls if s in (970001, 980001)))
            # The report must recheck pairing rather than trust that callers
            # previously invoked evaluation_job on every serialized result.
            with patch.object(pilot, 'episode', simulator):
                pilot.evaluation_job((root, p, 'original', 0, 'mappo-original-r0', 'vmedium-pmedium', 980001))
            with patch.object(pilot, 'verify_source'), \
                 patch('scripts.pedestrian_finetune.reporting.aggregate_scenario', return_value={}), \
                 patch('scripts.pedestrian_finetune.reporting.write_report') as writer:
                pilot.report(root, p)
                writer.assert_called_once()
                result_path = root / 'evaluation/mappo-guided-r0-vmedium-pmedium-980001.json'
                data = json.loads(result_path.read_text())
                data['result']['demand_sha256'] = 'other-demand'
                pilot.atomic_json(result_path, data)
                with self.assertRaisesRegex(ValueError, 'Unmatched demand/network'):
                    pilot.report(root, p)

    def test_cached_evaluation_rejects_unpaired_demand_or_changed_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='.finetune-pilot-') as folder:
            root = Path(folder)
            p = setup_study(root, parent_bytes())
            simulator = TinySimulator()
            self.baselines(root, p, simulator)
            args = (root, p, 'original', 0, 'mappo-original-r0', 'vmedium-pmedium', 980001)
            with patch.object(pilot, 'episode', simulator):
                path = Path(pilot.evaluation_job(args))
                original = json.loads(path.read_text())
                broken = copy.deepcopy(original)
                broken['result']['demand_sha256'] = 'other-demand'
                pilot.atomic_json(path, broken)
                with self.assertRaisesRegex(ValueError, 'Unmatched demand/network'):
                    pilot.evaluation_job(args)
                broken = copy.deepcopy(original)
                broken['checkpoint_sha256'] = 'other-model'
                pilot.atomic_json(path, broken)
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    pilot.evaluation_job(args)

    def test_parent_archive_manifest_and_existing_parent_weights_are_verified(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='.finetune-pilot-') as folder:
            root = Path(folder)
            raw = parent_bytes()
            protocol = dict(quick=False, episodes=128, replicas=3, fingerprint='parent-study',
                            source={'source.py': 'source-sha'}, initial_seeds=[1, 2, 3])
            members = {pilot.PARENT_PREFIX + 'protocol.json': json.dumps(protocol).encode(),
                       pilot.PARENT_PREFIX + 'models/mappo-control-r0/final.pt': raw}
            manifest = {key: hashlib.sha256(value).hexdigest() for key, value in members.items()}
            archive = root / 'parent.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('manifest.json', json.dumps(manifest))
                for key, value in members.items():
                    z.writestr(key, value)
            p = setup_study(root / 'study', raw)
            p['starting_models'] = pilot.archive_parent_info(archive, 1)
            pilot.prepare_parents(root / 'study', p, archive)
            parent_file = root / 'study/models/mappo-original-r0/final.pt'
            saved = torch.load(parent_file, weights_only=True)
            saved['model']['weights'][next(iter(saved['model']['weights']))].add_(1)
            pilot.atomic_torch(parent_file, saved)
            with self.assertRaisesRegex(ValueError, 'Wrong original checkpoint'):
                pilot.prepare_parents(root / 'study', p, archive)
            with zipfile.ZipFile(archive, 'a') as z:
                z.writestr('unexpected.txt', 'changed archive')
            with self.assertRaisesRegex(ValueError, 'Parent archive changed'):
                pilot.prepare_parents(root / 'study', p, archive)

    def test_crossing_completion_keeps_pending_distinct_from_entered_without_exit(self):
        out = TinySimulator()(None, None, 'medium', 10, 'fixed')
        counts = pilot.crossing_completion(out)
        self.assertEqual(counts[':C0_c0'], dict(entered_without_observed_exit=1, completed_visits=1))
        self.assertEqual(counts[':C0_c1'], dict(entered_without_observed_exit=0, completed_visits=0))
        self.assertEqual(len(counts), 12)

    def test_source_archive_rejects_changed_contents(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='.finetune-pilot-') as folder:
            root = Path(folder)
            p = {'source': {'fixture.py': hashlib.sha256(b'original').hexdigest()}}
            with zipfile.ZipFile(root / 'source.zip', 'w') as z:
                z.writestr('fixture.py', b'original')
            pilot.verify_source_archive(root, p)
            with zipfile.ZipFile(root / 'source.zip', 'w') as z:
                z.writestr('fixture.py', b'changed')
            with self.assertRaises(ValueError):
                pilot.verify_source_archive(root, p)


if __name__ == '__main__':
    unittest.main()
