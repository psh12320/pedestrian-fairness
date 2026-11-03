import copy
import io
import unittest

import torch

from src.traffic_control.audit_trial.learning import (
    ARMS, TrialNetworks, TrialLearner, clip_gradients, critic_backlogs,
    gradient_norm, prepare_trial_batch,
)
from src.traffic_control.fairness.control_observation import FEATURES as MEMORY_FEATURES
from src.traffic_control.ppo_repair.learning import RepairNetworks, RepairLearner, prepare_batch


FEATURES = {t: ['queue', 'phase', 'legal_EW', 'legal_NS', 'legal_PED'] for t in ('C0', 'C1')}


def rows_for(model, forced=False):
    trace = []
    for i in range(9):
        locals_ = {}
        for junction_index, tls in enumerate(model.features):
            values = {'queue': float(junction_index + i % 3), 'phase': float(i % 2),
                      'legal_EW': 1., 'legal_NS': float(not forced), 'legal_PED': 0.}
            values.update({name: float(i % 3) if forced else 0. for name in MEMORY_FEATURES})
            locals_[tls] = [values[n] for n in model.features[tls]]
        state = [x for tls in model.features for x in locals_[tls]]
        for junction_index, tls in enumerate(model.features):
            trace.append({'junction': tls, 'time': i + 1, 'observation': locals_[tls],
                'global_state': state, 'mask': [True, not forced, False],
                'action': 0 if forced else i % 2, 'vehicle_queue': i + 2,
                'uninserted_due': junction_index * 10 + i, 'pending_count': i % 2})
    return trace


def rollout(model, forced=False):
    return {'trace': rows_for(model, forced), 'horizon': 9}


class AuditTrialLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_update_is_exactly_the_existing_normalized_clock_update(self):
        for method in ('ippo', 'mappo'):
            previous = RepairNetworks(method, FEATURES, 'normalized-clock', 94001)
            current = TrialNetworks(method, FEATURES, 'legacy', 94001)
            old_learner, new_learner = RepairLearner(previous), TrialLearner(current)
            batch = [rollout(current)]
            for _ in range(2):
                a = old_learner.update(batch)
                b = new_learner.update(batch)
                for tls in FEATURES:
                    for key in ('loss', 'actor_loss', 'value_loss', 'entropy', 'gradient_norm',
                                'approx_kl_after', 'normalizer_mean', 'normalizer_scale'):
                        self.assertEqual(a[tls][key], b[tls][key], (method, tls, key))
            for key, value in previous.state_dict().items():
                self.assertTrue(torch.equal(value, current.state_dict()[key]), (method, key))

    def test_memory_and_backlog_preserve_initial_functions(self):
        for method in ('ippo', 'mappo'):
            base = TrialNetworks(method, FEATURES, 'legacy', 94001)
            for arm in ARMS:
                for backlog in (False, True):
                    model = TrialNetworks(method, FEATURES, arm, 94001, backlog)
                    for tls in FEATURES:
                        old = prepare_trial_batch(base, tls, [rollout(base)])
                        new = prepare_trial_batch(model, tls, [rollout(model)])
                        self.assertTrue(torch.allclose(base.actors[tls](old['observations']),
                            model.actors[tls](new['observations']), atol=1e-7, rtol=1e-6))
                        self.assertTrue(torch.allclose(old['old_values'], new['old_values'],
                            atol=1e-6, rtol=1e-6), (method, arm, backlog))
                        self.assertEqual(model.actors[tls][0].in_features, len(model.features[tls]))
                        self.assertEqual(model.observation_schema,
                            'controller-memory-v1' if 'memory' in arm else 'legacy-v2')

    def test_separate_clipping_does_not_scale_small_actor_gradient_by_large_critic(self):
        models = [TrialNetworks('ippo', FEATURES, arm, 4) for arm in ('legacy', 'separate')]
        for model in models:
            for p in model.actors['C0'].parameters():
                p.grad = torch.full_like(p, .001)
            for p in model.critics['C0'].parameters():
                p.grad = torch.full_like(p, 10.)
        before = gradient_norm(models[0].actors['C0'].parameters())
        self.assertLess(before, .5)
        clip_gradients(models[0], 'C0')
        clip_gradients(models[1], 'C0')
        self.assertLess(gradient_norm(models[0].actors['C0'].parameters()), before / 100)
        self.assertAlmostEqual(gradient_norm(models[1].actors['C0'].parameters()), before, places=7)
        self.assertAlmostEqual(gradient_norm(models[1].critics['C0'].parameters()), .5, places=5)

    def test_forced_batch_keeps_actor_fixed_even_with_adam_momentum(self):
        for arm in ARMS:
            model = TrialNetworks('ippo', FEATURES, arm, 8)
            learner = TrialLearner(model)
            learner.update([rollout(model)])
            before = copy.deepcopy(model.actors.state_dict())
            stats = learner.update([rollout(model, True)])
            self.assertTrue(all(torch.equal(v, model.actors.state_dict()[k]) for k, v in before.items()))
            self.assertTrue(all(v['decisions'] == 0 for v in stats.values()))

    def test_model_optimizer_and_normalizer_resume_exactly(self):
        for arm in ARMS:
            model = TrialNetworks('mappo', FEATURES, arm, 8)
            learner = TrialLearner(model)
            batch = [rollout(model)]
            learner.update(batch)
            stream = io.BytesIO()
            torch.save({'model': model.snapshot(), 'learner': learner.state()}, stream)
            stream.seek(0)
            saved = torch.load(stream, weights_only=True)
            other_model = TrialNetworks.from_snapshot(saved['model'])
            other = TrialLearner(other_model)
            other.restore(saved['learner'])
            left, right = learner.update(batch), other.update(batch)
            self.assertEqual(left, right)
            self.assertTrue(all(torch.equal(v, other_model.state_dict()[k]) for k, v in model.state_dict().items()))

    def test_backlog_is_critic_only_and_synchronized(self):
        for method in ('ippo', 'mappo'):
            model = TrialNetworks(method, FEATURES, 'legacy', 4, backlog=True)
            self.assertEqual(model.features, FEATURES)
            trace = rows_for(model)
            local = [r for r in trace if r['junction'] == 'C0']
            values = critic_backlogs(model, 'C0', local, trace)
            self.assertEqual(values[0], [0.] if method == 'ippo' else [0., 1.])
            b = prepare_trial_batch(model, 'C0', [{'trace': trace, 'horizon': 9}])
            self.assertEqual(b['inputs'].shape[1], 7 if method == 'ippo' else 13)
            with self.assertRaises(ValueError):
                model.critic_inputs(b['observations'], torch.zeros(len(local) - 1, 10),
                                    list(range(1, 9)), 9)
        broken = [r for r in trace if not (r['junction'] == 'C1' and r['time'] == 1)]
        with self.assertRaises(ValueError):
            critic_backlogs(model, 'C0', local, broken)

    def test_checkpoint_rejects_wrong_treatment_and_schema(self):
        model = TrialNetworks('ippo', FEATURES, 'legacy', 8)
        state = TrialLearner(model).state()
        with self.assertRaises(ValueError):
            TrialLearner(TrialNetworks('ippo', FEATURES, 'separate', 8)).restore(state)
        saved = model.snapshot()
        saved['features']['C0'].reverse()
        with self.assertRaises(ValueError):
            TrialNetworks.from_snapshot(saved)

    def test_on_policy_alignment_and_episode_boundaries(self):
        model = TrialNetworks('ippo', FEATURES, 'memory', 4)
        one = rollout(model)
        b = prepare_trial_batch(model, 'C0', [one, one])
        self.assertEqual(len(b['returns']), 16)
        final = one['trace'][-2]
        expected = -(final['vehicle_queue'] + final['uninserted_due'] + 3 * final['pending_count']) / 10
        self.assertAlmostEqual(float(b['returns'][7]), expected, places=6)
        self.assertAlmostEqual(float(b['returns'][15]), expected, places=6)
        one['trace'][0]['behavior_logp'] = -100
        with self.assertRaises(ValueError):
            prepare_trial_batch(model, 'C0', [one])
        one['trace'][0].pop('behavior_logp')
        one['trace'][0]['action'] = 2
        with self.assertRaises(ValueError):
            prepare_trial_batch(model, 'C0', [one])


if __name__ == '__main__':
    unittest.main()
