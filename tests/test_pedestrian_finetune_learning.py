"""Learning integrity checks; traffic benefit is determined by held-out SUMO."""
import copy
import io
import unittest

import torch

from scripts.pedestrian_finetune.learning import FineTuneLearner
from src.traffic_control.connected import incoming_groups
from src.traffic_control.ppo_repair.learning import RepairNetworks


FEATURES = {tls: [e + '/queue' for e in sorted(set.union(*groups.values()))]
            + [f':{tls}_c{i}/{suffix}' for i in range(4) for suffix in ('count', 'oldest')]
            + ['EW', 'NS'] for tls, groups in incoming_groups(3).items()}


def parent_model(seed=7):
    model = RepairNetworks('mappo', FEATURES, 'normalized-clock', seed)
    for tls in FEATURES:
        model.scales[tls].update(torch.tensor([-30., -10., -60., -20.]), model.critics[tls][-1])
    return model


def rollout(model, *, forced=False, young=False, mixed=False):
    rows = []
    for i in range(7):
        local = {}
        for tls, names in FEATURES.items():
            local[tls] = [(.2 if n.endswith('/queue') else
                           .2 if n.endswith('/count') else
                           (0. if young else (10 + i*10)/60) if n.endswith('/oldest') else
                           float(n == 'EW')) for n in names]
        state = sum(local.values(), [])
        for tls in FEATURES:
            mask = [False, True, False] if forced else [True, True, True]
            action = 1 if forced else i % 3
            if mixed and i in (1, 3):
                mask, action = ([False, True, False] if i == 1 else [False, False, False]), None
            elif mixed and i == 4:
                mask, action = [True, False, True], 2
            row = {'junction': tls, 'time': i, 'observation': local[tls], 'global_state': state,
                   'mask': mask, 'action': action, 'vehicle_queue': i+2,
                   'pending_count': 4, 'uninserted_due': 1}
            if action is not None:
                with torch.no_grad():
                    row['behavior_logp'] = float(model.distribution(tls, local[tls], mask).log_prob(torch.tensor(action)))
            rows.append(row)
    return {'trace': rows, 'horizon': 6}


def equal_state(a, b):
    return all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items())


class FineTuneLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_constructor_preserves_full_warm_start_and_starts_fresh_optimizer(self):
        parent = parent_model()
        for arm in ('continue', 'tail', 'guided'):
            student = RepairNetworks.from_snapshot(parent.snapshot())
            before = copy.deepcopy(student)
            learner = FineTuneLearner(student, parent, arm, {})
            self.assertTrue(equal_state(student, before))
            self.assertTrue(equal_state(learner.reference, parent))
            self.assertEqual(len(learner.optimizer.state), 0)
            self.assertEqual(learner.optimizer.param_groups[0]['lr'], 1e-4)
            self.assertTrue(all(not p.requires_grad for p in learner.reference.parameters()))

    def test_zero_guidance_matches_tail_arm_exactly(self):
        parent = parent_model()
        a, b = copy.deepcopy(parent), copy.deepcopy(parent)
        left = FineTuneLearner(a, parent, 'tail', {'guidance_weight': 0.})
        right = FineTuneLearner(b, parent, 'guided', {'guidance_weight': 0.})
        for i in range(2):
            left.update([rollout(a)], i)
            right.update([rollout(b)], i)
        self.assertTrue(equal_state(a, b))

    def test_no_urgent_labels_matches_tail_and_keeps_all_labels(self):
        parent = parent_model()
        a, b = copy.deepcopy(parent), copy.deepcopy(parent)
        left, right = FineTuneLearner(a, parent, 'tail', {}), FineTuneLearner(b, parent, 'guided', {})
        left.update([rollout(a, young=True)])
        result = right.update([rollout(b, young=True)])
        self.assertTrue(equal_state(a, b))
        self.assertTrue(all(s['teacher_labels'] == 6 and s['teacher_selected'] == 0 for s in result.values()))
        self.assertEqual(right.last_labels['counts']['free_decisions'], 18)
        self.assertEqual(right.last_labels['counts']['terminal_rows_excluded'], 3)

    def test_guidance_has_actor_effect_and_preserves_reference(self):
        parent = parent_model()
        original_parent = copy.deepcopy(parent)
        a, b = copy.deepcopy(parent), copy.deepcopy(parent)
        unguided, guided = FineTuneLearner(a, parent, 'tail', {}), FineTuneLearner(b, parent, 'guided', {})
        unguided.update([rollout(a)])
        result = guided.update([rollout(b)])
        self.assertFalse(equal_state(a.actors, b.actors))
        self.assertTrue(equal_state(guided.reference, original_parent))
        self.assertTrue(equal_state(parent, original_parent))
        self.assertTrue(all(p.grad is None for p in guided.reference.parameters()))
        self.assertTrue(all(s['teacher_selected'] == 5 and s['guidance_weight'] == .5 for s in result.values()))
        self.assertTrue(all(s['parent_kl_after'] >= -1e-7 for s in result.values()))

    def test_guidance_loss_divides_by_all_free_decisions(self):
        parent = parent_model()
        learner = FineTuneLearner(copy.deepcopy(parent), parent, 'guided', {'ppo_epochs': 1})
        data = rollout(learner.model)
        expected = {}
        for tls in FEATURES:
            rows = [r for r in data['trace'] if r['junction'] == tls][:-1]
            with torch.no_grad():
                dist = learner.model.distribution(tls, [r['observation'] for r in rows], [r['mask'] for r in rows])
                ce = -dist.log_prob(torch.full((6,), 2))
                expected[tls] = float(ce[1:].sum()/6)
        stats = learner.update([data])
        for tls in FEATURES:
            self.assertAlmostEqual(stats[tls]['teacher_loss'], expected[tls], places=6)

    def test_forced_actor_remains_identical_after_prior_adam_momentum(self):
        parent = parent_model()
        for arm in ('continue', 'tail', 'guided'):
            student = copy.deepcopy(parent)
            learner = FineTuneLearner(student, parent, arm, {})
            learner.update([rollout(student)])
            before_actor = copy.deepcopy(student.actors)
            before_critic = copy.deepcopy(student.critics)
            stats = learner.update([rollout(student, forced=True)], 1)
            self.assertTrue(equal_state(student.actors, before_actor))
            self.assertFalse(equal_state(student.critics, before_critic))
            self.assertTrue(all(s['decisions'] == 0 and s['actor_gradient_norm'] == 0 for s in stats.values()))

    def test_mixed_masks_and_absent_forced_actions_are_finite(self):
        parent = parent_model()
        learner = FineTuneLearner(copy.deepcopy(parent), parent, 'guided', {})
        stats = learner.update([rollout(learner.model, mixed=True)])
        self.assertTrue(all(s['decisions'] == 4 and s['teacher_selected'] == 3 for s in stats.values()))
        self.assertTrue(all(torch.isfinite(p).all() for p in learner.model.parameters()))

    def test_rejects_stale_log_probability_without_any_model_update(self):
        parent = parent_model()
        student = copy.deepcopy(parent)
        learner = FineTuneLearner(student, parent, 'guided', {})
        data = rollout(student)
        data['trace'][0]['behavior_logp'] += .5
        with self.assertRaisesRegex(ValueError, 'on-policy'):
            learner.update([data])
        self.assertTrue(equal_state(student, parent))

    def test_target_moments_are_extended_without_raw_prediction_change(self):
        parent = parent_model()
        learner = FineTuneLearner(copy.deepcopy(parent), parent, 'tail', {})
        result = learner.update([rollout(learner.model)])
        for tls in FEATURES:
            self.assertEqual(result[tls]['normalizer_count'], 10.)
            self.assertLess(result[tls]['normalizer_raw_prediction_max_error'], 1e-4)
            self.assertLess(result[tls]['reward_mean'], result[tls]['base_reward_mean'])

    def test_exact_resume_and_wrong_parent_or_arm_rejected(self):
        parent = parent_model()
        learner = FineTuneLearner(copy.deepcopy(parent), parent, 'guided', {})
        learner.update([rollout(learner.model)])
        stream = io.BytesIO()
        torch.save({'model': learner.model.snapshot(), 'learner': learner.state()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = FineTuneLearner(RepairNetworks.from_snapshot(saved['model']), parent, 'guided', {})
        restored.restore(saved['learner'])
        data = rollout(learner.model)
        a = learner.update([data], 1)
        b = restored.update([data], 1)
        self.assertTrue(equal_state(learner.model, restored.model))
        self.assertEqual(a, b)
        for reference, arm, cfg in ((parent_model(9), 'guided', {}),
                                    (parent, 'tail', {}), (parent, 'guided', {'learning_rate': 2e-4})):
            bad = FineTuneLearner(copy.deepcopy(parent), reference, arm, cfg)
            with self.assertRaises(ValueError):
                bad.restore(saved['learner'])

    def test_incomplete_episode_and_invalid_config_are_rejected(self):
        parent = parent_model()
        learner = FineTuneLearner(copy.deepcopy(parent), parent, 'guided', {})
        data = rollout(learner.model)
        data['trace'] = data['trace'][:-3]
        with self.assertRaisesRegex(ValueError, 'terminal'):
            learner.update([data])
        for cfg in ({'ppo_epochs': 1.5}, {'gamma': 2.}, {'learning_rate': -1.}, {'guidance_weight': float('nan')}):
            with self.assertRaises(ValueError):
                FineTuneLearner(copy.deepcopy(parent), parent, 'guided', cfg)


if __name__ == '__main__':
    unittest.main()
