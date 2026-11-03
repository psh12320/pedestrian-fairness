import copy
import io
from types import SimpleNamespace
import unittest
import torch
from src.traffic_control.connected import incoming_groups
from src.traffic_control.fairness.signals import baseline_action
from src.traffic_control.ppo_repair.learning import RepairNetworks, RepairLearner
from src.traffic_control.guided_exploration.learning import (
    teacher_action, rewards_for, StudyLearner, Novelty, demonstrations, imitate, Moments)

CFG = {'age_scale_seconds': 60., 'age_cap_units': 5., 'age_weight': 1.,
       'guidance_weight': .1, 'guidance_decay_episodes': 128, 'rnd_weight': .05, 'rnd_bonus_cap': 5.}
FEATURES = {t: [e+'/queue' for e in sorted(set.union(*groups.values()))] +
    [f':{t}_c{i}/{suffix}' for i in range(4) for suffix in ('count', 'oldest')] + ['EW', 'NS']
    for t, groups in incoming_groups(3).items()}


def trace(forced=False):
    rows = []
    for i in range(6):
        local = {}
        for tls, names in FEATURES.items():
            local[tls] = [1. if n.endswith('/queue') else (.2 if n.endswith('/count') else
                (i/6 if n.endswith('/oldest') else float(n == 'EW'))) for n in names]
        state = sum(local.values(), [])
        for tls in FEATURES:
            rows.append({'junction': tls, 'time': i+1, 'observation': local[tls], 'global_state': state,
                'mask': [True, False, False] if forced else [True, True, True],
                'action': 0 if forced else i % 3, 'vehicle_queue': i+2, 'pending_count': 4, 'uninserted_due': 0})
    return {'trace': rows, 'horizon': 6}


class GuidedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_teacher_matches_baseline_under_masks_and_queues(self):
        tls = 'C0'; names = FEATURES[tls]
        for stage in ('EW', 'NS', 'READY'):
            for pending in (0, 1):
                for queues in ([0, 0], [1, 0], [0, 1], [2, 3]):
                    for mask in ([1,1,1], [1,0,0], [0,1,1], [1,1,0]):
                        values = dict.fromkeys(names, 0.)
                        for axis, q in zip(('EW', 'NS'), queues):
                            edge = sorted(incoming_groups(3)[tls][axis])[0]
                            values[edge+'/queue'] = q/10
                        values[':C0_c0/count'] = pending/5
                        for axis in ('EW', 'NS'): values[axis] = float(stage == axis)
                        signal = SimpleNamespace(stage=stage, mask=lambda *args: mask)
                        expected = baseline_action('actuated', signal, 10, queues, [0,0], pending)
                        self.assertEqual(teacher_action(tls, [values[n] for n in names], names, mask), expected)

    def test_dense_reward_distinguishes_age_without_changing_counts(self):
        row = trace()['trace'][0]; names = FEATURES['C0']
        old = rewards_for([row], names, 'age', CFG)[0]
        row['observation'][names.index(':C0_c0/oldest')] = 2
        self.assertAlmostEqual(old - rewards_for([row], names, 'age', CFG)[0], 4.)
        self.assertEqual(rewards_for([row], names, 'control', CFG)[0], -1.4)
        row['observation'][names.index(':C0_c0/oldest')] = 1000
        self.assertAlmostEqual(old - rewards_for([row], names, 'age', CFG)[0], 25.)
        row['observation'][names.index(':C0_c0/count')] = 0
        self.assertEqual(rewards_for([row], names, 'age', CFG)[0], old)

    def test_teacher_integer_queue_tie_after_clearance(self):
        tls='C2'; names=FEATURES[tls]; values=dict.fromkeys(names,0.)
        for axis, counts in [('EW',[.5,.7]),('NS',[.4,.8])]:
            for edge, count in zip(sorted(incoming_groups(3)[tls][axis]),counts):
                values[edge+'/queue']=count
        self.assertEqual(teacher_action(tls,[values[n] for n in names],names,[True,True,False]),0)

    def test_control_matches_repaired_ppo(self):
        for method in ('ippo', 'mappo'):
            a = RepairNetworks(method, FEATURES, 'normalized-clock', 7)
            b = RepairNetworks(method, FEATURES, 'normalized-clock', 7)
            original = RepairLearner(a); new = StudyLearner(b, 'control', CFG, 7)
            for _ in range(2): original.update([trace()]); new.update([trace()])
            for key, v in a.state_dict().items():
                self.assertTrue(torch.allclose(v, b.state_dict()[key], atol=2e-6, rtol=1e-5), key)

    def test_forced_decisions_never_train_actor_even_after_momentum(self):
        for variant in ('control', 'age', 'guided', 'rnd'):
            model = RepairNetworks('mappo', FEATURES, 'normalized-clock', 7)
            learner = StudyLearner(model, variant, CFG, 7); learner.update([trace()])
            before = copy.deepcopy(model.actors.state_dict())
            stats = learner.update([trace(True)])
            self.assertTrue(all(torch.equal(v, model.actors.state_dict()[k]) for k, v in before.items()))
            self.assertTrue(all(s['decisions'] == 0 for s in stats.values()))

    def test_rnd_fixed_target_bounded_bonus_and_resume(self):
        model = RepairNetworks('mappo', FEATURES, 'normalized-clock', 7)
        learner = StudyLearner(model, 'rnd', CFG, 7)
        target = copy.deepcopy(learner.novelty.target.state_dict())
        stats = learner.update([trace()])
        self.assertTrue(all(0 <= d['intrinsic_bonus_mean'] <= 5 for d in stats.values()))
        self.assertTrue(all(torch.equal(v, learner.novelty.target.state_dict()[k]) for k,v in target.items()))
        self.assertTrue(all(p.grad is None for p in learner.novelty.target.parameters()))
        stream = io.BytesIO(); torch.save({'model': model.snapshot(), 'learner': learner.state()}, stream)
        stream.seek(0); saved = torch.load(stream, weights_only=True)
        restored = RepairNetworks.from_snapshot(saved['model']); other = StudyLearner(restored, 'rnd', CFG, 7)
        other.restore(saved['learner']); learner.update([trace()], 1); other.update([trace()], 1)
        self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(torch.equal(v, other.novelty.state_dict()[k]) for k,v in learner.novelty.state_dict().items()))

    def test_imitation_moves_only_actor_and_rejects_wrong_teacher(self):
        model = RepairNetworks('mappo', FEATURES, 'normalized-clock', 7)
        rollout = trace()
        for row in rollout['trace']: row['action'] = 2
        data = demonstrations(model, [{'rollout': rollout}])
        before = copy.deepcopy(model.critics.state_dict())
        stats = imitate(model, data, 30, 16, 7)
        self.assertTrue(all(s['training_teacher_probability'] > .8 for s in stats.values()))
        self.assertTrue(all(torch.equal(v, model.critics.state_dict()[k]) for k,v in before.items()))
        rollout['trace'][0]['action'] = 0
        with self.assertRaises(AssertionError): demonstrations(model, [{'rollout': rollout}])

    def test_guidance_decays_to_zero_without_eval_module(self):
        model = RepairNetworks('mappo', FEATURES, 'normalized-clock', 7)
        learner = StudyLearner(model, 'guided', CFG, 7)
        stats = learner.update([trace()], 128)
        self.assertTrue(all(s['guidance_weight'] == 0 for s in stats.values()))
        restored = RepairNetworks.from_snapshot(model.snapshot())
        self.assertFalse(hasattr(restored, 'novelty'))

    def test_moments_merge_and_eval_normalization_read_only(self):
        m = Moments(2); x = torch.tensor([[1.,2.], [3.,4.], [8.,1.]])
        m.update(x[:2]); m.update(x[2:]); before = copy.deepcopy(m.state_dict())
        self.assertTrue(torch.allclose(m.mean, x.double().mean(0)))
        m.normalize(x)
        self.assertTrue(all(torch.equal(v, m.state_dict()[k]) for k,v in before.items()))


if __name__ == '__main__': unittest.main()
