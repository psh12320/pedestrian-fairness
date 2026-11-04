"""Teacher advice is local, phase-safe and excludes forced/terminal labels."""
import copy
from types import SimpleNamespace
import unittest

from scripts.pedestrian_finetune.teacher import crossing_tail_cost, labelled_rollout, query_teacher
from src.traffic_control.fairness.signals import baseline_action


def sample(tls='C0', ew=1, ns=0, queue=(.5, .7, .4, .8), pending=1, age=25., mask=(1, 1, 1)):
    i = int(tls[1:])
    names = [f'h{i}e/queue', f'h{i+1}w/queue', f'ni{i}/queue', f'si{i}/queue',
             f':{tls}_c0/count', f':{tls}_c0/oldest', 'EW', 'NS',
             'legal_EW', 'legal_NS', 'legal_PED']
    values = [*queue, pending/5, age/60, ew, ns, *mask]
    return names, values


class TeacherTests(unittest.TestCase):
    def test_tail_cost_threshold_cap_and_pending_crossings(self):
        for age, pending, want in ((10,1,0), (20,1,0), (40,1,1), (60,1,4),
                                   (120,1,25), (1000,1,25), (1000,0,0)):
            names, obs = sample(age=age, pending=pending)
            self.assertAlmostEqual(crossing_tail_cost(obs, names), want)
        names, obs = sample(age=60)
        names += [':C0_c1/count', ':C0_c1/oldest']
        obs += [2/5, 100/60]
        self.assertAlmostEqual(crossing_tail_cost(obs, names), 20.)

    def test_tail_cost_rejects_nonlocal_and_invalid_scaling(self):
        names, obs = sample()
        for kwargs in ({'scale_seconds':0}, {'cap':-1}, {'threshold_seconds':float('nan')}):
            with self.assertRaises(ValueError):
                crossing_tail_cost(obs, names, **kwargs)
        names += [':C1_c1/count', ':C1_c1/oldest']
        obs += [1, 1]
        with self.assertRaises(ValueError):
            crossing_tail_cost(obs, names)

    def test_exact_actuated_actions_across_queue_phase_mask_cases(self):
        for tls in ('C0', 'C1', 'C2'):
            for stage in ('EW', 'NS', 'READY'):
                for mask in ((1,1,1), (1,1,0), (0,1,1), (1,0,0)):
                    for q in ((0,0,0,0), (.5,.7,.4,.8), (.1,.2,.8,.7)):
                        for pending in (0, 2):
                            names, obs = sample(tls, stage=='EW', stage=='NS', q, pending, mask=mask)
                            signal = SimpleNamespace(stage=stage, mask=lambda now, demand: mask)
                            queues = [sum(round(x*10) for x in q[:2]), sum(round(x*10) for x in q[2:])]
                            expected = baseline_action('actuated', signal, 100, queues, [0,0], pending)
                            got = query_teacher(tls, obs, names, mask, urgency_seconds=20)
                            self.assertEqual(got['action'], expected)
                            self.assertTrue(mask[got['action']])

    def test_urgent_guidance_only_when_free_pedestrian_action_legal(self):
        for age, pending, mask, expected in ((19.9,1,(1,1,1),False), (20,1,(1,1,1),True),
                                           (100,0,(1,1,1),False), (100,1,(1,1,0),False),
                                           (100,1,(0,0,1),False)):
            names, obs = sample(age=age, pending=pending, mask=mask)
            got = query_teacher('C0', obs, names, mask, urgency_seconds=20)
            self.assertEqual(got['guidance_selected'], expected)
            self.assertFalse(got['weight_is_confidence'])

    def test_closed_transition_has_no_teacher_action(self):
        names, obs = sample(mask=(0,0,0))
        with self.assertRaises(ValueError):
            query_teacher('C0', obs, names, (0,0,0), urgency_seconds=20)

    def test_rejects_nonlocal_features_and_mask_misalignment(self):
        names, obs = sample()
        with self.assertRaises(ValueError):
            query_teacher('C1', obs, names, (1,1,1), urgency_seconds=20)
        with self.assertRaises(ValueError):
            query_teacher('C0', obs, names, (1,1,0), urgency_seconds=20)

    def test_rollout_keeps_nonurgent_and_disagreeing_labels_without_outcome_filter(self):
        names, urgent = sample(age=30)
        _, young = sample(age=2)
        _, forced = sample(mask=(1,0,0))
        rows = [dict(junction='C0', time=1, observation=forced, mask=[1,0,0], action=0),
                dict(junction='C0', time=12, observation=urgent, mask=[1,1,1], action=0,
                     reward=-9999, unresolved_requests=100),
                dict(junction='C0', time=13, observation=young, mask=[1,1,1], action=2),
                dict(junction='C0', time=600, observation=urgent, mask=[1,1,1], action=2)]
        before = copy.deepcopy(rows)
        result = labelled_rollout(rows, {'C0': names}, horizon=600, urgency_seconds=20)
        self.assertEqual(result['data']['C0']['times'], [12., 13.])
        self.assertEqual(result['data']['C0']['labels'], [2,2])
        self.assertEqual(result['data']['C0']['guidance_weights'], [1.,0.])
        self.assertEqual(result['counts'], {'forced_rows_excluded':1, 'free_decisions':2,
                          'selected_guidance_decisions':1, 'student_teacher_disagreements':1,
                          'terminal_rows_excluded':1})
        self.assertEqual(rows, before)
        self.assertEqual(result, labelled_rollout(rows, {'C0':names}, horizon=600, urgency_seconds=20))

    def test_rejects_illegal_or_missing_student_decision(self):
        names, obs = sample(mask=(1,1,0))
        for action in (None, 2, True):
            rows = [dict(junction='C0', time=10, observation=obs, mask=[1,1,0], action=action)]
            with self.assertRaises(ValueError):
                labelled_rollout(rows, {'C0':names}, horizon=600, urgency_seconds=20)

    def test_threshold_and_duplicate_rows_are_rejected(self):
        names, obs = sample()
        with self.assertRaises(ValueError):
            query_teacher('C0', obs, names, [1,1,1], urgency_seconds=float('nan'))
        row = dict(junction='C0', time=10, observation=obs, mask=[1,1,1], action=0)
        with self.assertRaises(ValueError):
            labelled_rollout([row,row], {'C0':names}, horizon=600, urgency_seconds=20)


if __name__ == '__main__':
    unittest.main()
