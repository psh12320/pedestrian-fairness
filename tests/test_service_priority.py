import unittest
from src.traffic_control.service_priority.control import ProtectedPolicy, service_mask, service_components


class ServicePriorityTests(unittest.TestCase):
    def test_rule_only_removes_legal_actions(self):
        names = [':C0_c0/count', ':C0_c0/oldest']
        for bits in range(8):
            legal = tuple(bool(bits & (1 << i)) for i in range(3))
            for obs in ([0, 0], [1, 1]):
                mask = service_mask(obs, names, legal)
                self.assertTrue(all(not a or b for a, b in zip(mask, legal)))
        self.assertEqual(service_mask([1, 1], names, (True, False, False)), (True, False, False))
        self.assertEqual(service_mask([1, 1], names, (False, False, False)), (False, False, False))
        self.assertEqual(service_mask([1, 0], names, (True, True, True)), (False, False, True))

    def test_age_trigger_and_empty_crossings(self):
        names = [':C0_c0/count', ':C0_c0/oldest']
        self.assertEqual(service_mask([1, .1], names, (True, True, True), 20), (True, True, True))
        self.assertEqual(service_mask([1, 1], names, (True, True, True), 20), (False, False, True))
        self.assertEqual(service_mask([0, 0], names, (True, True, True)), (True, True, False))

    def test_training_uses_effective_sampling_mask(self):
        names = [':C0_c0/count', ':C0_c0/oldest']
        policy = ProtectedPolicy({'C0': names})
        trace = [{'junction':'C0', 'observation':[1, 0], 'mask':(True, True, True), 'action':2},
                 {'junction':'C0', 'observation':[1, 1], 'mask':(False, False, False), 'action':None}]
        policy.align_training_trace(trace)
        self.assertEqual(sum(trace[0]['mask']), 1)
        self.assertEqual(trace[0]['safety_mask'], (True, True, True))
        self.assertFalse(any(trace[1]['mask']))
        with self.assertRaises(AssertionError):
            policy.align_training_trace([{'junction':'C0', 'observation':[1,0], 'mask':(True,True,True), 'action':0}])

    def test_no_double_counted_person_total(self):
        base={'vehicles':{'completed_count':5,'unfinished_count':0,'uninserted_count':0},
              'persons':{'completed':5,'unfinished':0,'uninserted':0},
              'crossings':{'c':{'pending_at_cutoff':0,'unresolved_after_followup':0,'unknown_cohort_entries':0,'lost':0,'entries':5}}}
        import copy
        row=copy.deepcopy(base)
        row['persons'].update(completed=4,unfinished=1)
        row['crossings']['c']['pending_at_cutoff']=1
        components=service_components(row,base)
        self.assertEqual(components['person_unfinished_delta'],1)
        self.assertEqual(components['pending_at_cutoff_delta'],1)
        self.assertNotIn('total_unserved',components)


if __name__ == '__main__':
    unittest.main()
