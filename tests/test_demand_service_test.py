"""Intervention legality, isolation, and fixed experimental coverage."""
import unittest
from scripts.demand_service_test import EarlyServicePolicy, jobs, shared

torch = shared.torch


class Dummy:
    observation_schema = 'legacy-v2'
    features = {'C0': [f':C0_c{i}/count' for i in range(4)]}

    def distribution(self, junction, observations, masks):
        legal = torch.as_tensor(masks, dtype=torch.bool)
        weights = torch.tensor([.7, .2, .1]) * legal
        return torch.distributions.Categorical(probs=weights)


class EarlyServiceTests(unittest.TestCase):
    def setUp(self):
        self.base = Dummy()
        self.policy = EarlyServicePolicy(self.base)

    def test_pending_legal_ped_is_required(self):
        self.assertEqual(self.policy.distribution('C0', [0, .2, 0, 0], [1, 1, 1]).probs.tolist(), [0., 0., 1.])

    def test_does_not_bypass_minimum_or_clearance_mask(self):
        for mask in ([1, 0, 0], [1, 1, 0]):
            a = self.policy.distribution('C0', [.2, 0, 0, 0], mask).probs
            b = self.base.distribution('C0', [.2, 0, 0, 0], mask).probs
            self.assertTrue(torch.equal(a, b))

    def test_no_pending_preserves_policy_even_for_empty_ped_choice(self):
        self.assertTrue(torch.equal(self.policy.distribution('C0', [0]*4, [1]*3).probs,
                                    self.base.distribution('C0', [0]*4, [1]*3).probs))

    def test_batched_diagnostics_and_single_action_agree(self):
        obs = [[.2, 0, 0, 0], [0]*4, [0, 0, 0, .4]]
        masks = [[1]*3, [1]*3, [1, 1, 0]]
        batch = self.policy.distribution('C0', obs, masks).probs
        for i in range(3):
            self.assertTrue(torch.equal(batch[i], self.policy.distribution('C0', obs[i], masks[i]).probs))

    def test_no_simulations_or_replicas_omitted(self):
        p = dict(evaluation_seeds=list(range(10)), evaluation_scenarios=list('abcd'),
                 baselines=['fixed', 'actuated', 'pressure'], replicas=3,
                 arms=['original', 'continue', 'tail', 'guided', 'early'])
        rows = jobs(p)
        self.assertEqual(len(rows), 720)
        self.assertEqual(len(set(rows)), 720)
        self.assertEqual(sum(j[2] == 'mappo-early' for j in rows), 120)


if __name__ == '__main__':
    unittest.main()
