import random
import unittest
from scripts.audit_service_policy import choose_action, explained_variance


class PolicyAuditTests(unittest.TestCase):
    def test_argmax_near_tie_repeats_preferred_action(self):
        rng=random.Random(1)
        self.assertEqual({choose_action([.49,.51,0],[True,True,False],'argmax',rng) for _ in range(100)}, {1})

    def test_sampling_is_seeded_and_respects_mask(self):
        def draws(seed):
            rng=random.Random(seed)
            return [choose_action([.49,.51,0],[True,True,False],'sample',rng) for _ in range(100)]
        self.assertEqual(draws(2),draws(2))
        self.assertEqual(set(draws(2)),{0,1})
        self.assertTrue(all(choose_action([0,0,1],[False,False,True],mode,random.Random(3))==2
                            for mode in ('argmax','sample','uniform')))

    def test_variance_handles_constant_targets(self):
        self.assertIsNone(explained_variance([1,1],[1,1]))
        self.assertEqual(explained_variance([1,2,3],[1,2,3]),1)
        self.assertEqual(explained_variance([1,2,3],[0,0,0]),0)


if __name__=='__main__':
    unittest.main()
