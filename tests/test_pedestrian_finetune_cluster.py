"""Cluster scheduling must preserve complete independent model/seed coverage."""
import unittest
from scripts.pedestrian_finetune_cluster import stages


class ClusterScheduleTests(unittest.TestCase):
    def test_full_schedule_covers_all_models_and_pairs(self):
        p = dict(replicas=3, arms=['continue', 'tail', 'guided'], validation_seeds=[970001, 970002],
                 checkpoint_stages=[16, 32, 64], evaluation_seeds=list(range(980001, 980005)),
                 evaluation_scenarios=['vmedium-pmedium', 'pedestrian-burst', 'short-spacing', 'slow-walkers'])
        schedule = list(stages(p))
        self.assertEqual([x[0] for x in schedule], ['prepare', 'train-16', 'train-32', 'train-64',
                                                  'evaluation-baselines', 'evaluate'])
        self.assertEqual(len(schedule[0][2]), 2)
        for row, target in zip(schedule[1:4], [16, 32, 64]):
            self.assertEqual(len(set(row[2])), 9)
            self.assertEqual({j[-1] for j in row[2]}, {target})
            self.assertNotIn('original', {j[0] for j in row[2]})
        self.assertEqual(len(set(schedule[4][2])), 16)
        evaluation = schedule[5][2]
        expected = {(a, r, f'mappo-{a}-r{r}', s, seed) for a in ['original'] + p['arms']
                    for r in range(3) for s in p['evaluation_scenarios'] for seed in p['evaluation_seeds']}
        self.assertEqual(set(evaluation), expected)
        self.assertEqual(len(evaluation), 192)

    def test_smoke_uses_only_smoke_protocol_counts(self):
        p = dict(replicas=1, arms=['continue', 'tail', 'guided'], validation_seeds=[970001],
                 checkpoint_stages=[1, 2], evaluation_seeds=[980001],
                 evaluation_scenarios=['vmedium-pmedium'])
        schedule = list(stages(p))
        self.assertEqual(len(schedule[-1][2]), 4)
        self.assertEqual(len(schedule[-2][2]), 1)
        self.assertEqual([len(s[2]) for s in schedule[1:3]], [3, 3])


if __name__ == '__main__':
    unittest.main()
