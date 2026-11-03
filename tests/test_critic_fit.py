import io
import unittest
import torch
from scripts.critic_fit_check import TargetScale, FixedCritic, discounted_returns, prediction_metrics, fit_pair
from src.traffic_control.fairness.learning import advantages


class CriticFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_computed_returns(self):
        self.assertEqual(discounted_returns([1, 2, 3], 1), [6, 5, 3])
        self.assertEqual(discounted_returns([1, 2], .5), [2, 2])

    def test_units_roundtrip_and_gae(self):
        scale = TargetScale.fit(torch.tensor([-500., -100., -300.]))
        raw = torch.tensor([-200., -350.])
        decoded = scale.denormalize(scale.normalize(raw))
        self.assertTrue(torch.allclose(decoded, raw))
        a, t = advantages([1, 2], decoded.tolist(), .5, 1)
        self.assertEqual(t, [2, 2])
        self.assertEqual(a, [202, 352])
        self.assertEqual(float(TargetScale.fit(torch.tensor([3., 3.])).scale), 1.)

    def test_state_and_optimizer_resume(self):
        x = torch.tensor([[-1.], [0.], [1.]])
        y = torch.tensor([-200., -100., -50.])
        first = FixedCritic(1, 'normalized-output', 4, TargetScale.fit(y))
        optimizer = torch.optim.Adam(first.parameters(), lr=3e-4)
        def step(model, opt):
            opt.zero_grad(); model.loss(x, y).backward(); opt.step()
        step(first, optimizer)
        buffer = io.BytesIO()
        torch.save({'model': first.state_dict(), 'optimizer': optimizer.state_dict()}, buffer)
        buffer.seek(0); saved = torch.load(buffer, weights_only=True)
        resumed = FixedCritic(1, 'normalized-output', 999, TargetScale())
        resumed.load_state_dict(saved['model'])
        opt2 = torch.optim.Adam(resumed.parameters(), lr=3e-4)
        opt2.load_state_dict(saved['optimizer'])
        self.assertTrue(torch.equal(first(x), resumed(x)))
        step(first, optimizer); step(resumed, opt2)
        self.assertTrue(torch.equal(first(x), resumed(x)))

    def test_same_initial_weights_and_no_check_data_leakage(self):
        x = torch.tensor([[-1.], [0.], [1.]])
        y = torch.tensor([-200., -100., -50.])
        a = FixedCritic(1, 'raw-scaled-loss', 8, TargetScale.fit(y))
        b = FixedCritic(1, 'normalized-output', 8, TargetScale.fit(y))
        self.assertTrue(all(torch.equal(v, b.network.state_dict()[k]) for k, v in a.network.state_dict().items()))
        left = fit_pair((x, y), {'heldout': (x, y)}, 1, 8, budgets=(3,))
        right = fit_pair((x, y), {'heldout': (x, y-10000)}, 1, 8, budgets=(3,))
        self.assertEqual([r['assessments']['training'] for r in left], [r['assessments']['training'] for r in right])

    def test_constant_predictor_is_not_a_learned_improvement(self):
        y = torch.tensor([-200., -100., -50.])
        metrics = prediction_metrics(y, torch.full_like(y, -150), torch.tensor(-150.))
        self.assertEqual(metrics['mse_ratio_to_training_mean'], 1.)
        self.assertAlmostEqual(metrics['explained_variance'], 0.)

    def test_clock_control_starts_with_same_predictions(self):
        from scripts.critic_horizon_check import clock_model
        y = torch.tensor([-200., -100., -50.])
        zero_clock = torch.tensor([[-1., 0.], [0., 0.], [1., 0.]])
        real_clock = zero_clock.clone(); real_clock[:, -1] = torch.tensor([.8, .4, .1])
        model = clock_model(2, 8, y)
        self.assertTrue(torch.equal(model(zero_clock), model(real_clock)))


if __name__ == '__main__':
    unittest.main()
