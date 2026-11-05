"""Stage 2: constrained MAPPO learner (per-crossing Lagrangian on the RepairLearner path).

Extends ppo_repair's RepairLearner with the per-crossing cost critics + dual
multipliers from fairness/learning.py, so the actor optimises "no crossing's p95
worse" directly instead of a dense summed pedestrian penalty.

VERIFIED interface facts (see docs/stage2_plan.md):
  * main critic input  = 127 (global_state 126 + time-to-horizon clock), via
    model.critic_inputs()/model.prediction() with PopArt RunningValueScale.
  * cost critic input  = 126 (RAW global_state, NO clock), 4 outputs/junction,
    via model.cost_critics[tls](global_state).  <-- LANDMINE 1: never feed it the
    127-dim clock tensor.
  * checkpoint cost_critics are untrained dead weight -> reinitialized here.  <-- L2
  * 4 sorted crossings per junction :C{j}_c0..c3 map to the 4 cost outputs.

Reward, GAE, PPO epochs, clip, entropy, grad-clip, forced-row handling all match
RepairLearner exactly; only the constraint is added.
"""
from __future__ import annotations

import math
import torch

from src.traffic_control.fairness.learning import advantages
from src.traffic_control.fairness.networks import mlp
from src.traffic_control.ppo_repair.learning import RepairLearner, raw_return, prediction_stats


class ConstrainedRepairLearner(RepairLearner):
    """RepairLearner + per-crossing Lagrangian constraint.

    thresholds: {crossing_id: max_wait_seconds}. multipliers start at 0 and rise
    while the crossing violates beyond a 5% slack. The cost advantage is subtracted
    from the policy advantage, so the actor is pushed to keep each crossing's wait
    under threshold without a hand-tuned reward weight.
    """

    def __init__(self, model, weight=3, thresholds=None, cost_seed=940000):
        super().__init__(model, weight)
        self.thresholds = thresholds or {}
        self.multipliers = {c: 0. for c in self.thresholds}
        # LANDMINE 2: the checkpoint's cost_critics are untrained. Reinitialize them
        # deterministically on the RAW 126-dim global_state (NOT the 127 clock input).
        if self.thresholds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cost_seed)
                for tls in model.features:
                    model.cost_critics[tls] = mlp(model.global_dim, 4)
            # Adam must track the fresh cost-critic params.
            self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    def _cost_tensors(self, tls, rows, events, device):
        """Per-step cost (len(rewards), 4) and per-crossing violation rates, exactly
        as fairness/learning.py:59-73. rows are this junction's trace rows."""
        n = len(rows) - 1
        costs = torch.zeros((n, 4), device=device)
        local = sorted(c for c in self.thresholds if c.startswith(f":{tls}_c"))
        rates = {}
        for j, crossing in enumerate(local):
            requests = [e for e in events if e["crossing_id"] == crossing]
            threshold = self.thresholds[crossing]
            violations = 0
            for event in requests:
                bad = event["wait_seconds"] is None or event["wait_seconds"] > threshold
                violations += bad
                t = event["entered_at"] if event["entered_at"] is not None else rows[-1]["time"]
                index = min(n - 1, max(0, math.ceil(t) - 2))
                costs[index, j] += (float(bad) - .05) / max(1, len(requests))
            rates[crossing] = (violations / len(requests)) if requests else None
        return costs, local, rates

    def update(self, rollouts, events=None):
        if not self.thresholds:
            return super().update(rollouts)
        if not rollouts:
            raise ValueError("Empty rollout batch")
        if events is None:
            raise ValueError("Constrained update requires crossing events")
        device = next(self.model.parameters()).device

        def tensor(v, dtype=torch.float32):
            return torch.as_tensor(v, dtype=dtype, device=device)

        results = {}
        for tls in self.model.features:
            rows = [r for r in rollouts[0]["trace"] if r["junction"] == tls]
            # Single-rollout update (pilot calls update per episode); concatenation
            # across multi-rollout batches is handled by looping callers if needed.
            if len(rollouts) != 1:
                raise ValueError("ConstrainedRepairLearner expects one rollout per update")
            if len(rows) < 2:
                raise ValueError("Rollout too short")
            obs = tensor([r["observation"] for r in rows[:-1]])
            state = tensor([r["global_state"] for r in rows[:-1]])        # 126-dim, cost critic
            inputs = self.model.critic_inputs(obs, state,
                                              [r["time"] for r in rows[:-1]], rollouts[0]["horizon"])  # 127-dim
            actions = tensor([r["action"] if r["action"] is not None else 0 for r in rows[:-1]], torch.long)
            valid = tensor([r["action"] is not None and sum(r["mask"]) > 1 for r in rows[:-1]], torch.bool)
            masks = tensor([r["mask"] if any(r["mask"]) else [True, False, False] for r in rows[:-1]], torch.bool)
            rewards = [-(r["vehicle_queue"] + r["uninserted_due"] + self.weight * r["pending_count"]) / 10
                       for r in rows[1:]]
            with torch.no_grad():
                values = self.model.prediction(tls, inputs)
                old_cost = self.model.cost_critics[tls](state)            # LANDMINE 1: raw state
                old_logp = self.model.distribution(tls, obs, masks).log_prob(actions)
            adv, returns = advantages(rewards, values.tolist())
            adv, returns = tensor(adv), tensor(returns)

            costs, local, rates = self._cost_tensors(tls, rows, events, device)
            cost_returns = torch.flip(torch.cumsum(torch.flip(costs, [0]), dim=0), [0])
            cost_adv = cost_returns - old_cost
            if local:
                lambdas = tensor([self.multipliers[c] for c in local])
                adv = adv - (cost_adv * lambdas).sum(-1)
            if valid.any():
                adv = (adv - adv[valid].mean()) / adv[valid].std(unbiased=False).clamp_min(1e-6)

            # PopArt value-scale update on the main critic (RepairLearner behaviour).
            if self.model.repaired:
                with torch.no_grad():
                    before = self.model.prediction(tls, inputs).clone()
                self.model.scales[tls].update(returns, self.model.critics[tls][-1])
                with torch.no_grad():
                    after = self.model.prediction(tls, inputs)
                if not torch.allclose(before, after, atol=1e-4, rtol=1e-5):
                    raise AssertionError("Normalisation changed raw value predictions")

            for _ in range(4):
                distribution = self.model.distribution(tls, obs, masks)
                logp = distribution.log_prob(actions)
                ratio = (logp - old_logp).exp()
                surrogate = torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv)
                actor_loss = -surrogate[valid].mean() if valid.any() else logp.sum() * 0
                if self.model.repaired:
                    normalized = self.model.critics[tls](inputs).squeeze(-1)
                    value_loss = (normalized - self.model.scales[tls].normalize(returns)).square().mean()
                else:
                    value_loss = ((self.model.prediction(tls, inputs) - returns)
                                  / returns.std(unbiased=False).clamp_min(1.)).square().mean()
                cost_value = self.model.cost_critics[tls](state)          # raw state
                cost_loss = (cost_value - cost_returns).square().mean()
                entropy = distribution.entropy()[valid].mean() if valid.any() else actor_loss * 0
                loss = actor_loss + .5 * value_loss + .5 * cost_loss - .01 * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not valid.any():
                    for parameter in self.model.actors[tls].parameters():
                        parameter.grad = None
                gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)
                if not torch.isfinite(gradient):
                    raise FloatingPointError("Nonfinite gradient")
                self.optimizer.step()

            # Dual ascent on the per-crossing multipliers (fairness/learning.py:102).
            for crossing, rate in rates.items():
                if rate is not None:
                    self.multipliers[crossing] = min(100., max(0., self.multipliers[crossing]
                                                               + .05 * (rate - .05)))
            with torch.no_grad():
                value = self.model.prediction(tls, inputs)
            results[tls] = {"loss": float(loss.detach()), "actor_loss": float(actor_loss.detach()),
                            "value_loss": float(value_loss.detach()), "cost_loss": float(cost_loss.detach()),
                            "entropy": float(entropy.detach()), "gradient_norm": float(gradient),
                            "decisions": int(valid.sum()), "states": len(valid),
                            "multipliers": {c: round(self.multipliers[c], 4) for c in local},
                            "violation_rates": {c: rates[c] for c in local},
                            "after_fit_gae": prediction_stats(returns, value)}
        return results
