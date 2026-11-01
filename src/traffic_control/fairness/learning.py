"""Finite-horizon neural PPO and per-crossing Lagrangian extension.

Reference: marlbenchmark/on-policy. This is an independent implementation.
Rollouts retain forced-transition states for critic training, but mask their actor
loss. Reward at observation t+1 belongs to the action at t, never vice versa.
"""
import math
import torch


def advantages(rewards, values, gamma=.99, decay=.95):
    if len(rewards) != len(values):
        raise ValueError("Reward/value lengths differ")
    result = [0.] * len(rewards)
    running = 0.
    for i in reversed(range(len(rewards))):
        following = values[i + 1] if i + 1 < len(values) else 0.
        delta = rewards[i] + gamma * following - values[i]
        running = delta + gamma * decay * running
        result[i] = running
    return result, [a + v for a, v in zip(result, values)]


class Learner:
    def __init__(self, model, weight=3, thresholds=None, lr=3e-4):
        self.model, self.weight = model, weight
        self.thresholds = thresholds
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.multipliers = {c: 0. for c in thresholds or {}}
        self.last_statistics = {}

    def policy(self, tls, obs, global_state, mask):
        return self.model.act(tls, obs, mask)[0]

    def update(self, trace, events):
        torch.set_num_threads(1)
        all_stats = {}
        device = next(self.model.parameters()).device
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=device)
        for tls in self.model.features:
            rows = [r for r in trace if r["junction"] == tls]
            if len(rows) < 2:
                raise ValueError("Rollout too short")
            # The final observation supplies the last reward; no action after
            # the simulation horizon contributes a fictitious transition.
            observations = tensor([r["observation"] for r in rows[:-1]])
            states = tensor([r["global_state"] for r in rows[:-1]])
            actions = tensor([r["action"] or 0 for r in rows[:-1]], torch.long)
            valid = tensor([r["action"] is not None and sum(r["mask"]) > 1 for r in rows[:-1]], torch.bool)
            masks = tensor([r["mask"] if any(r["mask"]) else [True, False, False] for r in rows[:-1]], torch.bool)
            rewards = [-(r["vehicle_queue"] + r["uninserted_due"] + self.weight * r["pending_count"]) / 10
                       for r in rows[1:]]
            with torch.no_grad():
                old_value, old_cost = self.model.values(tls, observations, states)
                old_logp = self.model.distribution(tls, observations, masks).log_prob(actions)
            adv, returns = advantages(rewards, old_value.tolist())
            adv, returns = tensor(adv), tensor(returns)
            costs = torch.zeros((len(rewards), 4), device=device)
            local_crossings = sorted(c for c in self.thresholds or {} if c.startswith(f":{tls}_c"))
            rates = {}
            for j, crossing in enumerate(local_crossings):
                requests = [e for e in events if e["crossing_id"] == crossing]
                threshold = self.thresholds[crossing]
                violations = 0
                for event in requests:
                    # Unknown approach and pending requests are conservative violations.
                    bad = event["wait_seconds"] is None or event["wait_seconds"] > threshold
                    violations += bad
                    time = (event["entered_at"] if event["entered_at"] is not None else rows[-1]["time"])
                    index = min(len(rewards) - 1, max(0, math.ceil(time) - 2))
                    costs[index, j] += (float(bad) - .05) / max(1, len(requests))
                rates[crossing] = (violations / len(requests) if requests else None)
            # Undiscounted cost returns: each crossing is a separate constraint.
            cost_returns = torch.flip(torch.cumsum(torch.flip(costs, [0]), dim=0), [0])
            cost_adv = cost_returns - old_cost
            if local_crossings:
                lambdas = tensor([self.multipliers[c] for c in local_crossings])
                adv = adv - (cost_adv * lambdas).sum(-1)
            if valid.any():
                adv = (adv - adv[valid].mean()) / adv[valid].std(unbiased=False).clamp_min(1e-6)
            # Scale the value loss, preserving critic output units and avoiding
            # inconsistent renormalisation of stored old value predictions.
            value_scale = returns.std(unbiased=False).clamp_min(1.)
            for _ in range(4):
                distribution = self.model.distribution(tls, observations, masks)
                logp = distribution.log_prob(actions)
                ratio = (logp - old_logp).exp()
                surrogate = torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv)
                actor_loss = -surrogate[valid].mean() if valid.any() else logp.sum() * 0
                value, cost_value = self.model.values(tls, observations, states)
                value_loss = ((value - returns) / value_scale).square().mean()
                cost_loss = (cost_value - cost_returns).square().mean() if local_crossings else value_loss * 0
                entropy = distribution.entropy()[valid].mean() if valid.any() else actor_loss * 0
                loss = actor_loss + .5 * value_loss + .5 * cost_loss - .01 * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)
                self.optimizer.step()
            for crossing, rate in rates.items():
                if rate is not None:
                    self.multipliers[crossing] = min(100., max(0., self.multipliers[crossing] + .05 * (rate - .05)))
            all_stats[tls] = {"loss": float(loss.detach()), "entropy": float(entropy.detach()),
                              "gradient_norm": float(gradient), "decisions": int(valid.sum()),
                              "constraint_rates": rates}
        self.last_statistics = all_stats
        return all_stats
