"""Matched initial policies for observation and gradient-clipping ablations.

The default is numerically equivalent to normalized-clock RepairLearner. New
controller-memory columns start at zero, preserving the initial policy/value
functions. Optional external backlog is critic-only: one local count for IPPO,
the synchronized network count vector for MAPPO. No old learner is modified.
"""
import copy
import math

import torch
from torch import nn

from ..fairness.control_observation import FEATURES as MEMORY_FEATURES
from ..fairness.learning import advantages
from ..ppo_repair.learning import RepairNetworks, prediction_stats, raw_return
from . import SCHEMA


ARMS = ("legacy", "memory", "separate", "memory-separate")


def memory_features(names):
    """Insert exactly the same columns as fairness.environment.run."""
    if len(set(names)) != len(names) or any(n in MEMORY_FEATURES for n in names):
        raise ValueError("Expected unique legacy feature names")
    position = next((i for i, name in enumerate(names) if name.startswith("legal_")), len(names))
    return list(names[:position]) + list(MEMORY_FEATURES) + list(names[position:])


def expanded_mlp(old, inputs, old_to_new):
    """Copy a network, adding only zero first-layer columns; keep RNG unchanged."""
    if len(old_to_new) != old[0].in_features or len(set(old_to_new)) != len(old_to_new):
        raise ValueError("Invalid input-column map")
    if min(old_to_new, default=0) < 0 or max(old_to_new, default=0) >= inputs:
        raise ValueError("Input-column map out of bounds")
    new = copy.deepcopy(old)
    with torch.random.fork_rng(devices=[]):
        first = nn.Linear(inputs, old[0].out_features, device=old[0].weight.device,
                          dtype=old[0].weight.dtype)
    with torch.no_grad():
        first.weight.zero_()
        first.weight[:, old_to_new] = old[0].weight
        first.bias.copy_(old[0].bias)
    new[0] = first
    return new


class TrialNetworks(RepairNetworks):
    def __init__(self, method, legacy_features, arm="legacy", seed=93001, backlog=False):
        if arm not in ARMS:
            raise ValueError("Unknown audit arm: " + str(arm))
        self.arm, self.seed, self.backlog = arm, seed, bool(backlog)
        self.legacy_features = {tls: list(names) for tls, names in sorted(legacy_features.items())}
        if any(len(set(names)) != len(names) or any(n in MEMORY_FEATURES for n in names)
               for names in self.legacy_features.values()):
            raise ValueError("TrialNetworks expects the legacy observation schema")
        super().__init__(method, self.legacy_features, "normalized-clock", seed)
        self.memory = arm in ("memory", "memory-separate")
        self.separate_clipping = arm in ("separate", "memory-separate")
        self.observation_schema = "controller-memory-v1" if self.memory else "legacy-v2"
        original_local, original_global = self.local_dim, self.global_dim
        if self.memory:
            self.features = {tls: memory_features(names) for tls, names in self.legacy_features.items()}
            self.local_dim = len(next(iter(self.features.values())))
            self.global_dim = self.local_dim * len(self.features)
        local_maps = {tls: [self.features[tls].index(name) for name in names]
                      for tls, names in self.legacy_features.items()}
        global_map = [junction * self.local_dim + position
                      for junction, tls in enumerate(self.features) for position in local_maps[tls]]
        self.backlog_dim = (1 if method == "ippo" else len(self.features)) if self.backlog else 0
        for tls in self.features:
            if self.memory:
                self.actors[tls] = expanded_mlp(self.actors[tls], self.local_dim, local_maps[tls])
                cost_map = local_maps[tls] if method == "ippo" else global_map
                self.cost_critics[tls] = expanded_mlp(self.cost_critics[tls],
                    self.local_dim if method == "ippo" else self.global_dim, cost_map)
            if self.memory or self.backlog:
                base = self.local_dim if method == "ippo" else self.global_dim
                old_base = original_local if method == "ippo" else original_global
                mapping = (local_maps[tls] if method == "ippo" else global_map) + [base]
                if len(mapping) != old_base + 1:
                    raise AssertionError("Clock mapping lost a feature")
                self.critics[tls] = expanded_mlp(self.critics[tls], base + 1 + self.backlog_dim, mapping)

    def critic_inputs(self, observation, state, times, horizon, backlog=None):
        inputs = super().critic_inputs(observation, state, times, horizon)
        if self.backlog:
            if backlog is None:
                raise ValueError("Critic backlog is required, never silently zero-imputed")
            values = torch.as_tensor(backlog, dtype=inputs.dtype, device=inputs.device)
            if values.shape != inputs.shape[:-1] + (self.backlog_dim,):
                raise ValueError("Critic backlog shape mismatch")
            if not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid critic backlog")
            inputs = torch.cat((inputs, values), dim=-1)
        return inputs

    def snapshot(self):
        return {"schema": SCHEMA, "method": self.method, "arm": self.arm, "seed": self.seed,
                "backlog": self.backlog, "legacy_features": copy.deepcopy(self.legacy_features),
                "features": copy.deepcopy(self.features), "weights": copy.deepcopy(self.state_dict())}

    @classmethod
    def from_snapshot(cls, saved):
        if saved["schema"] != SCHEMA:
            raise ValueError("Wrong audit checkpoint schema")
        model = cls(saved["method"], saved["legacy_features"], saved["arm"],
                    saved["seed"], saved["backlog"])
        if model.features != saved["features"]:
            raise ValueError("Checkpoint feature order mismatch")
        model.load_state_dict(saved["weights"])
        return model


def critic_backlogs(model, tls, rows, trace):
    """Only already-due requests; no future demand. Counts scaled by ten."""
    if not model.backlog:
        return None
    if model.method == "ippo":
        return [[row["uninserted_due"] / 10.] for row in rows]
    by_time = {}
    for row in trace:
        local = by_time.setdefault(row["time"], {})
        if row["junction"] in local:
            raise ValueError("Duplicate junction/time trace row")
        local[row["junction"]] = row["uninserted_due"]
    try:
        return [[by_time[row["time"]][junction] / 10. for junction in model.features] for row in rows]
    except KeyError as error:
        raise ValueError("Missing synchronized MAPPO backlog") from error


def prepare_trial_batch(model, tls, rollouts, weight=3):
    device = next(model.parameters()).device
    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)
    episodes = []
    for rollout in rollouts:
        rows = [row for row in rollout["trace"] if row["junction"] == tls]
        if len(rows) < 2:
            raise ValueError("Rollout too short")
        obs = tensor([row["observation"] for row in rows[:-1]])
        states = tensor([row["global_state"] for row in rows[:-1]])
        inputs = model.critic_inputs(obs, states, [row["time"] for row in rows[:-1]],
            rollout["horizon"], critic_backlogs(model, tls, rows[:-1], rollout["trace"]))
        actions = tensor([row["action"] if row["action"] is not None else 0 for row in rows[:-1]], torch.long)
        valid = tensor([row["action"] is not None and sum(row["mask"]) > 1 for row in rows[:-1]], torch.bool)
        masks = tensor([row["mask"] if any(row["mask"]) else [True, False, False]
                        for row in rows[:-1]], torch.bool)
        if any(row["action"] is not None and not row["mask"][row["action"]] for row in rows[:-1]):
            raise ValueError("Action violates the collection mask")
        rewards = [-(row["vehicle_queue"] + row["uninserted_due"] + weight * row["pending_count"]) / 10
                   for row in rows[1:]]
        with torch.no_grad():
            values = model.prediction(tls, inputs)
            old_logp = model.distribution(tls, obs, masks).log_prob(actions)
        for i, row in enumerate(rows[:-1]):
            if "behavior_logp" in row and abs(float(old_logp[i]) - row["behavior_logp"]) > 1e-5:
                raise ValueError("Rollout is not on-policy or action mask changed")
        adv, targets = advantages(rewards, values.tolist())
        episodes.append({"observations": obs, "inputs": inputs, "actions": actions, "valid": valid,
            "masks": masks, "old_logp": old_logp, "old_values": values, "advantages": tensor(adv),
            "returns": tensor(targets), "monte_carlo": tensor(raw_return(rewards)), "rewards": tensor(rewards)})
    if not episodes:
        raise ValueError("Empty rollout batch")
    return {key: torch.cat([episode[key] for episode in episodes]) for key in episodes[0]}


def gradient_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None))


def clip_gradients(model, tls):
    """Return preclip norms and clip independent networks only when declared."""
    actor = gradient_norm(model.actors[tls].parameters())
    critic = gradient_norm(model.critics[tls].parameters())
    if model.separate_clipping:
        actor_raw = torch.nn.utils.clip_grad_norm_(model.actors[tls].parameters(), .5)
        critic_raw = torch.nn.utils.clip_grad_norm_(model.critics[tls].parameters(), .5)
        total = (actor_raw.square() + critic_raw.square()).sqrt()
    else:
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite gradient")
    return total, actor, critic


class TrialLearner:
    def __init__(self, model, weight=3):
        self.model, self.weight = model, weight
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    def state(self):
        return {"schema": SCHEMA, "arm": self.model.arm, "backlog": self.model.backlog,
                "weight": self.weight, "optimizer": copy.deepcopy(self.optimizer.state_dict())}

    def restore(self, saved):
        if (saved["schema"], saved["arm"], saved["backlog"], saved["weight"]) != (
                SCHEMA, self.model.arm, self.model.backlog, self.weight):
            raise ValueError("Learner checkpoint treatment mismatch")
        self.optimizer.load_state_dict(saved["optimizer"])

    def update(self, rollouts):
        if not rollouts:
            raise ValueError("Empty rollout batch")
        model = self.model
        batches = {tls: prepare_trial_batch(model, tls, rollouts, self.weight) for tls in model.features}
        results = {}
        for tls, batch in batches.items():
            valid, adv = batch["valid"], batch["advantages"]
            if valid.any():
                adv = (adv - adv[valid].mean()) / adv[valid].std(unbiased=False).clamp_min(1e-6)
            with torch.no_grad():
                before = model.prediction(tls, batch["inputs"]).clone()
            model.scales[tls].update(batch["returns"], model.critics[tls][-1])
            with torch.no_grad():
                after = model.prediction(tls, batch["inputs"])
            if not torch.allclose(before, after, atol=1e-4, rtol=1e-5):
                raise AssertionError("Normalization changed raw value predictions")
            grad_history = []
            for _ in range(4):
                dist = model.distribution(tls, batch["observations"], batch["masks"])
                logp = dist.log_prob(batch["actions"])
                ratio = (logp - batch["old_logp"]).exp()
                surrogate = torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv)
                actor_loss = -surrogate[valid].mean() if valid.any() else logp.sum() * 0
                normalized = model.critics[tls](batch["inputs"]).squeeze(-1)
                value_loss = (normalized - model.scales[tls].normalize(batch["returns"])).square().mean()
                entropy = dist.entropy()[valid].mean() if valid.any() else actor_loss * 0
                loss = actor_loss + .5 * value_loss - .01 * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not valid.any():
                    for parameter in model.actors[tls].parameters():
                        parameter.grad = None
                total, actor_grad, critic_grad = clip_gradients(model, tls)
                grad_history.append({"actor": actor_grad, "critic": critic_grad, "combined": float(total)})
                self.optimizer.step()
            with torch.no_grad():
                value = model.prediction(tls, batch["inputs"])
                dist = model.distribution(tls, batch["observations"], batch["masks"])
                delta = dist.log_prob(batch["actions"]) - batch["old_logp"]
                kl = ((delta.exp() - 1) - delta)[valid].mean() if valid.any() else torch.tensor(0.)
            results[tls] = {"loss": float(loss.detach()), "actor_loss": float(actor_loss.detach()),
                "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                "gradient_norm": float(total), "actor_gradient_norm": actor_grad,
                "critic_gradient_norm": critic_grad, "epoch_gradient_norms": grad_history,
                "approx_kl_after": float(kl), "decisions": int(valid.sum()), "states": len(valid),
                "episodes_in_update": len(rollouts), "reward_mean": float(batch["rewards"].mean()),
                "normalizer_count": float(model.scales[tls].count),
                "normalizer_mean": float(model.scales[tls].mean), "normalizer_scale": float(model.scales[tls].scale),
                "before_fit_mc": prediction_stats(batch["monte_carlo"], batch["old_values"]),
                "after_fit_mc": prediction_stats(batch["monte_carlo"], value),
                "after_fit_gae": prediction_stats(batch["returns"], value)}
        return results
