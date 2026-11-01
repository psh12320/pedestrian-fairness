"""Neural actors and critics for the forthcoming v2 PPO training engine.

This module alone is not a trained controller or a complete PPO implementation.
Actors have no global-state argument. Each junction owns separate parameters.
"""

import torch
from torch import nn
from torch.distributions import Categorical

from . import SCHEMA_VERSION


def mlp(inputs, outputs):
    layers = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64),
                           nn.Tanh(), nn.Linear(64, outputs))
    for module in layers:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=2 ** .5)
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(layers[-1].weight, gain=.01 if outputs == 3 else 1)
    return layers


class JunctionNetworks(nn.Module):
    def __init__(self, method, observation_features, seed=25001):
        super().__init__()
        if method not in ("ippo", "mappo"):
            raise ValueError(method)
        self.method = method
        self.features = {key: list(value) for key, value in sorted(observation_features.items())}
        dimensions = {len(value) for value in self.features.values()}
        if len(dimensions) != 1 or not dimensions or min(dimensions) == 0:
            raise ValueError("Expected nonempty observations of equal size")
        self.local_dim = dimensions.pop()
        self.global_dim = self.local_dim * len(self.features)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actors = nn.ModuleDict({key: mlp(self.local_dim, 3) for key in self.features})
            critic_dim = self.local_dim if method == "ippo" else self.global_dim
            self.critics = nn.ModuleDict({key: mlp(critic_dim, 1) for key in self.features})
            # Four distinct crossing cost estimates at every junction.
            self.cost_critics = nn.ModuleDict({key: mlp(critic_dim, 4) for key in self.features})

    def distribution(self, junction, observation, mask):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                                      device=next(self.parameters()).device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=observation.device)
        if observation.shape[-1] != self.local_dim or mask.shape[-1] != 3:
            raise ValueError("Observation/action schema mismatch")
        if not mask.any(dim=-1).all():
            raise ValueError("No legal decision during this transition")
        logits = self.actors[junction](observation)
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    def values(self, junction, observation, global_state):
        features = observation if self.method == "ippo" else global_state
        features = torch.as_tensor(features, dtype=torch.float32, device=next(self.parameters()).device)
        return self.critics[junction](features).squeeze(-1), self.cost_critics[junction](features)

    @torch.no_grad()
    def act(self, junction, observation, mask, deterministic=False):
        distribution = self.distribution(junction, observation, mask)
        action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item())

    def save(self, path):
        torch.save({"schema_version": SCHEMA_VERSION, "method": self.method,
                    "observation_features": self.features, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, expected_features):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["schema_version"] != SCHEMA_VERSION or data["observation_features"] != expected_features:
            raise ValueError("Checkpoint feature order/version mismatch")
        model = cls(data["method"], expected_features)
        model.load_state_dict(data["state_dict"])
        return model
