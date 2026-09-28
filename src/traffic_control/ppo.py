"""Small, dependency-free PPO models for IPPO and central-critic MAPPO.

Each junction has its own linear Bernoulli actor taking only local features.
IPPO has one local linear value estimator per junction. MAPPO has one shared
value estimator taking the concatenated observations of all junctions.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def gae(rewards: list[float], values: list[float], gamma: float, decay: float) -> list[float]:
    advantages = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * decay * running
        advantages[index] = running
    return advantages


class PPOModel:
    def __init__(self, method: str, objective: str, junction_ids: list[str],
                 hyperparameters: dict, seed: int):
        if method not in ("ippo", "mappo") or objective not in ("vehicle", "pedestrian"):
            raise ValueError((method, objective))
        self.method = method
        self.objective = objective
        self.junction_ids = sorted(junction_ids)
        self.hyperparameters = hyperparameters
        self.rng = random.Random(seed)
        self.obs_dim = 9
        self.global_dim = self.obs_dim * len(junction_ids)
        self.actors = {tls: [self.rng.uniform(-0.02, 0.02) for _ in range(self.obs_dim)]
                       for tls in self.junction_ids}
        self.critics = ({tls: [0.0] * self.obs_dim for tls in self.junction_ids}
                        if method == "ippo" else {"global": [0.0] * self.global_dim})
        self.episodes_trained = 0

    def select(self, tls: str, obs: list[float], global_obs: list[float],
               eligible: bool, training: bool = True) -> tuple[int, float, float]:
        if len(obs) != self.obs_dim or (training and len(global_obs) != self.global_dim):
            raise ValueError("Unexpected observation size")
        probability = min(1 - 1e-8, max(1e-8, sigmoid(dot(self.actors[tls], obs))))
        action = int(self.rng.random() < probability) if training and eligible else int(
            probability >= 0.5 and eligible)
        logp = math.log(probability if action else 1 - probability) if eligible else 0.0
        value = 0.0 if not training else (dot(self.critics[tls], obs) if self.method == "ippo"
                                          else dot(self.critics["global"], global_obs))
        return action, logp, value

    def update(self, traces: dict[str, list[dict]]) -> dict:
        hp = self.hyperparameters
        advantages: dict[str, list[float]] = {}
        targets: dict[str, list[float]] = {}
        length = min(len(trace) for trace in traces.values())
        if length == 0:
            raise ValueError("Empty PPO rollout")
        if self.method == "mappo":
            shared_rewards = [sum(traces[tls][i]["reward"] for tls in self.junction_ids)
                              for i in range(length)]
            shared_values = [traces[self.junction_ids[0]][i]["value"] for i in range(length)]
            shared_adv = gae(shared_rewards, shared_values, hp["gamma"], hp["gae_lambda"])
            for tls in self.junction_ids:
                advantages[tls] = shared_adv
            targets["global"] = [value + adv for value, adv in zip(shared_values, shared_adv)]
        else:
            for tls in self.junction_ids:
                trace = traces[tls][:length]
                values = [row["value"] for row in trace]
                adv = gae([row["reward"] for row in trace], values,
                          hp["gamma"], hp["gae_lambda"])
                advantages[tls] = adv
                targets[tls] = [value + advantage for value, advantage in zip(values, adv)]

        for _ in range(hp["update_epochs"]):
            for tls in self.junction_ids:
                rows = traces[tls][:length]
                valid = [i for i, row in enumerate(rows) if row["actor_mask"]]
                if not valid:
                    continue
                selected_adv = [advantages[tls][i] for i in valid]
                mean = statistics.mean(selected_adv)
                scale = statistics.pstdev(selected_adv) or 1.0
                gradient = [0.0] * self.obs_dim
                for i in valid:
                    row = rows[i]
                    obs = row["obs"]
                    probability = min(1 - 1e-8, max(1e-8, sigmoid(dot(self.actors[tls], obs))))
                    logp = math.log(probability if row["action"] else 1 - probability)
                    ratio = math.exp(max(-20.0, min(20.0, logp - row["old_logp"])))
                    advantage = (advantages[tls][i] - mean) / scale
                    unclipped = ((advantage >= 0 and ratio <= 1 + hp["clip_ratio"])
                                 or (advantage < 0 and ratio >= 1 - hp["clip_ratio"]))
                    coefficient = (ratio * advantage * (row["action"] - probability)
                                   if unclipped else 0.0)
                    entropy_gradient = (probability * (1 - probability)
                                        * math.log((1 - probability) / probability))
                    coefficient += hp["entropy_coefficient"] * entropy_gradient
                    for j, feature in enumerate(obs):
                        gradient[j] += coefficient * feature
                for j in range(self.obs_dim):
                    self.actors[tls][j] = max(-10.0, min(10.0, self.actors[tls][j]
                        + hp["actor_learning_rate"] * gradient[j] / len(valid)))

            critic_keys = self.junction_ids if self.method == "ippo" else ["global"]
            for key in critic_keys:
                rows = traces[key][:length] if self.method == "ippo" else traces[self.junction_ids[0]][:length]
                weights = self.critics[key]
                gradient = [0.0] * len(weights)
                for i, row in enumerate(rows):
                    features = row["obs"] if self.method == "ippo" else row["global_obs"]
                    error = targets[key][i] - dot(weights, features)
                    for j, feature in enumerate(features):
                        gradient[j] += error * feature
                for j in range(len(weights)):
                    weights[j] = max(-100.0, min(100.0, weights[j]
                        + hp["critic_learning_rate"] * gradient[j] / length))
        self.episodes_trained += 1
        return {"samples_per_agent": length,
                "decisions_per_agent": {tls: sum(row["actor_mask"] for row in traces[tls])
                                        for tls in self.junction_ids},
                "mean_reward": {tls: statistics.mean(row["reward"] for row in traces[tls][:length])
                                for tls in self.junction_ids}}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {"method": self.method, "objective": self.objective,
                   "junction_ids": self.junction_ids, "hyperparameters": self.hyperparameters,
                   "actors": self.actors, "critics": self.critics,
                   "episodes_trained": self.episodes_trained}
        path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PPOModel":
        content = json.loads(path.read_text(encoding="utf-8"))
        model = cls(content["method"], content["objective"], content["junction_ids"],
                    content["hyperparameters"], 0)
        model.actors = content["actors"]
        model.critics = content["critics"]
        model.episodes_trained = content["episodes_trained"]
        return model
