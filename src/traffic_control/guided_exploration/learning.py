"""Local teacher labels, dense crossing-age cost, and training-only RND.

All actions still pass through the unchanged shared physical safety machine.
The control arm is numerically checked against the previous repaired learner.
"""
import torch
from torch import nn
from ..connected import incoming_groups
from ..fairness.learning import advantages
from ..fairness.networks import mlp
from ..ppo_repair.learning import prepare_batch, RepairLearner

VARIANTS = ('control', 'age', 'guided', 'rnd')


def teacher_action(tls, observation, names, mask):
    """Exact local-observation transcription of baseline_action('actuated')."""
    legal = [i for i, value in enumerate(mask) if value]
    if not legal:
        raise ValueError('Teacher called outside a decision')
    values = dict(zip(names, observation))
    # Recover integer detector counts before tie-breaking: .5+.7 and .4+.8
    # are equal vehicle queues but unequal binary floating-point sums.
    queues = [sum(round(values[e + '/queue'] * 10) for e in incoming_groups(3)[tls][axis])
              for axis in ('EW', 'NS')]
    pending = sum(v for n, v in values.items() if n.startswith(':') and n.endswith('/count'))
    if pending > 0 and mask[2]:
        desired = 2
    elif values['EW'] or values['NS']:
        current = 0 if values['EW'] else 1
        desired = current if queues[current] > 0 else 1 - current
    else:
        desired = max((0, 1), key=lambda i: (queues[i], -i))
    return desired if desired in legal else legal[0]


def rewards_for(rows, names, variant, cfg):
    result = []
    for row in rows:
        reward = -(row['vehicle_queue'] + row['uninserted_due'] + 3 * row['pending_count']) / 10
        if variant != 'control':
            values = dict(zip(names, row['observation']))
            cost = 0.
            for n, age in values.items():
                if n.startswith(':') and n.endswith('/oldest') and values[n[:-7] + '/count'] > 0:
                    units = max(0., age * 60 / cfg['age_scale_seconds'])
                    cost += min(units, cfg['age_cap_units']) ** 2
            reward -= cfg['age_weight'] * cost
        result.append(reward)
    return result


class Moments(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer('count', torch.zeros((), dtype=torch.float64))
        self.register_buffer('mean', torch.zeros(size, dtype=torch.float64))
        self.register_buffer('m2', torch.zeros(size, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        x = x.detach().double()
        if not len(x) or not torch.isfinite(x).all():
            raise ValueError('Invalid RND samples')
        n = len(x); delta = x.mean(0) - self.mean; total = self.count + n
        self.m2.add_(((x - x.mean(0)) ** 2).sum(0) + delta.square() * self.count * n / total)
        self.mean.add_(delta * n / total); self.count.copy_(total)

    def normalize(self, x):
        variance = self.m2 / self.count.clamp_min(1)
        return ((x - self.mean.to(x)) / variance.clamp_min(.01).sqrt().to(x)).clamp(-5, 5)


class Novelty(nn.Module):
    def __init__(self, features, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 900000)
            self.target = nn.ModuleDict({t: mlp(len(n), 32) for t, n in features.items()})
            self.predictor = nn.ModuleDict({t: mlp(len(n), 32) for t, n in features.items()})
        self.target.requires_grad_(False)
        self.inputs = nn.ModuleDict({t: Moments(len(n)) for t, n in features.items()})
        self.errors = nn.ModuleDict({t: Moments(1) for t in features})
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

    @torch.no_grad()
    def training_bonus(self, tls, observations, cap):
        x = torch.as_tensor(observations, dtype=torch.float32)
        self.inputs[tls].update(x)
        x = self.inputs[tls].normalize(x)
        error = (self.predictor[tls](x) - self.target[tls](x)).square().mean(-1)
        self.errors[tls].update(error[:, None])
        # Nonnegative novelty divided by cumulative mean error; no eval updates.
        bonus = (error / self.errors[tls].mean.to(error).clamp_min(1e-8)).clamp(0, cap)
        return bonus, x

    def fit(self, data):
        losses = {}
        for tls, x in data.items():
            with torch.no_grad(): target = self.target[tls](x)
            for _ in range(4):
                loss = (self.predictor[tls](x) - target).square().mean()
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), .5)
                self.optimizer.step()
            losses[tls] = float(loss.detach())
        return losses


def demonstrations(model, outcomes):
    data = {t: {'observations': [], 'masks': [], 'labels': []} for t in model.features}
    for outcome in outcomes:
        for row in outcome['rollout']['trace']:
            if row['action'] is None:
                continue
            tls = row['junction']
            label = teacher_action(tls, row['observation'], model.features[tls], row['mask'])
            if label != row['action']:
                raise AssertionError('Local teacher disagrees with actual actuated trace')
            if sum(row['mask']) > 1:
                for key, value in [('observations', row['observation']), ('masks', row['mask']), ('labels', label)]:
                    data[tls][key].append(value)
    if any(not d['labels'] for d in data.values()):
        raise ValueError('No free teacher decisions at a junction')
    return data


def imitate(model, data, steps, batch_size, seed):
    """Only actor weights change; demonstration generation cost is reported."""
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    stats = {}
    for tls, d in data.items():
        x = torch.tensor(d['observations'], dtype=torch.float32)
        mask = torch.tensor(d['masks'], dtype=torch.bool)
        labels = torch.tensor(d['labels'], dtype=torch.long)
        for _ in range(steps):
            ids = torch.randint(len(labels), (min(batch_size, len(labels)),), generator=generator)
            loss = -model.distribution(tls, x[ids], mask[ids]).log_prob(labels[ids]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actors.parameters(), 1.)
            optimizer.step()
        with torch.no_grad():
            dist = model.distribution(tls, x, mask)
            stats[tls] = {'samples': len(labels), 'steps': steps,
                          'training_accuracy': float((dist.probs.argmax(-1) == labels).float().mean()),
                          'training_teacher_probability': float(dist.log_prob(labels).exp().mean())}
    return stats


class StudyLearner(RepairLearner):
    def __init__(self, model, variant, cfg, seed):
        super().__init__(model)
        if variant not in VARIANTS: raise ValueError(variant)
        self.variant, self.cfg = variant, cfg
        self.novelty = Novelty(model.features, seed) if variant == 'rnd' else None

    def state(self):
        return {'optimizer': self.optimizer.state_dict(),
                'novelty': self.novelty.state_dict() if self.novelty else None,
                'novelty_optimizer': self.novelty.optimizer.state_dict() if self.novelty else None}

    def restore(self, saved):
        self.optimizer.load_state_dict(saved['optimizer'])
        if bool(saved['novelty'] is not None) != bool(self.novelty is not None):
            raise ValueError('Wrong exploration checkpoint')
        if self.novelty:
            self.novelty.load_state_dict(saved['novelty'])
            self.novelty.optimizer.load_state_dict(saved['novelty_optimizer'])

    def update(self, rollouts, episode_index=0):
        if len(rollouts) != 1: raise ValueError('This study uses one episode per update')
        model = self.model; rollout = rollouts[0]; cfg = self.cfg
        batches = {}; rnd_inputs = {}; extras = {}
        for tls, names in model.features.items():
            b = prepare_batch(model, tls, rollouts)
            rows = [r for r in rollout['trace'] if r['junction'] == tls]
            raw_rewards = rewards_for(rows[1:], names, self.variant, cfg)
            rewards = torch.tensor(raw_rewards)
            extrinsic = rewards.clone()
            bonus = torch.zeros_like(rewards)
            if self.novelty:
                bonus, rnd_inputs[tls] = self.novelty.training_bonus(tls,
                    [r['observation'] for r in rows[1:]], cfg['rnd_bonus_cap'])
                rewards = rewards + cfg['rnd_weight'] * bonus
                raw_rewards = rewards.tolist()
            adv, targets = advantages(raw_rewards, b['old_values'].tolist())
            b['advantages'] = torch.tensor(adv); b['returns'] = torch.tensor(targets)
            b['teacher'] = torch.tensor([teacher_action(tls, r['observation'], names, r['mask'])
                if r['action'] is not None else 0 for r in rows[:-1]])
            batches[tls] = b
            extras[tls] = {'extrinsic_reward_mean': float(extrinsic.mean()),
                           'intrinsic_bonus_mean': float(bonus.mean())}
        guidance = (cfg['guidance_weight'] * max(0., 1. - episode_index / cfg['guidance_decay_episodes'])
                    if self.variant == 'guided' else 0.)
        results = {}
        for tls, b in batches.items():
            valid = b['valid']; adv = b['advantages']
            if valid.any(): adv = (adv - adv[valid].mean()) / adv[valid].std(unbiased=False).clamp_min(1e-6)
            with torch.no_grad(): before = model.prediction(tls, b['inputs']).clone()
            model.scales[tls].update(b['returns'], model.critics[tls][-1])
            with torch.no_grad(): after = model.prediction(tls, b['inputs'])
            if not torch.allclose(before, after, atol=1e-4, rtol=1e-5):
                raise AssertionError('Value normalisation changed raw outputs')
            for _ in range(4):
                dist = model.distribution(tls, b['observations'], b['masks'])
                logp = dist.log_prob(b['actions']); ratio = (logp - b['old_logp']).exp()
                surrogate = torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv)
                actor = -surrogate[valid].mean() if valid.any() else logp.sum() * 0
                entropy = dist.entropy()[valid].mean() if valid.any() else actor * 0
                teacher_loss = -dist.log_prob(b['teacher'])[valid].mean() if valid.any() else actor * 0
                normalized = model.critics[tls](b['inputs']).squeeze(-1)
                value_loss = (normalized - model.scales[tls].normalize(b['returns'])).square().mean()
                loss = actor + .5 * value_loss - .01 * entropy + guidance * teacher_loss
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                if not valid.any():
                    for p in model.actors[tls].parameters(): p.grad = None
                def norm(parameters):
                    return sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None) ** .5
                actor_grad = norm(model.actors[tls].parameters()); critic_grad = norm(model.critics[tls].parameters())
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                if not torch.isfinite(gradient): raise FloatingPointError('Nonfinite gradient')
                self.optimizer.step()
            with torch.no_grad():
                dist = model.distribution(tls, b['observations'], b['masks'])
                delta = dist.log_prob(b['actions']) - b['old_logp']
                kl = float(((delta.exp() - 1) - delta)[valid].mean()) if valid.any() else 0.
            results[tls] = {**extras[tls], 'loss': float(loss.detach()), 'decisions': int(valid.sum()),
                'states': len(valid), 'entropy': float(entropy.detach()), 'approx_kl': kl,
                'teacher_loss': float(teacher_loss.detach()), 'guidance_weight': guidance,
                'actor_gradient_norm': actor_grad, 'critic_gradient_norm': critic_grad,
                'gradient_norm': float(gradient), 'normalizer_count': float(model.scales[tls].count)}
        if self.novelty:
            for tls, loss in self.novelty.fit(rnd_inputs).items():
                results[tls]['rnd_predictor_loss'] = loss
        return results
