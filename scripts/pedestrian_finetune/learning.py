"""Conservative on-policy continuation with optional local pedestrian guidance.

Every arm retains the complete warm-start model, including value moments, and
starts fresh Adam with the same settings. The parent actor is a frozen copy.
Teacher labels are auxiliary supervised targets on *current student* states;
they never replace actions or behavior probabilities in the PPO objective.
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch

from src.traffic_control.fairness.learning import advantages
from src.traffic_control.ppo_repair.learning import prepare_batch
from .teacher import crossing_tail_cost, labelled_rollout

ARMS = ('continue', 'tail', 'guided')
SCHEMA = 'pedestrian-finetune-learner-v1'
DEFAULTS = {
    'learning_rate': 1e-4, 'adam_epsilon': 1e-5, 'ppo_epochs': 4,
    'gamma': .99, 'gae_lambda': .95, 'clip_epsilon': .2,
    'entropy_weight': .01, 'value_weight': .5, 'max_gradient_norm': .5,
    'pedestrian_weight': 3., 'parent_kl_weight': .02,
    'tail_threshold_seconds': 20., 'tail_scale_seconds': 20.,
    'tail_cap': 5., 'tail_weight': 1., 'guidance_weight': .5,
    'urgency_seconds': 20.,
}


def _weights_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(repr((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        # Avoid requiring Torch's optional NumPy bridge (older CPU wheels may
        # coexist with NumPy 2 in the student's existing environment).
        digest.update(bytes(tensor.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def _settings(cfg):
    result = {name: cfg.get(name, default) for name, default in DEFAULTS.items()}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in result.values()):
        raise ValueError('Fine-tuning settings must be finite numbers')
    if not isinstance(result['ppo_epochs'], int) or result['ppo_epochs'] < 1:
        raise ValueError('ppo_epochs must be a positive integer')
    positive = ('learning_rate', 'adam_epsilon', 'value_weight',
                'max_gradient_norm', 'tail_scale_seconds')
    if any(result[k] <= 0 for k in positive) or any(v < 0 for v in result.values()):
        raise ValueError('Invalid negative/zero fine-tuning setting')
    if not (0 <= result['gamma'] <= 1 and 0 <= result['gae_lambda'] <= 1
            and 0 < result['clip_epsilon'] < 1):
        raise ValueError('Invalid GAE or PPO clipping setting')
    return result


def _masked_kl(student, parent, masks):
    """KL(student || parent), avoiding undefined -inf - -inf masked terms.

    Some Torch versions return finite masked categorical KL while its backward
    pass still contains NaN. Mask log probabilities *before* subtraction.
    """
    student_log = student.logits.masked_fill(~masks, 0.)
    parent_log = parent.logits.masked_fill(~masks, 0.)
    return (student.probs * (student_log - parent_log)).sum(-1)


class FineTuneLearner:
    def __init__(self, model, reference, arm, cfg):
        if arm not in ARMS:
            raise ValueError('Unknown fine-tuning arm: ' + str(arm))
        if model is reference:
            raise ValueError('Student and parent must be distinct models')
        if (model.features != reference.features or model.method != reference.method
                or model.variant != reference.variant or not model.repaired):
            raise ValueError('Warm-start student and normalized parent schemas differ')
        self.model, self.arm, self.cfg = model, arm, _settings(cfg)
        # An owned copy also prevents accidental mutation by callers. No values
        # or moments from this model participate in critic fitting.
        self.reference = copy.deepcopy(reference).to(next(model.parameters()).device)
        self.reference.requires_grad_(False)
        self.reference.eval()
        self.reference_sha256 = _weights_hash(self.reference)
        self.optimizer = torch.optim.Adam(
            list(model.actors.parameters()) + list(model.critics.parameters()),
            lr=self.cfg['learning_rate'], eps=self.cfg['adam_epsilon'])
        self.last_labels = None

    def state(self):
        return {'schema': SCHEMA, 'arm': self.arm, 'cfg': copy.deepcopy(self.cfg),
                'reference_sha256': self.reference_sha256,
                'optimizer': copy.deepcopy(self.optimizer.state_dict())}

    def restore(self, saved):
        if (saved.get('schema') != SCHEMA or saved.get('arm') != self.arm
                or saved.get('cfg') != self.cfg
                or saved.get('reference_sha256') != self.reference_sha256):
            raise ValueError('Fine-tuning checkpoint protocol or parent mismatch')
        self.optimizer.load_state_dict(saved['optimizer'])

    def update(self, rollouts, episode_index=0):
        if len(rollouts) != 1:
            raise ValueError('Fine-tuning uses exactly one complete episode per update')
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError('Invalid episode index')
        model, cfg, rollout = self.model, self.cfg, rollouts[0]
        # Validates local/masked decisions and retains *all* free labels, not
        # merely the urgent or successful ones. Terminal and forced rows are out.
        labels = labelled_rollout(rollout['trace'], model.features,
                                 horizon=rollout['horizon'], urgency_seconds=cfg['urgency_seconds'])
        batches, extras = {}, {}
        for tls, names in model.features.items():
            rows = [r for r in rollout['trace'] if r['junction'] == tls]
            if not rows or float(rows[-1]['time']) != float(rollout['horizon']):
                raise ValueError('Incomplete episode: missing terminal observation')
            for row in rows[:-1]:
                if 'behavior_logp' in row and not math.isfinite(row['behavior_logp']):
                    raise ValueError('Nonfinite behavior log probability')
            # This checks stored collection log probabilities before any update.
            batch = prepare_batch(model, tls, rollouts, cfg['pedestrian_weight'])
            if not torch.isfinite(batch['old_logp'][batch['valid']]).all():
                raise ValueError('Nonfinite on-policy log probabilities')
            base = [-(r['vehicle_queue'] + r['uninserted_due'] + cfg['pedestrian_weight'] * r['pending_count']) / 10
                    for r in rows[1:]]
            costs = [crossing_tail_cost(r['observation'], names,
                        threshold_seconds=cfg['tail_threshold_seconds'],
                        scale_seconds=cfg['tail_scale_seconds'], cap=cfg['tail_cap'])
                     for r in rows[1:]]
            tail_weight = cfg['tail_weight'] if self.arm != 'continue' else 0.
            rewards = [reward - tail_weight * cost for reward, cost in zip(base, costs)]
            if not all(math.isfinite(v) for v in rewards):
                raise ValueError('Nonfinite training rewards')
            adv, returns = advantages(rewards, batch['old_values'].tolist(),
                                      gamma=cfg['gamma'], decay=cfg['gae_lambda'])
            batch['advantages'] = batch['old_values'].new_tensor(adv)
            batch['returns'] = batch['old_values'].new_tensor(returns)
            local = labels['data'][tls]
            if len(local['labels']) != int(batch['valid'].sum()):
                raise AssertionError('Teacher/PPO free-decision alignment changed')
            batch['teacher'] = torch.tensor(local['labels'], dtype=torch.long,
                                            device=batch['actions'].device)
            batch['teacher_weights'] = batch['old_values'].new_tensor(local['guidance_weights'])
            with torch.no_grad():
                batch['parent'] = self.reference.distribution(tls, batch['observations'], batch['masks'])
            batches[tls] = batch
            extras[tls] = {'base_reward_mean': sum(base) / len(base),
                           'tail_cost_mean': sum(costs) / len(costs),
                           'tail_weight': tail_weight,
                           'reward_mean': sum(rewards) / len(rewards),
                           'teacher_labels': len(local['labels']),
                           'teacher_selected': int(sum(local['guidance_weights']))}

        self.last_labels = labels
        results = {}
        guidance = cfg['guidance_weight'] if self.arm == 'guided' else 0.
        for tls, batch in batches.items():
            valid, adv = batch['valid'], batch['advantages']
            has_choices = bool(valid.any())
            if has_choices:
                adv = (adv - adv[valid].mean()) / adv[valid].std(unbiased=False).clamp_min(1e-6)
            with torch.no_grad():
                before = model.prediction(tls, batch['inputs']).clone()
            # Retain historical moments and append current targets. Resetting
            # moments alone would change the meaning of normalized critic output.
            model.scales[tls].update(batch['returns'], model.critics[tls][-1])
            with torch.no_grad():
                after = model.prediction(tls, batch['inputs'])
            normalizer_error = float((after - before).abs().max())
            if not torch.allclose(before, after, atol=1e-4, rtol=1e-5):
                raise AssertionError('Normalization changed raw value predictions')
            epochs = []
            for _ in range(cfg['ppo_epochs']):
                dist = model.distribution(tls, batch['observations'], batch['masks'])
                logp = dist.log_prob(batch['actions'])
                # Use a finite graph-connected zero: forced transitions can have
                # -inf logp for the placeholder action under a singleton mask.
                zero = dist.probs.sum() * 0
                entropy = dist.entropy()[valid].mean() if has_choices else zero
                if has_choices:
                    ratio = (logp[valid] - batch['old_logp'][valid]).exp()
                    clipped = ratio.clamp(1-cfg['clip_epsilon'], 1+cfg['clip_epsilon'])
                    actor_loss = -torch.minimum(ratio * adv[valid], clipped * adv[valid]).mean()
                    free_dist = torch.distributions.Categorical(logits=dist.logits[valid])
                    ce = -free_dist.log_prob(batch['teacher'])
                    teacher_loss = (ce * batch['teacher_weights']).sum() / valid.sum()
                    parent_kl = _masked_kl(dist, batch['parent'], batch['masks'])[valid].mean()
                else:
                    actor_loss, teacher_loss, parent_kl = zero, zero, zero
                normalized = model.critics[tls](batch['inputs']).squeeze(-1)
                value_loss = (normalized - model.scales[tls].normalize(batch['returns'])).square().mean()
                loss = (actor_loss + cfg['value_weight'] * value_loss - cfg['entropy_weight'] * entropy
                        + guidance * teacher_loss + cfg['parent_kl_weight'] * parent_kl)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite fine-tuning loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not has_choices:
                    # A zero gradient still permits Adam momentum to drift.
                    for parameter in model.actors[tls].parameters():
                        parameter.grad = None
                actor_norm = torch.nn.utils.clip_grad_norm_(model.actors[tls].parameters(), cfg['max_gradient_norm'])
                critic_norm = torch.nn.utils.clip_grad_norm_(model.critics[tls].parameters(), cfg['max_gradient_norm'])
                if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):
                    raise FloatingPointError('Nonfinite fine-tuning gradient')
                self.optimizer.step()
                epochs.append({'loss': float(loss.detach()), 'actor_loss': float(actor_loss.detach()),
                               'value_loss': float(value_loss.detach()), 'entropy': float(entropy.detach()),
                               'teacher_loss': float(teacher_loss.detach()),
                               'parent_kl_loss': float(parent_kl.detach()),
                               'actor_gradient_norm': float(actor_norm), 'critic_gradient_norm': float(critic_norm)})
            with torch.no_grad():
                dist = model.distribution(tls, batch['observations'], batch['masks'])
                if has_choices:
                    delta = dist.log_prob(batch['actions'])[valid] - batch['old_logp'][valid]
                    approx_kl = float(((delta.exp()-1)-delta).mean())
                    parent_kl_after = float(_masked_kl(dist, batch['parent'], batch['masks'])[valid].mean())
                else:
                    approx_kl, parent_kl_after = 0., 0.
            results[tls] = {**extras[tls], **{key: sum(e[key] for e in epochs)/len(epochs) for key in epochs[0]},
                            'decisions': int(valid.sum()), 'states': len(valid),
                            'guidance_weight': guidance, 'parent_kl_weight': cfg['parent_kl_weight'],
                            'parent_kl_direction': 'student_to_frozen_parent',
                            'approx_kl_after': approx_kl, 'parent_kl_after': parent_kl_after,
                            'normalizer_count': float(model.scales[tls].count),
                            'normalizer_mean': float(model.scales[tls].mean),
                            'normalizer_scale': float(model.scales[tls].scale),
                            'normalizer_raw_prediction_max_error': normalizer_error}
        return results
