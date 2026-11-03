"""Read-only implementation probes; never a SUMO performance experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.traffic_control.fairness.signals import SafeSignal, PHASES
from src.traffic_control.ppo_repair.learning import RepairNetworks, RepairLearner


def visible_signal(signal, now):
    """Legacy default signal-state fields in environment.py (traffic fixed)."""
    stages = (*PHASES, 'EW_YELLOW', 'NS_YELLOW', 'ALL_RED', 'PED_CLEARANCE', 'READY')
    return [signal.age(now) / 30, *[float(signal.stage == p) for p in stages],
            *map(float, signal.mask(now, (True, True, True)))]


def transition_alias():
    cfg = json.loads((ROOT / 'config/fairness_v2.json').read_text())
    plan = SimpleNamespace(config=cfg, clearance=20)
    to_ns, to_ped = SafeSignal(plan), SafeSignal(plan)
    to_ns.request(12, 1, (True, True, True))
    to_ped.request(12, 2, (True, True, True))
    same_yellow = visible_signal(to_ns, 13) == visible_signal(to_ped, 13)
    to_ns.advance(15, False); to_ped.advance(15, False)
    same_red = visible_signal(to_ns, 16) == visible_signal(to_ped, 16)
    to_ns.advance(17, False); to_ped.advance(17, False)
    # Clearance also depends on the past duration of emptiness, absent from obs.
    cleared_a, cleared_b = SafeSignal(plan), SafeSignal(plan)
    for s in (cleared_a, cleared_b): s.enter('PED_CLEARANCE', 0)
    cleared_a.empty_since, cleared_b.empty_since = 17, 19.8
    same_clearance = visible_signal(cleared_a, 20) == visible_signal(cleared_b, 20)
    cleared_a.advance(20, False); cleared_b.advance(20, False)
    # If two such states have targets -10 and -30, the best deterministic
    # current-input predictor has irreducible MSE 100 on this equal-weight pair.
    return {'yellow_visible_equal': same_yellow, 'all_red_visible_equal': same_red,
            'next_stages': [to_ns.stage, to_ped.stage],
            'clearance_visible_equal': same_clearance,
            'next_clearance_stages': [cleared_a.stage, cleared_b.stage],
            'illustrative_aliased_target_pair': [-10, -30], 'pair_minimum_mse': 100,
            'note': 'MSE is a constructed counterexample, not measured traffic critic error.'}


def positive_control(method, variant, updates=64, samples=192, seed=93001, separate_clip=False):
    """Three contextual bandits with known optimal actions and local-only actors.

    Every one-second choice has its cost in the next row, matching production
    rollout alignment. Context is random and observed. No simulator/old artifacts.
    """
    features = {t: ['bias', 'context0', 'context1', 'context2'] for t in ('C0', 'C1', 'C2')}
    model = RepairNetworks(method, features, variant, seed)
    learner = RepairLearner(model)
    rng = random.Random(seed)
    original_clip = torch.nn.utils.clip_grad_norm_
    gradient_rows = []
    def norm(parameters):
        return sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None) ** .5
    def clip(parameters, max_norm, *args, **kwargs):
        actor, critic = norm(model.actors.parameters()), norm(model.critics.parameters())
        if separate_clip:
            total = torch.tensor((actor**2 + critic**2)**.5)
            original_clip(model.actors.parameters(), max_norm, *args, **kwargs)
            original_clip(model.critics.parameters(), max_norm, *args, **kwargs)
        else:
            total = original_clip(parameters, max_norm, *args, **kwargs)
        gradient_rows.append({'actor': actor, 'critic': critic, 'total': float(total),
                              'clip_factor': min(1., max_norm/((actor if separate_clip else float(total))+1e-6))})
        return total
    def accuracy():
        with torch.no_grad():
            probabilities = []
            for j, t in enumerate(features):
                for context in range(3):
                    obs = [1., *[float(i == context) for i in range(3)]]
                    probabilities.append(float(model.distribution(t, obs, [True]*3).probs[(j+context)%3]))
        return statistics.mean(probabilities)
    initial = accuracy()
    for episode in range(updates):
        trace = []; costs = dict.fromkeys(features, 0.)
        for time in range(1, samples + 2):
            contexts = {t: rng.randrange(3) for t in features}
            obs = {t: [1., *[float(i == contexts[t]) for i in range(3)]] for t in features}
            state = sum(obs.values(), [])
            with torch.no_grad():
                for j, t in enumerate(features):
                    distribution = model.distribution(t, obs[t], [True]*3)
                    action = rng.choices(range(3), weights=distribution.probs.tolist())[0]
                    trace.append({'junction': t, 'time': time, 'observation': obs[t],
                        'global_state': state, 'mask': [True]*3, 'action': action,
                        'behavior_logp': float(distribution.log_prob(torch.tensor(action))),
                        'vehicle_queue': costs[t], 'uninserted_due': 0, 'pending_count': 0})
                    costs[t] = 0. if action == (j+contexts[t])%3 else 10.
        with patch('torch.nn.utils.clip_grad_norm_', clip):
            learner.update([{'trace': trace, 'horizon': samples+1}])
    return {'method': method, 'variant': variant, 'separate_clip': separate_clip,
            'updates': updates, 'samples_per_update_per_agent': samples,
            'initial_optimal_action_probability': initial, 'final_optimal_action_probability': accuracy(),
            'mean_actor_gradient_norm': statistics.mean(g['actor'] for g in gradient_rows),
            'mean_critic_gradient_norm': statistics.mean(g['critic'] for g in gradient_rows),
            'mean_actor_clip_factor': statistics.mean(g['clip_factor'] for g in gradient_rows),
            'note': 'Positive software control only; does not establish SUMO controller performance.'}


def historical_update_statistics():
    archive = ROOT / 'runs/ppo-repair-results.zip'
    if not archive.exists(): return []
    rows = []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('learning.json')):
            data = json.loads(z.read(name)); history = data['history']
            recent = [s for h in history[-4:] for s in h['update'].values()]
            rows.append({'label': data['label'], 'update_batches': len(history),
                         'free_decisions_per_episode_last4_updates': statistics.mean(s['decisions']/s['episodes_in_update'] for s in recent),
                         'entropy_last4_updates': statistics.mean(s['entropy'] for s in recent),
                         'approx_kl_last4_updates': statistics.mean(s['approx_kl_after'] for s in recent),
                         'gradient_norm_last4_updates': statistics.mean(s['gradient_norm'] for s in recent)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/deep-audit-rl.json')
    parser.add_argument('--updates', type=int, default=64)
    parser.add_argument('--clip-ablation', action='store_true', help='Also test separate actor/critic clipping on the toy task')
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {'scope': 'diagnostics only; original source/checkpoints unchanged',
        'transition_alias': transition_alias(),
        'gae_residual_weights': {str(n): (.99*.95)**n for n in (5,12,20,30,40,60,90)},
        'discount_weights': {str(n): .99**n for n in (5,12,20,30,40,60,90)},
        'historical_update_statistics': historical_update_statistics(), 'positive_controls': []}
    for method in ('ippo', 'mappo'):
        for variant in ('original', 'normalized-clock'):
            row = positive_control(method, variant, args.updates)
            result['positive_controls'].append(row)
            print(json.dumps(row), flush=True)
    if args.clip_ablation:
        for method in ('ippo', 'mappo'):
            row = positive_control(method, 'normalized-clock', args.updates, separate_clip=True)
            result['positive_controls'].append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Evidence:', args.output, flush=True)


if __name__ == '__main__': main()
