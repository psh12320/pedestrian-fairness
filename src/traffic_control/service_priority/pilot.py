"""Controlled exploratory pilot. No final-test seeds are used here.

Train/evaluate protected and unprotected PPO with identical budgets. A service
rule is not a guarantee of crossing-level p95 non-inferiority.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import statistics
import tempfile

from ..fairness.analysis import summarise, selection_score, quantile
from ..fairness.demand import scenarios
from ..fairness.environment import run
from ..fairness.pipeline import atomic_json
from ..scenario import ROOT
from .control import ProtectedPolicy, service_components

SCENARIOS = ('vmedium-pmedium', 'vmedium-plow', 'pedestrian-burst', 'short-spacing')
BASELINES = ('actuated', 'pressure', 'priority-pressure', 'age-pressure')
METHODS = ('ippo', 'mappo')


def episode(root, kind, seed, cfg, features=None, model=None, train=False, protected=True):
    trace = []
    wrapper = None
    if kind in ('priority-pressure', 'age-pressure') or model is not None and protected:
        wrapper = ProtectedPolicy(features, model, not train, 20. if kind == 'age-pressure' else 0.)
    callback = wrapper if wrapper else (
        (lambda tls, obs, state, mask: model.act(tls, obs, mask, not train)[0]) if model else None)
    with tempfile.TemporaryDirectory(prefix='service-', dir=root) as folder:
        metrics = run('policy' if callback else kind, seed, folder, overrides=cfg,
                      policy=callback, trace_sink=trace)
        events = json.loads((Path(folder) / 'events.json').read_text())
    if wrapper:
        wrapper.align_training_trace(trace)
    phases = {tls: {s: sum(b[0] - a[0] for a, b in zip(history, history[1:] + [(cfg.get('end_seconds', 2400), '')]) if a[1] == s)
                    for s in {h[1] for h in history}} for tls, history in metrics['phase_history'].items()}
    diagnostics = {'phase_seconds': phases, 'policy_calls': wrapper.calls if wrapper else 0,
        'restricted_calls': wrapper.restricted if wrapper else 0,
        'forced_pedestrian_calls': wrapper.forced_pedestrian if wrapper else 0,
        'actor_decisions': sum(r['action'] is not None and sum(r['mask']) > 1 for r in trace),
        'max_pending_age_seconds': max((r['observation'][i] * 60 for r in trace
            for i, n in enumerate(metrics['observation_features'][r['junction']]) if n.endswith('/oldest')), default=0)}
    return summarise(metrics, events), metrics['observation_features'], trace, events, diagnostics


def cfg_for(protocol, scenario, training=False):
    cfg = dict(scenarios()[scenario])
    if protocol['quick']:
        cfg.update(end_seconds=180, warmup_seconds=0, measurement_end_seconds=120,
                   demand_end_seconds=90, burst_start_seconds=40)
    elif training:
        cfg.update(end_seconds=600, warmup_seconds=0, measurement_end_seconds=600, demand_end_seconds=600)
    return cfg


def baseline_job(args):
    root, protocol, scenario, seed = args
    root = Path(root)
    path = root / f'baseline-{scenario}-{seed}.json'
    if path.exists():
        return json.loads(path.read_text())
    records, features, demand_hash = [], None, None
    for kind in BASELINES:
        result, features, _, _, diag = episode(root, kind, seed, cfg_for(protocol, scenario), features)
        if demand_hash is not None and result['demand_sha256'] != demand_hash:
            raise AssertionError('Baseline demands differ')
        demand_hash = result['demand_sha256']
        records.append({'controller': kind, 'result': result, 'diagnostics': diag})
    data = {'scenario': scenario, 'seed': seed, 'features': features, 'records': records}
    atomic_json(path, data)
    return data


def train_job(args):
    import torch
    from ..fairness.networks import JunctionNetworks
    from ..fairness.learning import Learner
    root, protocol, method, protected, replica = args
    root = Path(root)
    label = f'{"priority" if protected else "plain"}-{method}-r{replica}'
    directory = root / label
    directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return label
    torch.set_num_threads(1)
    init_seed = 46001 + replica
    torch.manual_seed(init_seed)
    first = json.loads(next(root.glob('baseline-*.json')).read_text())
    features = first['features']
    model = JunctionNetworks(method, features, init_seed)
    learner = Learner(model, weight=3)
    checkpoint = directory / 'latest.pt'
    start, best = 0, (float('inf'),) * 3
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(saved['weights']); learner.optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']); start, best = saved['episode'], tuple(saved['best'])
    for i in range(start, protocol['episodes']):
        scenario = protocol['scenarios'][i % len(protocol['scenarios'])]
        # Same episodes/demand and initialisation across protected/unprotected arms.
        seed = 60001 + replica * 10000 + i
        result, _, trace, events, diag = episode(root, 'policy', seed, cfg_for(protocol, scenario, True),
                                                features, model, True, protected)
        update = learner.update(trace, events)
        if (i + 1) % protocol['validate_every'] == 0 or i + 1 == protocol['episodes']:
            scores = []
            evaluation = []
            for path in sorted(root.glob('baseline-*.json')):
                baseline = json.loads(path.read_text())
                if baseline['scenario'] not in ('vmedium-pmedium', 'vmedium-plow'):
                    continue
                outcome, _, _, _, d = episode(root, 'policy', baseline['seed'], cfg_for(protocol, baseline['scenario']),
                                               features, model, protected=protected)
                ref = baseline['records'][0]['result']
                scores.append(selection_score(outcome, ref))
                evaluation.append({'scenario': baseline['scenario'], 'seed': baseline['seed'],
                                   'result': outcome, 'diagnostics': d})
            score = tuple(statistics.mean(s[j] for s in scores) for j in range(3))
            atomic_json(directory / f'validation-{i+1}.json', evaluation)
            if score < best:
                best = score
                model.save(directory / 'best.tmp'); (directory / 'best.tmp').replace(directory / 'best.pt')
                atomic_json(directory / 'selected.json', {'episode': i+1, 'score': best})
        torch.save({'episode': i+1, 'weights': model.state_dict(), 'optimizer': learner.optimizer.state_dict(),
                    'rng': torch.get_rng_state(), 'best': best}, directory / 'latest.tmp')
        (directory / 'latest.tmp').replace(checkpoint)
        # A crash after checkpoint save must not lead to duplicate rollout updates.
        atomic_json(directory / f'episode-{i+1}.json', {'episode': i+1, 'update': update, 'diagnostics': diag})
    selected = JunctionNetworks.load(directory / 'best.pt', features)
    evaluation = []
    for path in sorted(root.glob('baseline-*.json')):
        b = json.loads(path.read_text())
        result, _, _, _, diag = episode(root, 'policy', b['seed'], cfg_for(protocol, b['scenario']),
                                        features, selected, protected=protected)
        assert result['demand_sha256'] == b['records'][0]['result']['demand_sha256']
        evaluation.append({'scenario': b['scenario'], 'seed': b['seed'], 'controller': label,
                           'result': result, 'diagnostics': diag})
    atomic_json(directory / 'evaluation.json', evaluation)
    atomic_json(directory / 'complete.json', {'episodes': protocol['episodes']})
    return label


def report(root):
    rows = []
    baselines = {}
    for path in sorted(root.glob('baseline-*.json')):
        group = json.loads(path.read_text())
        baselines[group['scenario'], group['seed']] = group['records'][0]['result']
        rows.extend({**r, 'scenario': group['scenario'], 'seed': group['seed']} for r in group['records'])
    for path in sorted(root.glob('*/evaluation.json')):
        rows.extend(json.loads(path.read_text()))
    pairs = []
    for row in rows:
        result, b = row['result'], baselines[row['scenario'], row['seed']]
        loss, ref = result['vehicles']['completed_mean_time_loss_seconds'], b['vehicles']['completed_mean_time_loss_seconds']
        deltas = {c: v['p95'] - b['crossings'][c]['p95'] if v['p95'] is not None and b['crossings'][c]['p95'] is not None else None
                  for c, v in result['crossings'].items()}
        vehicle_delta = loss - ref if loss is not None and ref is not None else None
        service = selection_score(result, b)[0]
        pairs.append({'controller': row['controller'], 'scenario': row['scenario'], 'seed': row['seed'],
            'vehicle_difference': vehicle_delta, 'crossing_p95_differences': deltas,
            'service_gate_score': service, 'components': service_components(result, b),
            'diagnostics': row['diagnostics'],
            'point_pass': vehicle_delta is not None and vehicle_delta < 0 and service == 0 and
                          all(x is not None and x <= 0 for x in deltas.values())})
    atomic_json(root / 'paired_diagnostics.json', pairs)
    lines = ['# Service priority exploratory pilot', '',
             'Validation/development data only. No held-out superiority claim; no confidence intervals from a smoke run.', '',
             '| Scenario | Controller | Seed | Vehicle delta s | Service gate | Point pass |', '|---|---|---:|---:|---:|---|']
    for r in pairs:
        lines.append(f"| {r['scenario']} | {r['controller']} | {r['seed']} | {r['vehicle_difference']} | {r['service_gate_score']} | {r['point_pass']} |")
    (root / 'REPORT.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--stage', choices=('baselines', 'train', 'report', 'all'), default='all')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--replicas', type=int, default=3)
    args = p.parse_args()
    if min(args.workers, args.episodes, args.replicas) < 1:
        p.error('Workers, episodes and replicas must be positive')
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    protocol = {'quick': args.quick, 'episodes': 2 if args.quick else args.episodes,
        'replicas': 1 if args.quick else args.replicas, 'validate_every': 8,
        'scenarios': list(SCENARIOS), 'seeds': [45001] if args.quick else [47001, 47002],
        'status': 'exploratory; all evaluations are validation, not final tests',
        'reserved_test_seeds_not_used': list(range(58001,58021)),
        'source': {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted([*(ROOT/'src/traffic_control').rglob('*.py'), *(ROOT/'config').glob('*.json')])},
        'python': platform.python_version()}
    path = root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Source/protocol changed; use a new output directory')
    atomic_json(path, protocol)
    if args.stage in ('baselines', 'all'):
        with ProcessPoolExecutor(args.workers) as pool:
            for row in pool.map(baseline_job, [(str(root), protocol, s, seed) for s in SCENARIOS for seed in protocol['seeds']]):
                print('Baseline', row['scenario'], row['seed'], flush=True)
    if args.stage in ('train', 'all'):
        expected = len(SCENARIOS) * len(protocol['seeds'])
        if len(list(root.glob('baseline-*.json'))) != expected:
            raise ValueError('Complete baselines first')
        with ProcessPoolExecutor(args.workers) as pool:
            for label in pool.map(train_job, [(str(root), protocol, m, protect, r) for m in METHODS
                                               for protect in (False, True) for r in range(protocol['replicas'])]):
                print('Trained', label, flush=True)
    report(root)


if __name__ == '__main__':
    main()
