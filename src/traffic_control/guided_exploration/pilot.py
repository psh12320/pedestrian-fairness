"""Resumable, seed-locked development study of guidance and exploration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import time
import torch
from ..scenario import ROOT
from ..fairness.pipeline import atomic_json
from ..ppo_repair.learning import RepairNetworks
from ..ppo_repair.pilot import (episode as old_episode, make_protocol as repair_protocol,
    cached_json, atomic_torch, digest, TRAIN_SCENARIOS, EVAL_SCENARIOS, BASELINES)
from ..ppo_repair.reporting import comparison
from . import SCHEMA
from .learning import StudyLearner, demonstrations, imitate, teacher_action, rewards_for


def labels(protocol):
    return [(m, v, r, f'{m}-{v}-r{r}') for m in protocol['methods']
            for v in protocol['variants'] for r in range(protocol['replicas'])]


def make_protocol(quick=False, methods=None):
    cfg = json.loads((ROOT / 'config/guided_exploration.json').read_text())
    # Reuse only version discovery/source hashing, not earlier demand or outputs.
    previous = repair_protocol(quick)
    if methods is not None: cfg['methods'] = methods
    if not cfg['methods'] or any(m not in ('ippo', 'mappo') for m in cfg['methods']):
        raise ValueError('Supported study methods are ippo and mappo')
    if quick:
        cfg.update(replicas=1, episodes=4, validation_episodes=[4],
                   demonstration_episodes=4, imitation_steps=60, guidance_decay_episodes=4)
    protocol = {**cfg, 'quick': quick, 'status': 'exploratory development study',
        'initial_seeds': list(range(85001, 85001 + cfg['replicas'])),
        'demonstration_seeds': list(range(860001, 860001 + cfg['demonstration_episodes'])),
        'training_seed_base': 870001, 'training_scenarios': list(TRAIN_SCENARIOS),
        'validation_seeds': [880001] if quick else [880001, 880002],
        'evaluation_seeds': [890001] if quick else list(range(890001, 890007)),
        'evaluation_scenarios': ['vmedium-pmedium'] if quick else list(EVAL_SCENARIOS),
        'baselines': list(BASELINES), 'reserved_final_seeds_not_used': list(range(58001, 58021)),
        'selection': 'Final budget only; no model or replica selection; teacher fidelity is diagnostic',
        'reward': 'base -(vehicle_queue+uninserted_due+3*pending)/10; age arms add bounded quadratic oldest age per crossing',
        'imitation': 'Actuated demonstrations, actor-only pretraining and decaying online teacher cross entropy',
        'rnd': 'Local observations; fixed random target; train-only moments, bonus and predictor; cumulative-mean error scaling',
        'safety': 'Unchanged fairness SafeSignal and phase plan for every controller',
        'python': platform.python_version(), 'torch': str(torch.__version__), 'sumo': previous['sumo'],
        'source': previous['source']}
    for p in sorted((ROOT / 'scripts').glob('*guided_exploration*')):
        protocol['source'][p.relative_to(ROOT).as_posix()] = digest(p)
    seed_sets = [set(protocol['demonstration_seeds']),
        {protocol['training_seed_base'] + r * 1000 + e for r in range(cfg['replicas']) for e in range(cfg['episodes'])},
        set(protocol['validation_seeds']), set(protocol['evaluation_seeds']), set(protocol['reserved_final_seeds_not_used'])]
    if any(a & b for i, a in enumerate(seed_sets) for b in seed_sets[i+1:]): raise ValueError('Overlapping seeds')
    protocol['fingerprint'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def episode(root, protocol, scenario, seed, controller, model=None, training=False):
    outcome = old_episode(root, protocol, scenario, seed, controller, model, training)
    # The old probe targets base reward; it is invalid for age/RND value targets.
    for tls, d in outcome['diagnostics'].items():
        d.pop('critic_probe', None)
        free = [r for r in outcome['rollout']['trace'] if r['junction'] == tls and
                r['action'] is not None and sum(r['mask']) > 1]
        if model and free:
            target = [teacher_action(tls, r['observation'], model.features[tls], r['mask']) for r in free]
            with torch.no_grad():
                dist = model.distribution(tls, [r['observation'] for r in free], [r['mask'] for r in free])
                d['teacher_probability'] = float(dist.log_prob(torch.tensor(target)).exp().mean())
                d['teacher_greedy_agreement'] = float((dist.probs.argmax(-1) == torch.tensor(target)).float().mean())
            d['sampled_teacher_agreement'] = sum(r['action'] == a for r, a in zip(free, target)) / len(free)
    return outcome


def compact(outcome):
    return {k: v for k, v in outcome.items() if k != 'rollout'}


def reference(root, protocol, scenario, seed, outcome=None):
    data = cached_json(root / 'baselines' / f'{scenario}-{seed}.json', protocol['fingerprint'])
    if not data: raise ValueError('Missing paired baseline')
    ref = next(r for r in data['records'] if r['controller'] == 'actuated')
    if outcome and (ref['result']['demand_sha256'] != outcome['result']['demand_sha256'] or
                    ref['network_sha256'] != outcome['network_sha256']):
        raise ValueError('Demand/network pairing failed')
    return ref


def baseline_job(args):
    folder, protocol, scenario, seed = args; root = Path(folder)
    path = root / 'baselines' / f'{scenario}-{seed}.json'
    if cached_json(path, protocol['fingerprint']): return str(path)
    torch.set_num_threads(1)
    records = [{'controller': c, **compact(episode(root, protocol, scenario, seed, c))} for c in BASELINES]
    if len({r['result']['demand_sha256'] for r in records}) != 1 or len({r['network_sha256'] for r in records}) != 1:
        raise ValueError('Unmatched baseline runs')
    atomic_json(path, {'fingerprint': protocol['fingerprint'], 'scenario': scenario, 'seed': seed, 'records': records})
    return str(path)


def prepare_demonstrations(root, protocol):
    path = root / 'teacher' / 'demonstrations.pt'; manifest = root / 'teacher' / 'demonstrations.json'
    old = cached_json(manifest, protocol['fingerprint'])
    if old:
        if not path.exists() or digest(path) != old['data_sha256']: raise ValueError('Demonstrations changed')
        return
    first = reference(root, protocol, 'vmedium-pmedium', protocol['validation_seeds'][0])
    model = RepairNetworks('mappo', first['features'], 'normalized-clock', protocol['initial_seeds'][0])
    data = {t: {'observations': [], 'masks': [], 'labels': []} for t in model.features}; records = []
    for i, seed in enumerate(protocol['demonstration_seeds']):
        scenario = protocol['training_scenarios'][i % len(protocol['training_scenarios'])]
        out = episode(root, protocol, scenario, seed, 'actuated', training=True)
        local = demonstrations(model, [out])
        for t in data:
            for k in data[t]: data[t][k].extend(local[t][k])
        records.append({'scenario': scenario, 'seed': seed, 'seconds': out['rollout']['horizon'],
                        'demand_sha256': out['result']['demand_sha256'], 'network_sha256': out['network_sha256']})
        print('Teacher demonstration', i+1, flush=True)
    atomic_torch(path, {'fingerprint': protocol['fingerprint'], 'data': data, 'features': model.features})
    atomic_json(manifest, {'fingerprint': protocol['fingerprint'], 'records': records,
        'data_sha256': digest(path), 'free_decisions': {t: len(d['labels']) for t, d in data.items()}})


def teacher_job(args):
    folder, protocol, replica = args; root = Path(folder); torch.set_num_threads(1)
    path = root / 'teacher' / f'final-r{replica}.pt'; log = path.with_suffix('.json')
    old = cached_json(log, protocol['fingerprint'])
    if old:
        if digest(path) != old['checkpoint_sha256']: raise ValueError('Imitation checkpoint changed')
        return replica
    saved = torch.load(root / 'teacher/demonstrations.pt', map_location='cpu', weights_only=True)
    if saved['fingerprint'] != protocol['fingerprint']: raise ValueError('Wrong demonstrations')
    seed = protocol['initial_seeds'][replica]
    model = RepairNetworks('mappo', saved['features'], 'normalized-clock', seed)
    stats = imitate(model, saved['data'], protocol['imitation_steps'], protocol['imitation_batch_size'], seed)
    records = []; refs = []
    for demand in protocol['validation_seeds']:
        out = episode(root, protocol, 'vmedium-pmedium', demand, 'policy', model)
        refs.append(reference(root, protocol, 'vmedium-pmedium', demand, out)['result'])
        records.append({'scenario': 'vmedium-pmedium', 'seed': demand, **compact(out)})
    fidelity = comparison([[o['result'] for o in records]], [refs], draws=0)
    # Exact primary-criterion comparison is visible, not a selection/filtering gate.
    atomic_torch(path, {'fingerprint': protocol['fingerprint'], 'model': model.snapshot(), 'replica': replica})
    atomic_json(log, {'fingerprint': protocol['fingerprint'], 'replica': replica,
        'checkpoint_sha256': digest(path), 'training': stats, 'validation': records, 'fidelity_vs_actuated': fidelity})
    return replica


def train_job(args):
    folder, protocol, method, variant, replica, label = args; root = Path(folder)
    directory = root / 'models' / label; directory.mkdir(parents=True, exist_ok=True)
    fp = protocol['fingerprint']; done = cached_json(directory / 'complete.json', fp)
    if done:
        if digest(directory / 'final.pt') != done['checkpoint_sha256']: raise ValueError('Final model changed')
        return label
    torch.set_num_threads(1); seed = protocol['initial_seeds'][replica]; torch.manual_seed(seed)
    features = reference(root, protocol, 'vmedium-pmedium', protocol['validation_seeds'][0])['features']
    model = RepairNetworks(method, features, 'normalized-clock', seed)
    learner = StudyLearner(model, variant, protocol, seed)
    latest = directory / 'latest.pt'; completed = 0; history = []; validations = []
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved['fingerprint'] != fp or saved['label'] != label: raise ValueError('Wrong resume checkpoint')
        model.load_state_dict(saved['model']['weights']); learner.restore(saved['learner'])
        torch.set_rng_state(saved['rng']); completed = saved['episodes']; history = saved['history']; validations = saved['validations']
    elif variant == 'guided':
        teacher = torch.load(root / 'teacher' / f'final-r{replica}.pt', map_location='cpu', weights_only=True)
        if teacher['fingerprint'] != fp: raise ValueError('Wrong imitation source')
        copied = RepairNetworks.from_snapshot(teacher['model'])
        model.actors.load_state_dict(copied.actors.state_dict())
    while completed < protocol['episodes']:
        start = time.perf_counter(); i = completed
        scenario = protocol['training_scenarios'][i % len(protocol['training_scenarios'])]
        demand = protocol['training_seed_base'] + replica * 1000 + i
        out = episode(root, protocol, scenario, demand, 'policy', model, training=True)
        update = learner.update([out['rollout']], i); completed += 1
        history.append({'episodes': completed, 'scenario': scenario, 'seed': demand,
            'demand_sha256': out['result']['demand_sha256'], 'network_sha256': out['network_sha256'],
            'vehicles': out['result']['vehicles'], 'diagnostics': out['diagnostics'], 'update': update,
            'seconds': time.perf_counter()-start})
        if completed in protocol['validation_episodes']:
            for demand in protocol['validation_seeds']:
                out = episode(root, protocol, 'vmedium-pmedium', demand, 'policy', model)
                reference(root, protocol, 'vmedium-pmedium', demand, out)
                validations.append({'episodes': completed, 'scenario': 'vmedium-pmedium', 'seed': demand, **compact(out)})
        atomic_torch(latest, {'fingerprint': fp, 'label': label, 'episodes': completed,
            'model': model.snapshot(), 'learner': learner.state(), 'rng': torch.get_rng_state(),
            'history': history, 'validations': validations})
        atomic_json(directory / 'learning.json', {'fingerprint': fp, 'label': label, 'history': history, 'validations': validations})
        print(label, 'episodes', completed, flush=True)
    atomic_json(directory / 'learning.json', {'fingerprint': fp, 'label': label, 'history': history, 'validations': validations})
    atomic_torch(directory / 'final.pt', {'fingerprint': fp, 'label': label, 'episodes': completed, 'model': model.snapshot()})
    atomic_json(directory / 'complete.json', {'fingerprint': fp, 'label': label, 'episodes': completed,
        'update_batches': len(history), 'checkpoint_sha256': digest(directory / 'final.pt')})
    return label


def evaluation_job(args):
    folder, protocol, label, scenario, seed = args; root = Path(folder); torch.set_num_threads(1)
    checkpoint = (root / 'teacher' / f"final-r{label.split('-r')[-1]}.pt" if label.startswith('imitation-')
                  else root / 'models' / label / 'final.pt')
    sha = digest(checkpoint); path = root / 'evaluation' / f'{label}-{scenario}-{seed}.json'
    old = cached_json(path, protocol['fingerprint'])
    if old:
        if old['checkpoint_sha256'] != sha: raise ValueError('Changed evaluated model')
        return str(path)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['fingerprint'] != protocol['fingerprint']: raise ValueError('Wrong evaluation checkpoint')
    if not label.startswith('imitation-') and saved['episodes'] != protocol['episodes']: raise ValueError('Unfinished training')
    model = RepairNetworks.from_snapshot(saved['model']); model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = episode(root, protocol, scenario, seed, 'policy', model)
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()): raise AssertionError('Evaluation changed model')
    reference(root, protocol, scenario, seed, out)
    atomic_json(path, {'fingerprint': protocol['fingerprint'], 'label': label, 'scenario': scenario,
                      'seed': seed, 'checkpoint_sha256': sha, **compact(out)})
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/guided-exploration'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=('ippo', 'mappo'))
    parser.add_argument('--stage', choices=('all', 'baselines', 'teacher', 'train', 'evaluate', 'report'), default='all')
    args = parser.parse_args()
    if args.workers < 1: parser.error('Positive worker count required')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol(args.quick, args.methods)
    cached_json(root / 'protocol.json', protocol['fingerprint'])
    atomic_json(root / 'protocol.json', protocol)
    for d in ('baselines', 'teacher', 'models', 'evaluation'): (root / d).mkdir(exist_ok=True)
    def parallel(function, jobs):
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(function, jobs): print('Completed', result, flush=True)
    if args.stage in ('all', 'baselines'):
        settings = sorted({(s, d) for s in protocol['evaluation_scenarios'] for d in protocol['evaluation_seeds']} |
                          {('vmedium-pmedium', d) for d in protocol['validation_seeds']})
        parallel(baseline_job, [(str(root), protocol, s, d) for s, d in settings])
    if args.stage in ('all', 'teacher'):
        prepare_demonstrations(root, protocol)
        parallel(teacher_job, [(str(root), protocol, r) for r in range(protocol['replicas'])])
    if args.stage in ('all', 'train'):
        parallel(train_job, [(str(root), protocol, *j) for j in labels(protocol)])
    if args.stage in ('all', 'evaluate'):
        names = [j[-1] for j in labels(protocol)] + [f'imitation-r{r}' for r in range(protocol['replicas'])]
        parallel(evaluation_job, [(str(root), protocol, l, s, d) for l in names
            for s in protocol['evaluation_scenarios'] for d in protocol['evaluation_seeds']])
    if args.stage in ('all', 'report'):
        from .reporting import report
        report(root, protocol)


if __name__ == '__main__': main()
