"""Small, frozen-trajectory critic regression diagnostic (not PPO training).

Raw-unit output with scaled loss versus a normalised output, identical inputs,
architecture, initial weights and Adam update budgets. Normalisation statistics
come exclusively from training trajectories. Complete demand seeds are held out.
All fits start afresh; optional saved actors only generate frozen-policy data.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import statistics as st
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from src.traffic_control.fairness.networks import mlp, JunctionNetworks
from src.traffic_control.fairness.pipeline import atomic_json
from src.traffic_control.fairness.environment import run
from src.traffic_control.fairness.analysis import summarise
from src.traffic_control.service_priority.control import service_mask


class TargetScale(nn.Module):
    """Fixed training-set transform; buffers round-trip with the value model."""
    def __init__(self, mean=0., scale=1.):
        super().__init__()
        self.register_buffer('mean', torch.tensor(float(mean)))
        self.register_buffer('scale', torch.tensor(float(scale)))

    @classmethod
    def fit(cls, training_targets):
        if not training_targets.numel() or not torch.isfinite(training_targets).all():
            raise ValueError('Expected finite, nonempty training targets')
        return cls(training_targets.mean(), training_targets.std(unbiased=False).clamp_min(1.))

    def normalize(self, raw):
        return (raw - self.mean) / self.scale

    def denormalize(self, normalized):
        return normalized * self.scale + self.mean


class FixedCritic(nn.Module):
    def __init__(self, inputs, mode, seed, transform):
        super().__init__()
        if mode not in ('raw-scaled-loss', 'normalized-output'):
            raise ValueError(mode)
        self.mode = mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = mlp(inputs, 1)
        self.transform = transform

    def forward(self, observations):
        output = self.network(observations).squeeze(-1)
        return self.transform.denormalize(output) if self.mode == 'normalized-output' else output

    def loss(self, observations, raw_targets):
        output = self.network(observations).squeeze(-1)
        error = (output - self.transform.normalize(raw_targets) if self.mode == 'normalized-output'
                 else (output - raw_targets) / self.transform.scale)
        return .5 * error.square().mean()


def discounted_returns(rewards, gamma=.99):
    following = 0.
    result = []
    for reward in reversed(rewards):
        following = float(reward) + gamma * following
        result.append(following)
    return list(reversed(result))


def prediction_metrics(target, prediction, training_mean):
    if not target.numel():
        return None
    error = prediction - target
    mse = float(error.square().mean())
    constant_mse = float((target - training_mean).square().mean())
    variance = float(target.var(unbiased=False))
    return {'n': target.numel(), 'rmse': mse ** .5, 'bias': float(error.mean()),
            'prediction_mean': float(prediction.mean()), 'target_mean': float(target.mean()),
            'target_sd': variance ** .5, 'constant_rmse': constant_mse ** .5,
            'mse_ratio_to_training_mean': mse / constant_mse if constant_mse > 1e-12 else None,
            'explained_variance': 1 - float(error.var(unbiased=False)) / variance if variance > 1e-12 else None}


def fit_pair(training, checks, input_dim, seed, budgets=(128, 512)):
    """Checks is never used to update weights, statistics or choose a checkpoint."""
    torch.set_num_threads(1)
    x, targets = training
    results = []
    for mode in ('raw-scaled-loss', 'normalized-output'):
        scale = TargetScale.fit(targets)
        model = FixedCritic(input_dim, mode, seed, scale)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        for step in range(max(budgets) + 1):
            if step in (0, *budgets):
                with torch.no_grad():
                    assessments = {name: prediction_metrics(y, model(z), scale.mean) for name, (z, y) in checks.items()}
                    assessments['training'] = prediction_metrics(targets, model(x), scale.mean)
                results.append({'mode': mode, 'updates': step, 'assessments': assessments,
                                'training_target_mean': float(scale.mean), 'training_target_scale': float(scale.scale)})
            if step == max(budgets): break
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, targets)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite value loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.network.parameters(), .5)
            if not torch.isfinite(gradient): raise FloatingPointError('Non-finite critic gradient')
            optimizer.step()
    return results


def toy_check():
    x = torch.linspace(-1, 1, 129).unsqueeze(1)
    h = torch.linspace(-.99, .99, 128).unsqueeze(1)
    return fit_pair((x, -300 + 40*x[:, 0]), {'heldout': (h, -300 + 40*h[:, 0])}, 1, 4246)


def collect(args):
    output_string, protocol, seed = args
    output = Path(output_string)
    path = output / f'trajectory-{seed}.pt'
    if path.exists(): return str(path)
    torch.set_num_threads(1)
    model = None
    callback = None
    if protocol['behavior'] != 'actuated':
        pilot = Path(protocol['pilot'])
        features = json.loads(next(pilot.glob('baseline-*.json')).read_text())['features']
        model = JunctionNetworks.load(pilot/protocol['behavior']/'best.pt', features)
        rngs = {tls: random.Random(700000 + seed + i*1000003) for i, tls in enumerate(sorted(features))}
        def callback(tls, obs, state, physical):
            mask = service_mask(obs, features[tls], physical) if protocol['behavior'].startswith('priority-') else physical
            with torch.no_grad(): probs = model.distribution(tls, obs, mask).probs.tolist()
            legal = [i for i, yes in enumerate(mask) if yes]
            return rngs[tls].choices(legal, weights=[probs[i] for i in legal])[0]
    trace = []
    with tempfile.TemporaryDirectory(prefix='collect-', dir=output) as folder:
        metrics = run('actuated' if callback is None else 'policy', seed, folder,
                      overrides=protocol['configuration'], policy=callback, trace_sink=trace)
        events = json.loads((Path(folder)/'events.json').read_text())
    features = metrics['observation_features']
    packed = {'seed': seed, 'fingerprint': protocol['fingerprint'], 'features': features,
              'result': summarise(metrics, events), 'sumo_version': metrics['sumo_version'], 'junctions': {}}
    for tls in sorted(features):
        local = [r for r in trace if r['junction'] == tls]
        rewards = [-(r['vehicle_queue'] + r['uninserted_due'] + 3*r['pending_count'])/10 for r in local[1:]]
        packed['junctions'][tls] = {
            'observations': torch.tensor([r['observation'] for r in local[:-1]], dtype=torch.float32),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'returns': torch.tensor(discounted_returns(rewards), dtype=torch.float32),
            'times': torch.tensor([r['time'] for r in local[:-1]], dtype=torch.float32),
            'free_decision': torch.tensor([r['action'] is not None and sum(r['mask']) > 1 for r in local[:-1]], dtype=torch.bool)}
        if protocol['behavior'].startswith('priority-'):
            packed['junctions'][tls]['free_decision'] = torch.tensor([
                r['action'] is not None and sum(service_mask(r['observation'], features[tls], r['mask'])) > 1
                for r in local[:-1]], dtype=torch.bool)
    temporary = path.with_suffix('.tmp')
    torch.save(packed, temporary); temporary.replace(path)
    return str(path)


def data_for(rows, tls, method, subset='all'):
    xs, ys = [], []
    for data in rows:
        local = data['junctions'][tls]
        x = local['observations'] if method == 'ippo' else torch.cat([
            data['junctions'][other]['observations'] for other in sorted(data['junctions'])], dim=1)
        mask = torch.ones(len(x), dtype=torch.bool)
        if subset == 'interior': mask = (local['times'] >= 150) & (local['times'] < 400)
        elif subset == 'free_decisions': mask = local['free_decision']
        elif subset != 'all': raise ValueError(subset)
        xs.append(x[mask]); ys.append(local['returns'][mask])
    return torch.cat(xs), torch.cat(ys)


def fit_job(args):
    output_string, protocol, method, initial_seed = args
    output = Path(output_string)
    path = output/f'fit-{method}-{initial_seed}.json'
    if path.exists(): return str(path)
    torch.set_num_threads(1)
    train = [torch.load(output/f'trajectory-{seed}.pt', weights_only=True) for seed in protocol['training_demand_seeds']]
    held = [torch.load(output/f'trajectory-{seed}.pt', weights_only=True) for seed in protocol['check_demand_seeds']]
    assert all(r['fingerprint'] == protocol['fingerprint'] for r in train+held)
    results = []
    started = time.perf_counter()
    for index, tls in enumerate(sorted(train[0]['junctions'])):
        training = data_for(train, tls, method)
        checks = {f'seed-{r["seed"]}': data_for([r], tls, method) for r in held}
        checks.update(heldout=data_for(held, tls, method),
                      interior=data_for(held, tls, method, 'interior'),
                      free_decisions=data_for(held, tls, method, 'free_decisions'))
        results.extend(dict(r, junction=tls) for r in fit_pair(training, checks, training[0].shape[1],
                                                              initial_seed+index*100003, protocol['budgets']))
    atomic_json(path, {'fingerprint': protocol['fingerprint'], 'method': method,
                       'initial_seed': initial_seed, 'seconds': time.perf_counter()-started, 'results': results})
    return str(path)


def report(output, protocol):
    rows = [json.loads((output/f'fit-{method}-{seed}.json').read_text()) for method in ('ippo', 'mappo')
            for seed in protocol['initial_seeds']]
    summary = []
    for method in ('ippo', 'mappo'):
        for mode in ('raw-scaled-loss', 'normalized-output'):
            for budget in protocol['budgets']:
                selected = [r for group in rows if group['method'] == method for r in group['results']
                            if r['mode'] == mode and r['updates'] == budget]
                summary.append({'method': method, 'mode': mode, 'updates': budget,
                    'mean_holdout_mse_ratio': st.mean(r['assessments']['heldout']['mse_ratio_to_training_mean'] for r in selected),
                    'mean_holdout_bias': st.mean(r['assessments']['heldout']['bias'] for r in selected),
                    'mean_holdout_ev': st.mean(r['assessments']['heldout']['explained_variance'] for r in selected),
                    'mean_train_mse_ratio': st.mean(r['assessments']['training']['mse_ratio_to_training_mean'] for r in selected),
                    'mean_interior_mse_ratio': st.mean(r['assessments']['interior']['mse_ratio_to_training_mean'] for r in selected),
                    'mean_free_decision_mse_ratio': st.mean(r['assessments']['free_decisions']['mse_ratio_to_training_mean'] for r in selected)})
    per_initialization = []
    for group in rows:
        rs = [r for r in group['results'] if r['mode'] == 'normalized-output' and r['updates'] == max(protocol['budgets'])]
        per_initialization.append({'method': group['method'], 'seed': group['initial_seed'],
            'mean_holdout_mse_ratio': st.mean(r['assessments']['heldout']['mse_ratio_to_training_mean'] for r in rs)})
    gate = all(r['mean_holdout_mse_ratio'] < 1 for r in per_initialization)
    atomic_json(output/'summary.json', {'protocol': protocol, 'summary': summary,
        'per_initialization': per_initialization, 'normalized_beats_constant_each_initialization': gate})
    lines = ['# Critic-fitting diagnostic', '',
        f"Behavior policy: {protocol['behavior']}. Six 600-second medium-demand trajectories; four complete demand seeds for fitting and two separate seeds for checking. Three new critic initialisations per observation setting. Fixed targets, no actor training and no traffic-policy improvement claim.", '',
        'Both fits use the same 64x64 tanh architecture, initial network tensors, full-batch Adam (3e-4, epsilon 1e-5), gradient limit 0.5 and update counts. Raw output uses the existing scaled-error formula; the other output predicts standardised returns and is converted back to raw units for every metric. The normaliser sees training targets only. This isolates output/target scaling on fixed data; it does not reproduce PPO bootstrap updates.', '',
        '| Critic inputs | Output | Updates | Holdout MSE / constant | Holdout bias | Explained variance | Training MSE / constant | Interior MSE / constant | Free-choice MSE / constant |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['method']} | {r['mode']} | {r['updates']} | {r['mean_holdout_mse_ratio']:.3f} | {r['mean_holdout_bias']:.2f} | {r['mean_holdout_ev']:.3f} | {r['mean_train_mse_ratio']:.3f} | {r['mean_interior_mse_ratio']:.3f} | {r['mean_free_decision_mse_ratio']:.3f} |")
    lines += ['', 'Ratios below 1 beat a constant training-mean predictor on the same check data. Bias is prediction minus target in reward units, not seconds. Rows average three junctions and three initialisations; these are descriptive values without confidence intervals.', '',
        f'Predeclared gate (normalised fit beats the constant predictor, averaging junctions, in every method/initialisation at 512 updates): **{gate}**.', '',
        'The interior slice (150-399 seconds) and free-choice slice diagnose boundary/forced-state effects; they were specified before fitting and are not alternative success criteria. The finite-horizon targets depend on remaining time, which is absent from the current observation. Stochastic arrivals also limit predictability. Fixed-policy supervised fits do not establish PPO convergence, comparative IPPO/MAPPO traffic performance, or every-crossing fairness. The true final test seeds remain untouched.', '',
        'Read toy.json and fit-*.json for the known-answer test, individual junctions, separate check seeds and initialisation variation. Neither heldout loss nor these check seeds select model checkpoints; both fixed budgets are reported.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    archive = output.parent/(output.name+'-results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted([*output.glob('*.json'), output/'REPORT.md']): z.write(path, path.name)
    print('RESULTS', archive, 'gate', gate, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/critic-fit-actuated'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--model', default='actuated')
    parser.add_argument('--stage', choices=('all', 'collect', 'fit', 'report'), default='all')
    args = parser.parse_args()
    if args.workers < 1: parser.error('workers must be positive')
    if (args.model != 'actuated') != bool(args.pilot): parser.error('Pass --pilot and a saved --model together')
    output = args.output.resolve()
    if args.pilot and (output == args.pilot.resolve() or args.pilot.resolve() in output.parents):
        parser.error('Keep new output separate from the pilot')
    protocol = {'schema': 'critic-fit-v1', 'behavior': args.model,
        'pilot': str(args.pilot.resolve()) if args.pilot else None,
        'training_demand_seeds': [49101, 49102, 49103, 49104],
        'check_demand_seeds': [49201, 49202], 'initial_seeds': [49301, 49302, 49303],
        'budgets': [128, 512], 'gamma': .99, 'reward_weight': 3,
        'configuration': {'end_seconds': 600, 'warmup_seconds': 0, 'measurement_end_seconds': 600,
                          'demand_end_seconds': 600, 'vehicle_rate_per_hour': 1920, 'pedestrian_rate_per_hour': 1200},
        'gate': 'Every method/init averaged over junctions: normalized 512-step holdout MSE < training-mean baseline MSE',
        'source': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted([* (ROOT/'src/traffic_control').rglob('*.py'), *(ROOT/'config').glob('*.json'), Path(__file__)])}}
    if args.pilot:
        prior = json.loads((args.pilot/'protocol.json').read_text())
        if prior['quick']: parser.error('Use full pilot checkpoints; engineering checkpoints are not this diagnostic')
        for path, sha in prior['source'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != sha: raise ValueError('Pilot source changed: '+path)
        protocol['checkpoint_sha256'] = hashlib.sha256((args.pilot/args.model/'best.pt').read_bytes()).hexdigest()
    protocol['fingerprint'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Diagnostic changed; use a fresh output directory')
    atomic_json(manifest, protocol)
    started = time.perf_counter()
    if args.stage in ('all', 'collect'):
        with ProcessPoolExecutor(args.workers) as pool:
            jobs = [(str(output), protocol, seed) for seed in protocol['training_demand_seeds']+protocol['check_demand_seeds']]
            for path in pool.map(collect, jobs): print('Collected', Path(path).name, flush=True)
    if args.stage in ('all', 'fit'):
        atomic_json(output/'toy.json', toy_check())
        with ProcessPoolExecutor(args.workers) as pool:
            jobs = [(str(output), protocol, method, seed) for method in ('ippo', 'mappo') for seed in protocol['initial_seeds']]
            for path in pool.map(fit_job, jobs): print('Fitted', Path(path).name, flush=True)
    if args.stage in ('all', 'fit', 'report'): report(output, protocol)
    print(f'Elapsed {time.perf_counter()-started:.1f} seconds', flush=True)


if __name__ == '__main__':
    main()
