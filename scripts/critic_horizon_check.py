"""Follow-up: isolate a finite-horizon critic's missing remaining-time input.

Triggered after the first fitting check failed its gate. Use two new check demand
seeds, the same four training trajectories, equal budgets, and zero-initialised
extra input weights. No actor or signal controller is trained or changed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.critic_fit_check import TargetScale, FixedCritic, prediction_metrics, data_for, collect
from src.traffic_control.fairness.pipeline import atomic_json


def with_clock(rows, tls, method, enabled):
    x, y = data_for(rows, tls, method)
    clock = torch.cat([(600-r['junctions'][tls]['times'])/600 for r in rows])
    return torch.cat((x, (clock if enabled else torch.zeros_like(clock)).unsqueeze(1)), dim=1), y


def clock_model(inputs, seed, targets):
    model = FixedCritic(inputs, 'normalized-output', seed, TargetScale.fit(targets))
    # Both arms have identical initial predictions; only learning can use the clock.
    with torch.no_grad(): model.network[0].weight[:, -1].zero_()
    return model


def fit(args):
    parent_string, output_string, protocol, method, seed = args
    parent, output = Path(parent_string), Path(output_string)
    path = output/f'fit-{method}-{seed}.json'
    if path.exists(): return str(path)
    torch.set_num_threads(1)
    training = [torch.load(parent/f'trajectory-{s}.pt', weights_only=True) for s in protocol['training_demand_seeds']]
    checks = [torch.load(output/f'trajectory-{s}.pt', weights_only=True) for s in protocol['check_demand_seeds']]
    assert all(r['fingerprint'] == protocol['fingerprint'] for r in checks)
    results = []
    for index, tls in enumerate(sorted(training[0]['junctions'])):
        for enabled in (False, True):
            x, y = with_clock(training, tls, method, enabled)
            test_x, test_y = with_clock(checks, tls, method, enabled)
            model = clock_model(x.shape[1], seed+index*100003, y)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
            for step in range(513):
                if step in (0, 128, 512):
                    with torch.no_grad():
                        assessments = {'training': prediction_metrics(y, model(x), model.transform.mean),
                                       'heldout': prediction_metrics(test_y, model(test_x), model.transform.mean)}
                        for r in checks:
                            z, t = with_clock([r], tls, method, enabled)
                            assessments[f'seed-{r["seed"]}'] = prediction_metrics(t, model(z), model.transform.mean)
                    results.append({'junction': tls, 'clock': enabled, 'updates': step, 'assessments': assessments})
                if step == 512: break
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(x, y)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite value loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), .5); optimizer.step()
    atomic_json(path, {'method': method, 'initial_seed': seed, 'fingerprint': protocol['fingerprint'], 'results': results})
    return str(path)


def report(output, protocol):
    groups = [json.loads((output/f'fit-{m}-{s}.json').read_text()) for m in ('ippo', 'mappo') for s in protocol['initial_seeds']]
    summary = []
    for method in ('ippo', 'mappo'):
        for clock in (False, True):
            for step in (128, 512):
                rs = [r for g in groups if g['method'] == method for r in g['results'] if r['clock'] == clock and r['updates'] == step]
                summary.append({'method': method, 'clock': clock, 'updates': step,
                    'mean_holdout_mse_ratio': st.mean(r['assessments']['heldout']['mse_ratio_to_training_mean'] for r in rs),
                    'mean_train_mse_ratio': st.mean(r['assessments']['training']['mse_ratio_to_training_mean'] for r in rs),
                    'mean_holdout_ev': st.mean(r['assessments']['heldout']['explained_variance'] for r in rs),
                    'mean_holdout_bias': st.mean(r['assessments']['heldout']['bias'] for r in rs)})
    gates = []
    for g in groups:
        values = {clock: st.mean(r['assessments']['heldout']['mse_ratio_to_training_mean'] for r in g['results'] if r['clock'] == clock and r['updates'] == 512) for clock in (False, True)}
        gates.append({'method': g['method'], 'seed': g['initial_seed'], 'no_clock': values[False], 'clock': values[True],
                      'beats_both_constant_and_no_clock': values[True] < min(1, values[False])})
    passed = all(g['beats_both_constant_and_no_clock'] for g in gates)
    atomic_json(output/'summary.json', {'protocol': protocol, 'summary': summary, 'gates': gates, 'passed': passed})
    lines = ['# Remaining-time critic check', '',
        f"Behavior: {protocol['behavior']}. Follow-up prompted by the failed first fitting gate; fresh check seeds 49401/49402 were specified before this run. Same four training seeds, three initialisations, and 128/512-update budgets. Both critics predict normalised values, use identical network shapes, and start with zero weight on the extra input. The control receives zero in that input; the intervention receives remaining-time / 600. Actors remain untouched.", '',
        '| Inputs | Remaining time | Updates | Holdout MSE / constant | Training MSE / constant | Holdout explained variance | Holdout bias |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['method']} | {r['clock']} | {r['updates']} | {r['mean_holdout_mse_ratio']:.3f} | {r['mean_train_mse_ratio']:.3f} | {r['mean_holdout_ev']:.3f} | {r['mean_holdout_bias']:.2f} |")
    lines += ['', f'Predeclared gate: every method/init at 512 updates beats both the constant predictor and its no-clock control (junction-average MSE ratios): **{passed}**.', '',
        'A fixed-horizon return depends on time remaining. Phase age in the existing observation is not episode time remaining. A successful regression here would support adding time information to a finite-horizon critic; it would not demonstrate better traffic control. A continuing-task formulation with correct truncation bootstrapping is a different design and must not be silently mixed with this finite-horizon test.', '',
        'The new seed pair is a development diagnostic, not the reserved final test set. Individual junction and seed results are retained. Averages are descriptive; temporally adjacent states are not independent samples for significance testing. Fixed-policy Monte Carlo fits differ from on-policy PPO/GAE training.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    path = output.parent/(output.name+'-results.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted([*output.glob('*.json'), output/'REPORT.md']): z.write(p, p.name)
    print('RESULTS', path, 'gate', passed, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('runs/critic-fit-actuated'))
    parser.add_argument('--output', type=Path, default=Path('runs/critic-horizon-actuated'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    if args.workers < 1 or output == parent: parser.error('Use a separate output and positive workers')
    prior = json.loads((parent/'protocol.json').read_text())
    for p, sha in prior['source'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != sha: raise ValueError('Parent source changed: '+p)
    protocol = dict(prior, schema='critic-horizon-v1', check_demand_seeds=[49401, 49402],
        parent_fingerprint=prior['fingerprint'],
        gate='Every method/init: clock 512-step junction-average MSE beats constant and no-clock',
        horizon_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trajectory_hashes={str(s): hashlib.sha256((parent/f'trajectory-{s}.pt').read_bytes()).hexdigest() for s in prior['training_demand_seeds']})
    protocol.pop('fingerprint')
    protocol['fingerprint'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol: raise ValueError('Changed protocol; use new output')
    atomic_json(manifest, protocol)
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for p in pool.map(collect, [(str(output), protocol, s) for s in protocol['check_demand_seeds']]): print('Collected', Path(p).name, flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [(str(parent), str(output), protocol, m, s) for m in ('ippo', 'mappo') for s in protocol['initial_seeds']]
        for p in pool.map(fit, jobs): print('Fitted', Path(p).name, flush=True)
    report(output, protocol)
    print(f'Elapsed {time.perf_counter()-started:.1f} seconds', flush=True)


if __name__ == '__main__':
    main()
