"""Frozen-policy critic test: 4 versus 32 independent training trajectories.

Both arms use normalised value outputs and remaining episode time. Initial
weights AND the output transform are shared, fitted using the small training
set only. Eight fresh check seeds never affect optimisation or model selection.
Fixed full-batch update counts are not equal-compute budgets: the large arm
processes eight times as many training states per update. No actor is trained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.critic_fit_check import collect, FixedCritic, TargetScale, prediction_metrics
from scripts.critic_horizon_check import with_clock
from src.traffic_control.fairness.pipeline import atomic_json

TRAIN = list(range(49101, 49133))
CHECK = list(range(49501, 49509))
INITIAL = [49301, 49302, 49303]
METHODS = ('ippo', 'mappo')
SIZES = (4, 32)
EXPECTED_ACTORS = {
    'plain-mappo-r0': '10644c0e9d43a5b3c0d6d2be289915f3a6e3622e4e568f2df5142d249604d14a',
    'priority-mappo-r0': '83654be5ad0bf01aad7d07db39227e5fda75199036a2d2188e4ceb02193e72a6',
}


def check_split(training, checks):
    if len(set(training)) != len(training) or len(set(checks)) != len(checks):
        raise ValueError('Repeated demand seed')
    if set(training) & set(checks): raise ValueError('Training/check overlap')
    if (set(training) | set(checks)) & set(range(58001, 58021)):
        raise ValueError('Final test seeds are reserved')


def value_model(inputs, initial_seed, shared_targets):
    model = FixedCritic(inputs, 'normalized-output', initial_seed, TargetScale.fit(shared_targets))
    with torch.no_grad(): model.network[0].weight[:, -1].zero_()
    return model


def paired_gate(small, large):
    """Compare absolute errors; normalisation denominators must not create wins."""
    if len(small) != len(large) or not small: raise ValueError('Unmatched metrics')
    s = st.mean(m['rmse']**2 for m in small)
    l = st.mean(m['rmse']**2 for m in large)
    shared = st.mean(m['constant_rmse']**2 for m in large)
    own = st.mean(m['own_constant_rmse']**2 for m in large)
    return {'small_mse': s, 'large_mse': l, 'large_to_small_mse': l/s,
            'large_to_shared_constant_mse': l/shared, 'large_to_own_constant_mse': l/own,
            'passed': l < min(s, shared, own)}


def load_trajectory(path, protocol):
    row = torch.load(path, weights_only=True)
    if row['fingerprint'] != protocol['fingerprint']:
        raise ValueError('Cached trajectory fingerprint differs: '+str(path))
    if row['seed'] != int(path.stem.rsplit('-', 1)[1]): raise ValueError('Wrong cached seed')
    return row


def collect_checked(args):
    root, protocol, seed = args
    path = Path(collect(args))
    load_trajectory(path, protocol)
    return str(path)


def validate_fit(row, protocol, method, initial):
    if (row['fingerprint'], row['method'], row['initial_seed']) != (protocol['fingerprint'], method, initial):
        raise ValueError('Fit provenance does not match the frozen protocol')
    expected = {(tls, size, step) for tls in ('C0','C1','C2') for size in SIZES for step in (0,128,512)}
    records = {(r['junction'], r['training_trajectories'], r['updates']): r for r in row['results']}
    if len(records) != len(row['results']) or set(records) != expected:
        raise ValueError('Incomplete or duplicate fit records')
    assessment_names = {'training','heldout', *[f'seed-{s}' for s in protocol['check_demand_seeds']]}
    for r in records.values():
        if set(r['assessments']) != assessment_names:
            raise ValueError('Incomplete fit assessments')
        for m in r['assessments'].values():
            if m['n'] <= 0 or any(v is not None and not math.isfinite(v) for v in m.values()):
                raise ValueError('Nonfinite or empty fit assessment')
    for tls in ('C0','C1','C2'):
        a,b = records[tls,4,0],records[tls,32,0]
        if (a['shared_target_mean'],a['shared_target_scale']) != (b['shared_target_mean'],b['shared_target_scale']):
            raise ValueError('Arms have different target transforms')
        for name in assessment_names - {'training'}:
            left,right=a['assessments'][name],b['assessments'][name]
            for field in left.keys() - {'own_constant_rmse'}:
                if left[field] != right[field]:
                    raise ValueError('Arms do not start from identical check predictions')


def fit_job(args):
    root, protocol, method, initial = args
    output = Path(root); path = output/f'fit-{method}-{initial}.json'
    if path.exists():
        row = json.loads(path.read_text())
        validate_fit(row, protocol, method, initial)
        return str(path)
    torch.set_num_threads(1)
    train = [load_trajectory(output/f'trajectory-{s}.pt', protocol) for s in protocol['training_demand_seeds']]
    held = [load_trajectory(output/f'trajectory-{s}.pt', protocol) for s in protocol['check_demand_seeds']]
    started = time.perf_counter(); rows = []
    for j, tls in enumerate(sorted(train[0]['junctions'])):
        datasets = {n: with_clock(train[:n], tls, method, True) for n in SIZES}
        held_data = {f'seed-{r["seed"]}': with_clock([r], tls, method, True) for r in held}
        held_data['heldout'] = with_clock(held, tls, method, True)
        small_targets = datasets[4][1]
        for size in SIZES:
            x, y = datasets[size]
            model = value_model(x.shape[1], initial + j*100003, small_targets)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
            for step in range(513):
                if step in (0, 128, 512):
                    assessments = {}
                    with torch.no_grad():
                        for name, (z, target) in {**held_data, 'training': (x, y)}.items():
                            m = prediction_metrics(target, model(z), model.transform.mean)
                            m['own_constant_rmse'] = float((target-y.mean()).square().mean())**.5
                            assessments[name] = m
                    rows.append({'junction': tls, 'training_trajectories': size, 'updates': step,
                        'training_states': len(x), 'shared_target_mean': float(model.transform.mean),
                        'shared_target_scale': float(model.transform.scale), 'own_training_target_mean': float(y.mean()),
                        'assessments': assessments})
                if step == 512: break
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(x, y)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite critic loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                if not torch.isfinite(grad): raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
    result = {'fingerprint': protocol['fingerprint'], 'method': method,
        'initial_seed': initial, 'seconds': time.perf_counter()-started, 'results': rows}
    validate_fit(result, protocol, method, initial)
    atomic_json(path, result)
    return str(path)


def bootstrap_ratio(cells, initial_seeds, check_seeds, repetitions=2000):
    """Paired crossed resampling of critic initialisation and whole demand seeds.

    Keep the three junctions together; never resample adjacent simulator states.
    This exploratory interval is conditional on these two fixed training sets.
    """
    rng = random.Random(4246495)
    ratios = []
    for _ in range(repetitions):
        initial = rng.choices(initial_seeds, k=len(initial_seeds))
        demand = rng.choices(check_seeds, k=len(check_seeds))
        small = large = 0.
        for i in initial:
            for d in demand:
                a, b = cells[i, d]; small += a; large += b
        ratios.append(large/small)
    ratios.sort()
    return [ratios[int(.025*repetitions)], ratios[min(repetitions-1, int(.975*repetitions))]]


def report(output, protocol):
    groups = [json.loads((output/f'fit-{m}-{s}.json').read_text()) for m in METHODS for s in INITIAL]
    for group in groups:
        validate_fit(group, protocol, group['method'], group['initial_seed'])
    summaries = []; gates = []; comparisons = []; detail = []
    for method in METHODS:
        subset = [g for g in groups if g['method'] == method]
        for size in SIZES:
            for step in (128, 512):
                rs = [r for g in subset for r in g['results'] if r['training_trajectories']==size and r['updates']==step]
                mse = st.mean(r['assessments']['heldout']['rmse']**2 for r in rs)
                constant = st.mean(r['assessments']['heldout']['constant_rmse']**2 for r in rs)
                own = st.mean(r['assessments']['heldout']['own_constant_rmse']**2 for r in rs)
                summaries.append({'method': method, 'trajectories': size, 'updates': step,
                    'heldout_rmse': mse**.5, 'heldout_mse_to_shared_constant': mse/constant,
                    'heldout_mse_to_own_constant': mse/own,
                    'training_rmse': st.mean(r['assessments']['training']['rmse']**2 for r in rs)**.5,
                    'heldout_bias': st.mean(r['assessments']['heldout']['bias'] for r in rs)})
        cells = {}; case_count = 0; improved = 0; below_own = 0
        for group in subset:
            lookup = {(r['junction'], r['training_trajectories']): r for r in group['results'] if r['updates']==512}
            small = [lookup[t,4]['assessments']['heldout'] for t in ('C0','C1','C2')]
            large = [lookup[t,32]['assessments']['heldout'] for t in ('C0','C1','C2')]
            gates.append({'method': method, 'seed': group['initial_seed'], **paired_gate(small, large)})
            for seed in CHECK:
                a = [lookup[t,4]['assessments'][f'seed-{seed}'] for t in ('C0','C1','C2')]
                b = [lookup[t,32]['assessments'][f'seed-{seed}'] for t in ('C0','C1','C2')]
                cells[group['initial_seed'],seed] = (st.mean(m['rmse']**2 for m in a), st.mean(m['rmse']**2 for m in b))
                for tls, old, new in zip(('C0','C1','C2'),a,b):
                    case_count += 1; improved += new['rmse'] < old['rmse']; below_own += new['rmse'] < new['own_constant_rmse']
                    detail.append({'method':method,'initial_seed':group['initial_seed'],'demand_seed':seed,
                                   'junction':tls,'small_rmse':old['rmse'],'large_rmse':new['rmse'],
                                   'own_constant_rmse':new['own_constant_rmse']})
        ratio = sum(b for a,b in cells.values())/sum(a for a,b in cells.values())
        comparisons.append({'method':method,'large_to_small_mse':ratio,
            'exploratory_paired_95_interval':bootstrap_ratio(cells,INITIAL,CHECK),
            'individual_cases':case_count,'cases_improved':improved,'cases_below_own_constant':below_own})
    passed = all(g['passed'] for g in gates)
    atomic_json(output/'summary.json', {'protocol':protocol,'summary':summaries,'comparisons':comparisons,
        'initialization_gates':gates,'passed':passed,'individual_cases':detail})
    lines = ['# Independent-trajectory count check','',
        f"Frozen behavior policy: {protocol['behavior']}. Four versus 32 independent 600-second training trajectories; eight fresh development check seeds; three new critic initialisations. Both arms use remaining time and normalised values, identical initial weights, and one common output transform fitted only on the four-seed subset. No actor training.",'',
        '| Inputs | Training trajectories | Updates | Check RMSE | Train RMSE | Check MSE / shared constant | Check MSE / own constant | Check bias |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['method']} | {r['trajectories']} | {r['updates']} | {r['heldout_rmse']:.2f} | {r['training_rmse']:.2f} | {r['heldout_mse_to_shared_constant']:.3f} | {r['heldout_mse_to_own_constant']:.3f} | {r['heldout_bias']:.2f} |")
    lines += ['','The shared constant predicts the four-trajectory training mean; the own constant predicts each arm\'s training mean. Absolute squared errors decide improvements, so changing a normalisation denominator cannot create a win. RMSE/bias are reward units, not seconds.','',
        '| Inputs | 32 / 4 check MSE | Exploratory paired 95% interval | Junction/seed/init cases improved | Cases beating own constant |',
        '|---|---:|---|---:|---:|']
    for r in comparisons:
        lo,hi=r['exploratory_paired_95_interval']
        lines.append(f"| {r['method']} | {r['large_to_small_mse']:.3f} | [{lo:.3f}, {hi:.3f}] | {r['cases_improved']}/{r['individual_cases']} | {r['cases_below_own_constant']}/{r['individual_cases']} |")
    lines += ['',f'Predeclared gate: every method/initialisation at 512 updates improves on four trajectories and beats both constant references: **{passed}**.','',
        'The bootstrap resamples whole check demand seeds and critic initialisations with paired arms, keeping junctions together. It is exploratory and conditional on these fixed training datasets; it omits uncertainty from sampling alternative training datasets and uses only three initialisations. Individual cases are correlated, not independent statistical trials.','',
        'Both arms use 128/512 full-batch updates, but the 32-trajectory arm processes eight times as many states per update. This is a fixed-update comparison, not equal compute or equal sample exposure. The smaller arm is the prespecified first four seeds, not a selected favourable subset. No checkpoint is selected using check results.','',
        'Source and demand/model hashes are retained. Test seeds 58001-58020 are untouched. This prediction test neither changes the traffic actor nor establishes IPPO/MAPPO traffic superiority, every-crossing fairness, or deployed benefit.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    archive = output.parent/(output.name+'-results.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted([*output.glob('*.json'),output/'REPORT.md']):z.write(p,p.name)
    print('RESULTS',archive,'gate',passed,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path)
    parser.add_argument('--model',choices=('actuated',*EXPECTED_ACTORS),default='actuated')
    parser.add_argument('--output',type=Path,default=Path('runs/critic-data-actuated'))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--stage',choices=('all','collect','fit','report'),default='all')
    args=parser.parse_args()
    if args.workers<1:parser.error('positive workers required')
    if (args.model!='actuated') != bool(args.pilot):parser.error('Pass saved model and pilot together')
    output=args.output.resolve()
    if args.pilot and (output==args.pilot.resolve() or args.pilot.resolve() in output.parents):parser.error('Separate output from pilot')
    check_split(TRAIN,CHECK)
    source_paths=[*(ROOT/'src/traffic_control').rglob('*.py'),*(ROOT/'config').glob('*.json'),
                  Path(__file__),ROOT/'scripts/critic_fit_check.py',ROOT/'scripts/critic_horizon_check.py']
    protocol={'schema':'critic-data-volume-v1','behavior':args.model,'pilot':str(args.pilot.resolve()) if args.pilot else None,
        'training_demand_seeds':TRAIN,'small_training_demand_seeds':TRAIN[:4],'check_demand_seeds':CHECK,
        'initial_seeds':INITIAL,'budgets':[128,512],'gamma':.99,'reward_weight':3,
        'configuration':{'end_seconds':600,'warmup_seconds':0,'measurement_end_seconds':600,'demand_end_seconds':600,
                         'vehicle_rate_per_hour':1920,'pedestrian_rate_per_hour':1200},
        'transform':'Fixed four-trajectory mean/scale shared by both arms; check data excluded',
        'gate':'Every method/init at 512: absolute MSE32 < MSE4 and both constant references',
        'source':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}}
    if args.pilot:
        prior=json.loads((args.pilot/'protocol.json').read_text())
        if prior['quick']:parser.error('Engineering checkpoints are not the learned-policy diagnostic')
        for path,sha in prior['source'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('Pilot source changed: '+path)
        protocol['checkpoint_sha256']=hashlib.sha256((args.pilot/args.model/'best.pt').read_bytes()).hexdigest()
        if protocol['checkpoint_sha256'] != EXPECTED_ACTORS[args.model]:
            raise ValueError('Actor differs from the previously reviewed diagnostic: '+args.model)
    protocol['fingerprint']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Changed protocol; use new output')
    atomic_json(path,protocol);started=time.perf_counter()
    if args.stage in ('all','collect'):
        with ProcessPoolExecutor(args.workers) as pool:
            for p in pool.map(collect_checked,[(str(output),protocol,s) for s in TRAIN+CHECK]):print('Collected',Path(p).name,flush=True)
    if args.stage in ('all','fit'):
        with ProcessPoolExecutor(args.workers) as pool:
            for p in pool.map(fit_job,[(str(output),protocol,m,s) for m in METHODS for s in INITIAL]):print('Fitted',Path(p).name,flush=True)
    if args.stage in ('all','fit','report'):report(output,protocol)
    print(f'Elapsed {time.perf_counter()-started:.1f} seconds',flush=True)


if __name__=='__main__':main()
