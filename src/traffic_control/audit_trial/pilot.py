"""Local, resumable factorial test of observed controller memory and PPO clipping."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from ..scenario import ROOT, executable

# Reuse the existing portable installation; no downloads or dependency installs.
if (ROOT / '.local/sumo-1.27.1/tools').exists():
    sys.path.insert(0, str(ROOT / '.local/sumo-1.27.1/tools'))
import torch
from ..connected import incoming_groups
from ..fairness.analysis import summarise
from ..fairness.demand import scenarios
from ..fairness.environment import run
from ..fairness.pipeline import atomic_json
from ..ppo_repair.pilot import atomic_torch, digest, cached_json
from .learning import ARMS, TrialNetworks, TrialLearner

BASELINES = ('fixed', 'actuated', 'pressure', 'uniform')
METHODS = ('ippo', 'mappo')
TRAIN_SCENARIOS = ('vmedium-pmedium', 'vmedium-plow', 'pedestrian-burst', 'short-spacing')
EVAL_SCENARIOS = ('vmedium-pmedium', 'pedestrian-burst', 'short-spacing', 'slow-walkers')
SCHEMA = 'audit-repair-factorial-v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def labels(protocol):
    return [(m, a, r, f'{m}-{a}-r{r}') for r in range(protocol['replicas'])
            for a in protocol['variants'] for m in protocol['methods']]


def make_protocol(quick=False):
    sources = sorted([*(ROOT / 'src/traffic_control').rglob('*.py'),
                      ROOT / 'config/fairness_v2.json'])
    p = dict(schema=SCHEMA, quick=quick, episodes=4 if quick else 64,
             replicas=1 if quick else 2, methods=list(METHODS), variants=list(ARMS),
             initial_seeds=[94001] if quick else [94001, 94002],
             training_seed_base=930001, training_seed_stride=1000,
             training_scenarios=list(TRAIN_SCENARIOS),
             checkpoint_stages=[4] if quick else [16, 32, 64],
             validation_episodes=[4] if quick else [16, 32, 64], validation_seeds=[950001],
             evaluation_seeds=[960001] if quick else list(range(960001, 960005)),
             evaluation_scenarios=['vmedium-pmedium'] if quick else list(EVAL_SCENARIOS),
             baselines=list(BASELINES), training_horizon_seconds=180 if quick else 600,
             evaluation_horizon_seconds=180 if quick else 2400,
             reward_weight=3, gamma=.99, gae_lambda=.95, ppo_epochs=4,
             learning_rate=.0003, entropy_weight=.01, value_weight=.5, max_grad_norm=.5,
             bootstrap_draws=0 if quick else 1000,
             selection='Final fixed-budget checkpoint only; validation is diagnostic, not selection',
             inference='Categorical with fixed per-demand/per-junction streams in training and evaluation',
             scope='Development repair ablation, normalized-clock control, pedestrian reward weight3; not final proposal study',
             observation_factor='legacy42 versus47 controller-memory inputs; equal initial functions; actors remain local',
             clipping_factor='joint versus separate actor and critic gradient-norm clipping',
             backlog_input=False, safety='Identical legal phases and junction collision checks in every arm',
             reserved_final_seeds_not_used=list(range(58001, 58021)),
             output_limit_bytes=512 * 1024 ** 2, minimum_free_bytes=512 * 1024 ** 2,
             python=platform.python_version(), torch=str(torch.__version__),
             sumo=subprocess.run([executable('sumo'), '--version'], capture_output=True,
                                 text=True, check=True).stdout.splitlines()[0],
             source={f.relative_to(ROOT).as_posix(): digest(f) for f in sources})
    groups = [{p['training_seed_base'] + r * p['training_seed_stride'] + e
               for r in range(p['replicas']) for e in range(p['episodes'])},
              set(p['validation_seeds']), set(p['evaluation_seeds']), set(p['reserved_final_seeds_not_used'])]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError('Seed sets overlap')
    p['fingerprint'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    return p


def verify_source(protocol):
    changed = [name for name, sha in protocol['source'].items()
               if not (ROOT / name).exists() or digest(ROOT / name) != sha]
    if changed:
        raise RuntimeError('Frozen source changed; use the source snapshot/new output: ' + ', '.join(changed))


def resource_check(root, protocol):
    if shutil.disk_usage(root).free < protocol['minimum_free_bytes']:
        raise RuntimeError('Experiment stopped: less than 512 MiB disk space remains')
    total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if total > protocol['output_limit_bytes']:
        raise RuntimeError('Experiment stopped: output exceeded its 512 MiB limit')
    if (root / 'STOP').exists():
        raise InterruptedError('STOP requested; current completed results are saved')


def cfg_for(protocol, scenario, training=False, model=None):
    cfg = dict(scenarios()[scenario])
    if protocol['quick']:
        cfg.update(end_seconds=180, warmup_seconds=0, measurement_end_seconds=120,
                   demand_end_seconds=90, burst_start_seconds=40)
    elif training:
        cfg.update(end_seconds=600, warmup_seconds=0, measurement_end_seconds=600, demand_end_seconds=600)
    cfg['observation_schema'] = model.observation_schema if model else 'legacy-v2'
    return cfg


def episode(root, protocol, scenario, seed, controller, model=None, training=False):
    root = Path(root)
    resource_check(root, protocol)
    verify_source(protocol)
    torch.set_num_threads(1)
    cfg = cfg_for(protocol, scenario, training, model)
    horizon = cfg.get('end_seconds', 2400)
    streams = {t: random.Random(900000 + seed + i * 1000003)
               for i, t in enumerate(('C0', 'C1', 'C2'))}
    trace, logged = [], {t: [] for t in streams}

    def policy(tls, obs, global_state, mask):
        legal = [i for i, yes in enumerate(mask) if yes]
        if len(legal) == 1:
            action, lp = legal[0], 0.
        else:
            with torch.no_grad():
                distribution = model.distribution(tls, obs, mask) if model else None
                probs = distribution.probs.tolist() if model else list(map(float, mask))
                action = streams[tls].choices(legal, weights=[probs[i] for i in legal])[0]
                lp = float(distribution.log_prob(torch.tensor(action))) if model else None
        if model:
            logged[tls].append(lp)
        return action

    with tempfile.TemporaryDirectory(prefix='rollout-', dir=root) as folder:
        try:
            metrics = run('policy' if model is not None or controller == 'uniform' else controller,
                          seed, folder, overrides=cfg,
                          policy=policy if model is not None or controller == 'uniform' else None,
                          trace_sink=trace)
            events = json.loads((Path(folder) / 'events.json').read_text())
        except BaseException:
            failed = root / 'failures'
            failed.mkdir(exist_ok=True)
            for name in ('failure.json', 'sumo.log'):
                if (Path(folder) / name).exists():
                    shutil.copyfile(Path(folder) / name, failed / f'{os.getpid()}-{seed}-{name}')
            raise
    features = metrics['observation_features']
    if model is not None and features != model.features:
        raise AssertionError('Model/environment feature order differs')
    diagnostics = {}
    for tls in features:
        rows = [r for r in trace if r['junction'] == tls]
        if model:
            cursor = 0
            for row in rows:
                if row['action'] is not None:
                    row['behavior_logp'] = logged[tls][cursor]
                    cursor += 1
            if cursor != len(logged[tls]):
                raise AssertionError('Action/log-probability alignment failed')
        # The last observation is the terminal reward endpoint. Its requested
        # action has no observed successor and is excluded by the learner too.
        free = [r for r in rows[:-1] if r['action'] is not None and sum(r['mask']) > 1]
        counts = Counter(r['action'] for r in free)
        gaps, longest = {a: 0 for a in ('EW', 'NS')}, {a: 0 for a in ('EW', 'NS')}
        for row in rows:
            obs = dict(zip(features[tls], row['observation']))
            for axis, edges in incoming_groups(3)[tls].items():
                queued = any(obs[e + '/queue'] > 0 for e in edges)
                gaps[axis] = gaps[axis] + 1 if queued and not obs[axis] else 0
                longest[axis] = max(longest[axis], gaps[axis])
        durations = Counter()
        history = metrics['phase_history'][tls]
        for (start, stage), (end, _) in zip(history, history[1:] + [(horizon, '')]):
            durations[stage] += end - start
        d = dict(free_decisions=len(free), free_action_counts={str(i): counts[i] for i in range(3)},
                 max_queued_no_green_seconds=longest, phase_seconds=dict(durations))
        if model and free:
            with torch.no_grad():
                dist = model.distribution(tls, [r['observation'] for r in free], [r['mask'] for r in free])
                d['mean_entropy'] = float(dist.entropy().mean())
                d['mean_max_action_probability'] = float(dist.probs.max(-1).values.mean())
        diagnostics[tls] = d
    return dict(result=summarise(metrics, events), features=features, diagnostics=diagnostics,
                sumo_version=metrics['sumo_version'], network_sha256=metrics['network_sha256'],
                vehicles_end=metrics['vehicles_end'], phase_history=metrics['phase_history'],
                events=events, rollout={'trace': trace, 'horizon': horizon})


def compact(outcome):
    return {k: v for k, v in outcome.items() if k not in ('rollout', 'events')}


def save_events(path, events):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix('.tmp')
    temp.write_bytes(gzip.compress(json.dumps(events, separators=(',', ':')).encode(), mtime=0))
    temp.replace(path)
    return digest(path)


def baseline_job(args):
    root, protocol, scenario, seed = args
    root = Path(root)
    path = root / 'baselines' / f'{scenario}-{seed}.json'
    cached = cached_json(path, protocol['fingerprint'])
    if cached:
        return str(path)
    records = []
    for controller in protocol['baselines']:
        out = episode(root, protocol, scenario, seed, controller)
        event_path = root / 'events' / f'baseline-{controller}-{scenario}-{seed}.json.gz'
        sha = save_events(event_path, out['events'])
        records.append(dict(controller=controller, events_sha256=sha, **compact(out)))
    if len({r['result']['demand_sha256'] for r in records}) != 1 or len({r['network_sha256'] for r in records}) != 1:
        raise AssertionError('Baseline demand/network mismatch')
    atomic_json(path, dict(fingerprint=protocol['fingerprint'], scenario=scenario, seed=seed, records=records))
    return str(path)


def reference(root, protocol, scenario, seed, outcome):
    data = cached_json(root / 'baselines' / f'{scenario}-{seed}.json', protocol['fingerprint'])
    if not data:
        raise ValueError('Missing paired baseline')
    ref = data['records'][0]
    if (ref['result']['demand_sha256'] != outcome['result']['demand_sha256']
            or ref['network_sha256'] != outcome['network_sha256']):
        raise AssertionError('Demand/network differs from baseline')


def train_job(args):
    root, protocol, method, arm, replica, label, target = args
    root = Path(root)
    directory = root / 'models' / label
    directory.mkdir(exist_ok=True, parents=True)
    fp = protocol['fingerprint']
    torch.set_num_threads(1)
    torch.manual_seed(protocol['initial_seeds'][replica])
    first = json.loads(next((root / 'baselines').glob('*.json')).read_text())['records'][0]
    model = TrialNetworks(method, first['features'], arm, protocol['initial_seeds'][replica])
    learner = TrialLearner(model, protocol['reward_weight'])
    latest = directory / 'latest.pt'
    completed, history, validations = 0, [], []
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved['fingerprint'] != fp or saved['label'] != label:
            raise ValueError('Checkpoint protocol/label mismatch')
        model.load_state_dict(saved['model']['weights'])
        learner.restore(saved['learner'])
        torch.set_rng_state(saved['rng'])
        completed, history, validations = saved['episodes'], saved['history'], saved['validations']
    while completed < min(target, protocol['episodes']):
        started = time.perf_counter()
        index = completed
        scenario = protocol['training_scenarios'][index % len(protocol['training_scenarios'])]
        seed = protocol['training_seed_base'] + replica * protocol['training_seed_stride'] + index
        out = episode(root, protocol, scenario, seed, 'policy', model, training=True)
        update = learner.update([out['rollout']])
        completed += 1
        history.append(dict(episodes=completed, optimizer_update_batches=len(history) + 1,
                            rollouts=[dict(episode=completed, scenario=scenario, seed=seed,
                                demand_sha256=out['result']['demand_sha256'], network_sha256=out['network_sha256'],
                                vehicles=out['result']['vehicles'], diagnostics=out['diagnostics'])],
                            update=update, seconds=time.perf_counter() - started))
        if completed in protocol['validation_episodes']:
            for seed in protocol['validation_seeds']:
                val = episode(root, protocol, 'vmedium-pmedium', seed, 'policy', model)
                reference(root, protocol, 'vmedium-pmedium', seed, val)
                validations.append(dict(episodes=completed, scenario='vmedium-pmedium', seed=seed, **compact(val)))
        atomic_torch(latest, dict(fingerprint=fp, label=label, episodes=completed,
                                  model=model.snapshot(), learner=learner.state(), rng=torch.get_rng_state(),
                                  history=history, validations=validations))
        atomic_json(directory / 'learning.json', dict(fingerprint=fp, label=label, history=history, validations=validations))
        print(label, 'episode', completed, 'seconds', round(time.perf_counter() - started, 2), flush=True)
    # Re-export from the last atomic checkpoint, even after an interrupted export.
    atomic_json(directory / 'learning.json', dict(fingerprint=fp, label=label, history=history, validations=validations))
    if completed == protocol['episodes']:
        done = cached_json(directory / 'complete.json', fp)
        if done:
            if done['checkpoint_sha256'] != digest(directory / 'final.pt'):
                raise ValueError('Completed checkpoint changed')
        else:
            atomic_torch(directory / 'final.pt', dict(fingerprint=fp, label=label, episodes=completed, model=model.snapshot()))
            atomic_json(directory / 'complete.json', dict(fingerprint=fp, label=label, episodes=completed,
                        update_batches=len(history), checkpoint_sha256=digest(directory / 'final.pt')))
    return label


def evaluation_job(args):
    root, protocol, method, arm, replica, label, scenario, seed = args
    root = Path(root)
    checkpoint = root / 'models' / label / 'final.pt'
    sha = digest(checkpoint)
    path = root / 'evaluation' / f'{label}-{scenario}-{seed}.json'
    cached = cached_json(path, protocol['fingerprint'])
    if cached:
        if cached['checkpoint_sha256'] != sha:
            raise ValueError('Evaluated checkpoint changed')
        return str(path)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['fingerprint'] != protocol['fingerprint'] or saved['episodes'] != protocol['episodes']:
        raise ValueError('Wrong/incomplete model')
    model = TrialNetworks.from_snapshot(saved['model'])
    model.eval()
    out = episode(root, protocol, scenario, seed, 'policy', model)
    reference(root, protocol, scenario, seed, out)
    events_sha = save_events(root / 'events' / f'{label}-{scenario}-{seed}.json.gz', out['events'])
    atomic_json(path, dict(fingerprint=protocol['fingerprint'], label=label, scenario=scenario, seed=seed,
                          checkpoint_sha256=sha, events_sha256=events_sha, **compact(out)))
    return str(path)


def progress(root, protocol, state='running', stage=None, error=None):
    jobs = labels(protocol)
    models, validation = {}, 0
    for _, _, _, label in jobs:
        path = root / 'models' / label / 'learning.json'
        data = json.loads(path.read_text()) if path.exists() else {}
        history = data.get('history', [])
        models[label] = history[-1]['episodes'] if history else 0
        validation += len(data.get('validations', []))
    groups = len(list((root / 'baselines').glob('*.json')))
    evaluation = len(list((root / 'evaluation').glob('*.json')))
    expected_groups = len({(s, d) for s in protocol['evaluation_scenarios'] for d in protocol['evaluation_seeds']}
                          | {('vmedium-pmedium', d) for d in protocol['validation_seeds']})
    result = dict(state=state, stage=stage, error=error, pid=os.getpid(), updated_utc=utc(),
                  fingerprint=protocol['fingerprint'], model_episodes=models,
                  training_completed=sum(models.values()), training_total=len(jobs) * protocol['episodes'],
                  baseline_completed=groups * len(protocol['baselines']), baseline_total=expected_groups * len(protocol['baselines']),
                  validation_completed=validation, validation_total=len(jobs) * len(protocol['validation_episodes']) * len(protocol['validation_seeds']),
                  evaluation_completed=evaluation, evaluation_total=len(jobs) * len(protocol['evaluation_scenarios']) * len(protocol['evaluation_seeds']))
    result['completed_simulations'] = sum(result[k] for k in ('training_completed', 'baseline_completed', 'validation_completed', 'evaluation_completed'))
    result['total_simulations'] = sum(result[k] for k in ('training_total', 'baseline_total', 'validation_total', 'evaluation_total'))
    atomic_json(root / 'status.json', result)
    return result


@contextmanager
def exclusive_lock(root):
    handle = (root / 'writer.lock').open('a+b')
    handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('Another writer owns this study')
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == 'nt': msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def pool_stage(root, protocol, workers, stage, function, arguments):
    progress(root, protocol, stage=stage)
    with ProcessPoolExecutor(workers) as pool:
        pending = {pool.submit(function, arg) for arg in arguments}
        try:
            while pending:
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    print(stage, future.result(), flush=True)
                progress(root, protocol, stage=stage)
        except BaseException:
            # Signal active workers to stop at their next simulation boundary.
            (root / 'STOP').touch()
            for future in pending:
                future.cancel()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/audit-repair-trial')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--stage', choices=('all', 'baselines', 'train', 'evaluate', 'report'), default='all')
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error('Use one or two CPU workers for this local study')
    root = args.output.resolve()
    if args.status:
        print((root / 'status.json').read_text() if (root / 'status.json').exists() else 'Not started')
        return
    root.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    with exclusive_lock(root):
        p = make_protocol(args.quick)
        old = cached_json(root / 'protocol.json', p['fingerprint'])
        if old and old != p:
            raise ValueError('Protocol changed; use a new output directory')
        atomic_json(root / 'protocol.json', p)
        for name in ('baselines', 'models', 'evaluation', 'events'):
            (root / name).mkdir(exist_ok=True)
        if not (root / 'source.zip').exists():
            with zipfile.ZipFile(root / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for name in p['source']:
                    archive.write(ROOT / name, name)
        jobs = labels(p)
        try:
            resource_check(root, p)
            if args.stage in ('all', 'baselines'):
                # Four validation baselines supply the observation schema first.
                pool_stage(root, p, args.workers, 'prepare', baseline_job,
                           [(str(root), p, 'vmedium-pmedium', s) for s in p['validation_seeds']])
            if args.stage in ('all', 'train'):
                for target in p['checkpoint_stages']:
                    pool_stage(root, p, args.workers, f'train-{target}', train_job,
                               [(str(root), p, *job, target) for job in jobs])
            if args.stage in ('all', 'baselines'):
                pool_stage(root, p, args.workers, 'evaluation-baselines', baseline_job,
                           [(str(root), p, s, d) for s in p['evaluation_scenarios'] for d in p['evaluation_seeds']])
            if args.stage in ('all', 'evaluate'):
                pool_stage(root, p, args.workers, 'evaluate', evaluation_job,
                           [(str(root), p, *job, s, d) for s in p['evaluation_scenarios']
                            for d in p['evaluation_seeds'] for job in jobs])
            if args.stage in ('all', 'report'):
                from .reporting import report
                progress(root, p, stage='report')
                report(root, p, draws=p['bootstrap_draws'])
                progress(root, p, state='complete', stage='complete')
                print('Complete:', root / 'REPORT.md', flush=True)
            else:
                progress(root, p, state='stage_complete', stage=args.stage)
        except BaseException as exc:
            progress(root, p, state='stopped' if isinstance(exc, InterruptedError) else 'failed',
                     error=repr(exc))
            raise


if __name__ == '__main__':
    main()
