"""Resumable CPU study: retain vehicle gains while improving crossing tails.

Run from the repository root with python -m scripts.pedestrian_finetune.pilot.
No source/config under src/ is modified by this supplementary experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile

from src.traffic_control.audit_trial.pilot import (
    ROOT, torch, utc, episode, compact, digest, cached_json, exclusive_lock,
    verify_source, resource_check, TRAIN_SCENARIOS, EVAL_SCENARIOS,
)
from src.traffic_control.ppo_repair.learning import RepairNetworks
from src.traffic_control.scenario import executable
from .learning import FineTuneLearner
from .teacher import labelled_rollout

ARMS = ('continue', 'tail', 'guided')
BASELINES = ('fixed', 'actuated', 'pressure')
PARENT_PREFIX = 'runs/guided-exploration/'
DEFAULT_ARCHIVE = ROOT / 'runs/guided-exploration-results.zip'
LEARNER_CFG = dict(learning_rate=1e-4, adam_epsilon=1e-5, ppo_epochs=4,
    gamma=.99, gae_lambda=.95, clip_epsilon=.2, entropy_weight=.01, value_weight=.5,
    max_gradient_norm=.5, pedestrian_weight=3., parent_kl_weight=.02,
    tail_threshold_seconds=20., tail_scale_seconds=20., tail_cap=5., tail_weight=1.,
    guidance_weight=.5, urgency_seconds=20.)


def replace_with_retry(temp, target):
    """Retry Windows sharing violations; never remove/truncate an old checkpoint."""
    for attempt in range(51):
        try:
            temp.replace(target)
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 50:
                raise
            time.sleep(.1)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    replace_with_retry(temp, path)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def atomic_torch(path, data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    atomic_bytes(path, buffer.getvalue())


def save_gzip(path, data):
    raw = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    atomic_bytes(path, gzip.compress(raw, mtime=0))
    return digest(path)


def archive_parent_info(archive, replicas):
    """Verify the original manifest before accepting any executable-free weights."""
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('manifest.json'))
        names = [PARENT_PREFIX + 'protocol.json'] + [
            PARENT_PREFIX + f'models/mappo-control-r{r}/final.pt' for r in range(replicas)]
        for name in names:
            if hashlib.sha256(z.read(name)).hexdigest() != manifest.get(name):
                raise ValueError('Parent archive manifest mismatch: ' + name)
        original = json.loads(z.read(names[0]))
        if original['quick'] or original['episodes'] != 128 or original['replicas'] != 3:
            raise ValueError('Expected the complete three-replica, 128-episode parent study')
        return dict(archive_sha256=digest(archive), protocol_sha256=manifest[names[0]],
                    fingerprint=original['fingerprint'], episodes=original['episodes'],
                    source=original['source'], initial_seeds=original['initial_seeds'],
                    checkpoints={f'mappo-original-r{r}': {'member': names[r + 1],
                        'sha256': manifest[names[r + 1]], 'parent_label': f'mappo-control-r{r}'}
                                 for r in range(replicas)})


def make_protocol(quick=False, archive=DEFAULT_ARCHIVE):
    sources = sorted([*(ROOT / 'src/traffic_control').rglob('*.py'),
                      *(ROOT / 'scripts/pedestrian_finetune').glob('*.py'),
                      ROOT / 'config/fairness_v2.json'])
    p = dict(schema='pedestrian-warm-start-v1', quick=quick,
        episodes=2 if quick else 64, replicas=1 if quick else 3,
        methods=['mappo'], variants=list(ARMS), arms=list(ARMS), baselines=list(BASELINES),
        training_seed_base=990001, training_seed_stride=1000,
        training_scenarios=list(TRAIN_SCENARIOS),
        validation_episodes=[2] if quick else [16, 32, 64],
        checkpoint_stages=[1, 2] if quick else [16, 32, 64],
        validation_seeds=[970001] if quick else [970001, 970002],
        evaluation_seeds=[980001] if quick else list(range(980001, 980005)),
        evaluation_scenarios=['vmedium-pmedium'] if quick else list(EVAL_SCENARIOS),
        training_horizon_seconds=180 if quick else 600,
        evaluation_horizon_seconds=180 if quick else 2400,
        bootstrap_draws=0 if quick else 1000, learner=LEARNER_CFG,
        starting_models=archive_parent_info(archive, 1 if quick else 3),
        parent_interpretation='Original MAPPO-control uses pedestrian weight 3; it is vehicle-efficient, not a vehicle-only-reward model',
        training='Full parent actor/critic/PopArt state retained; fresh Adam, separate gradient clipping and parent KL common to all arms',
        reward='Base -(queue+uninserted_due+3*pending)/10; tail/guided subtract sum of capped squared excess crossing age above 20 seconds',
        guidance='Guided only: local actuated requested-action labels on student states; legal PED and oldest pending age >=20 seconds; CE sum divided by all free choices',
        dataset='All free training decisions retained, including nonselected advice; no failure/outcome filtering; forced/terminal rows excluded',
        inference='Categorical actions with paired per-demand/per-junction random streams',
        selection='Final fixed-budget checkpoint only; all starting replicas; no checkpoint selection by validation or evaluation',
        contrasts=[['mappo-tail', 'mappo-continue'], ['mappo-guided', 'mappo-tail'], ['mappo-guided', 'mappo-continue']],
        observation_schema='legacy-v2; 42 local features; MAPPO critic has global state and horizon clock',
        safety='Identical existing safe phases, occupancy clearance and collision checks for every controller',
        scope='Exploratory supplementary MAPPO fine-tuning; partial crossing protection does not replace original every-crossing criterion',
        limitations=['Synthetic, assumed demand; no deployed signal claim',
                    'Four evaluation demand seeds and three parent replicas are a development study, not final five-by-ten confirmation',
                    '600-second training resets versus 2400-second evaluation remain different',
                    'Physical pedestrian stalls can remain; entry wait and entered-without-exit are reported separately',
                    '20-second threshold is an engineering hypothesis, not a pedestrian p95 guarantee',
                    'Short-spacing scenario also changes demand; it is not an isolated geometry experiment'],
        reserved_final_seeds_not_used=list(range(58001, 58021)),
        output_limit_bytes=512 * 1024**2, minimum_free_bytes=512 * 1024**2,
        python=platform.python_version(), torch=str(torch.__version__),
        sumo=subprocess.run([executable('sumo'), '--version'], capture_output=True,
                            text=True, check=True).stdout.splitlines()[0],
        source={f.relative_to(ROOT).as_posix(): digest(f) for f in sources})
    groups = [{p['training_seed_base'] + r * p['training_seed_stride'] + e
               for r in range(p['replicas']) for e in range(p['episodes'])},
              set(p['validation_seeds']), set(p['evaluation_seeds']), set(p['reserved_final_seeds_not_used'])]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError('Overlapping seed sets')
    p['simulation_totals'] = dict(training=3 * p['replicas'] * p['episodes'],
        validation=3 * p['replicas'] * len(p['validation_episodes']) * len(p['validation_seeds']),
        baselines=len(BASELINES) * (len(p['validation_seeds']) + len(p['evaluation_seeds']) * len(p['evaluation_scenarios'])),
        evaluation=4 * p['replicas'] * len(p['evaluation_seeds']) * len(p['evaluation_scenarios']))
    p['fingerprint'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    return p


def labels(p, include_original=False):
    return [(arm, r, f'mappo-{arm}-r{r}') for r in range(p['replicas'])
            for arm in (['original'] if include_original else []) + p['arms']]


def prepare_parents(root, p, archive):
    if digest(archive) != p['starting_models']['archive_sha256']:
        raise ValueError('Parent archive changed')
    with zipfile.ZipFile(archive) as z:
        for label, info in p['starting_models']['checkpoints'].items():
            path = root / 'models' / label / 'final.pt'
            if not path.exists():
                raw = z.read(info['member'])
                if hashlib.sha256(raw).hexdigest() != info['sha256']:
                    raise ValueError('Parent checkpoint mismatch')
                atomic_bytes(path, raw)
            load_model(root, p, label)


def load_model(root, p, label):
    path = root / 'models' / label / 'final.pt'
    sha = digest(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    original = p['starting_models']['checkpoints'].get(label)
    if original:
        if (sha != original['sha256'] or saved['label'] != original['parent_label'] or
            saved['fingerprint'] != p['starting_models']['fingerprint'] or
            saved['episodes'] != p['starting_models']['episodes']):
            raise ValueError('Wrong original checkpoint: ' + label)
    else:
        done = cached_json(path.parent / 'complete.json', p['fingerprint'])
        parent_label = 'mappo-original-r' + label.rsplit('-r', 1)[1]
        expected_parent = p['starting_models']['checkpoints'][parent_label]['sha256']
        if (not done or done['checkpoint_sha256'] != sha or done['episodes'] != p['episodes'] or
            done['label'] != label or done['parent_sha256'] != expected_parent or
            saved['parent_sha256'] != expected_parent or
            saved['fingerprint'] != p['fingerprint'] or saved['label'] != label or
            saved['episodes'] != p['episodes']):
            raise ValueError('Wrong/incomplete fine-tuned checkpoint: ' + label)
    model = RepairNetworks.from_snapshot(saved['model'])
    if model.method != 'mappo' or model.variant != 'normalized-clock':
        raise ValueError('Unexpected model architecture')
    model.observation_schema = 'legacy-v2'
    return model, sha


def crossing_completion(out):
    counts = {c: dict(entered_without_observed_exit=0, completed_visits=0)
              for c in out['result']['crossings']}
    for event in out['events']:
        if event['entered_at'] is not None:
            key = 'entered_without_observed_exit' if event['exited_at'] is None else 'completed_visits'
            counts[event['crossing_id']][key] += 1
    return counts


def record_outcome(root, out, name):
    sha = save_gzip(root / 'events' / (name + '.json.gz'), out['events'])
    return dict(**compact(out), events_sha256=sha, crossing_completion=crossing_completion(out))


def reference(root, p, scenario, seed, out):
    rows = cached_json(root / 'baselines' / f'{scenario}-{seed}.json', p['fingerprint'])
    if (not rows or rows['scenario'] != scenario or rows['seed'] != seed or
        [r['controller'] for r in rows['records']] != p['baselines']):
        raise ValueError('Missing paired baseline')
    for row in rows['records']:
        if (row['network_sha256'] != out['network_sha256'] or
            row['result']['demand_sha256'] != out['result']['demand_sha256']):
            raise ValueError('Unmatched demand/network')
        if digest(root / 'events' / f'baseline-{row["controller"]}-{scenario}-{seed}.json.gz') != row['events_sha256']:
            raise ValueError('Paired baseline event archive changed')


def verify_source_archive(root, p):
    with zipfile.ZipFile(root / 'source.zip') as z:
        if set(z.namelist()) != set(p['source']):
            raise ValueError('Source archive file list differs')
        for name, sha in p['source'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError('Source archive changed: ' + name)


def baseline_job(args):
    root, p, scenario, seed = args
    root = Path(root)
    path = root / 'baselines' / f'{scenario}-{seed}.json'
    cached = cached_json(path, p['fingerprint'])
    if cached:
        if (cached['scenario'] != scenario or cached['seed'] != seed or
            [r['controller'] for r in cached['records']] != p['baselines']):
            raise ValueError('Incomplete cached baselines')
        return str(path)
    rows = []
    for controller in p['baselines']:
        out = episode(root, p, scenario, seed, controller)
        rows.append(dict(controller=controller, **record_outcome(root, out, f'baseline-{controller}-{scenario}-{seed}')))
    if len({r['network_sha256'] for r in rows}) != 1 or len({r['result']['demand_sha256'] for r in rows}) != 1:
        raise ValueError('Unmatched baselines')
    atomic_json(path, dict(fingerprint=p['fingerprint'], scenario=scenario, seed=seed, records=rows))
    return str(path)


def train_job(args):
    root, p, arm, replica, label, target = args
    root = Path(root)
    directory = root / 'models' / label
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    parent, parent_sha = load_model(root, p, f'mappo-original-r{replica}')
    model = RepairNetworks.from_snapshot(parent.snapshot())
    model.observation_schema = 'legacy-v2'
    learner = FineTuneLearner(model, parent, arm, p['learner'])
    torch.manual_seed(1000001 + replica)
    latest = directory / 'latest.pt'
    completed, history, validations = 0, [], []
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if (saved['fingerprint'] != p['fingerprint'] or saved['label'] != label or
            saved['parent_sha256'] != parent_sha):
            raise ValueError('Wrong resume state')
        model.load_state_dict(saved['model']['weights'])
        learner.restore(saved['learner'])
        torch.set_rng_state(saved['rng'])
        completed, history, validations = saved['episodes'], saved['history'], saved['validations']
    while completed < min(target, p['episodes']):
        started = time.perf_counter()
        scenario = p['training_scenarios'][completed % len(p['training_scenarios'])]
        seed = p['training_seed_base'] + replica * p['training_seed_stride'] + completed
        out = episode(root, p, scenario, seed, 'policy', model, training=True)
        dataset = labelled_rollout(out['rollout']['trace'], out['features'],
                                  horizon=out['rollout']['horizon'], urgency_seconds=p['learner']['urgency_seconds'])
        # Label every free training decision before the on-policy update.
        data_path = root / 'teacher-data' / label / f'{completed + 1:03d}.json.gz'
        dataset.update(fingerprint=p['fingerprint'], label=label, episode=completed + 1,
                       scenario=scenario, seed=seed, parent_sha256=parent_sha,
                       demand_sha256=out['result']['demand_sha256'], network_sha256=out['network_sha256'])
        dataset_sha = save_gzip(data_path, dataset)
        update = learner.update([out['rollout']], episode_index=completed)
        completed += 1
        history.append(dict(episodes=completed, scenario=scenario, seed=seed,
            demand_sha256=out['result']['demand_sha256'], network_sha256=out['network_sha256'],
            dataset_sha256=dataset_sha, vehicles=out['result']['vehicles'],
            diagnostics=out['diagnostics'], crossing_completion=crossing_completion(out),
            update=update, seconds=time.perf_counter() - started))
        if completed in p['validation_episodes']:
            for val_seed in p['validation_seeds']:
                val = episode(root, p, 'vmedium-pmedium', val_seed, 'policy', model)
                reference(root, p, 'vmedium-pmedium', val_seed, val)
                validations.append(dict(episodes=completed, scenario='vmedium-pmedium', seed=val_seed,
                    **record_outcome(root, val, f'validation-{label}-{completed}-{val_seed}')))
        atomic_torch(latest, dict(fingerprint=p['fingerprint'], label=label, parent_sha256=parent_sha,
            episodes=completed, model=model.snapshot(), learner=learner.state(), rng=torch.get_rng_state(),
            history=history, validations=validations))
        atomic_json(directory / 'learning.json', dict(fingerprint=p['fingerprint'], label=label,
            parent_sha256=parent_sha, history=history, validations=validations))
        print(label, 'episode', completed, 'seconds', round(time.perf_counter() - started, 2), flush=True)
    atomic_json(directory / 'learning.json', dict(fingerprint=p['fingerprint'], label=label,
        parent_sha256=parent_sha, history=history, validations=validations))
    if completed == p['episodes']:
        done = cached_json(directory / 'complete.json', p['fingerprint'])
        if done:
            load_model(root, p, label)
        else:
            atomic_torch(directory / 'final.pt', dict(fingerprint=p['fingerprint'], label=label,
                parent_sha256=parent_sha, episodes=completed, model=model.snapshot()))
            atomic_json(directory / 'complete.json', dict(fingerprint=p['fingerprint'], label=label,
                parent_sha256=parent_sha, episodes=completed, checkpoint_sha256=digest(directory / 'final.pt')))
    return label


def evaluation_job(args):
    root, p, arm, replica, label, scenario, seed = args
    root = Path(root)
    model, sha = load_model(root, p, label)
    path = root / 'evaluation' / f'{label}-{scenario}-{seed}.json'
    cached = cached_json(path, p['fingerprint'])
    if cached:
        if cached['checkpoint_sha256'] != sha:
            raise ValueError('Evaluated checkpoint changed')
        reference(root, p, scenario, seed, cached)
        return str(path)
    model.eval()
    before = model.snapshot()['weights']
    out = episode(root, p, scenario, seed, 'policy', model)
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise AssertionError('Evaluation mutated model')
    reference(root, p, scenario, seed, out)
    atomic_json(path, dict(fingerprint=p['fingerprint'], label=label, scenario=scenario, seed=seed,
        checkpoint_sha256=sha, **record_outcome(root, out, f'{label}-{scenario}-{seed}')))
    return str(path)


def progress(root, p, state='running', stage=None, error=None):
    episodes, validation = {}, 0
    for _, _, label in labels(p):
        data = cached_json(root / 'models' / label / 'learning.json', p['fingerprint']) or {}
        episodes[label] = len(data.get('history', []))
        validation += len(data.get('validations', []))
    counts = dict(training=sum(episodes.values()), validation=validation,
        baselines=len(list((root / 'baselines').glob('*.json'))) * len(p['baselines']),
        evaluation=len(list((root / 'evaluation').glob('*.json'))))
    result = dict(updated=utc(), fingerprint=p['fingerprint'], state=state, stage=stage,
        error=error, model_episodes=episodes, completed=counts, planned=p['simulation_totals'],
        completed_simulations=sum(counts.values()), total_simulations=sum(p['simulation_totals'].values()))
    atomic_json(root / 'status.json', result)
    return result


def pool_stage(root, p, workers, stage, function, arguments):
    verify_source(p)
    progress(root, p, stage=stage)
    # A single worker runs in this process: avoid an extra Torch interpreter on
    # a memory-constrained laptop. Atomic exports still support later resume.
    if workers == 1:
        for arg in arguments:
            print(stage, function(arg), flush=True)
            progress(root, p, stage=stage)
        return
    with ProcessPoolExecutor(workers) as pool:
        pending = {pool.submit(function, arg) for arg in arguments}
        try:
            while pending:
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    print(stage, future.result(), flush=True)
                progress(root, p, stage=stage)
        except BaseException:
            (root / 'STOP').touch()
            for future in pending:
                future.cancel()
            raise


def report(root, p):
    from .reporting import aggregate_scenario, write_report
    verify_source(p)
    verify_source_archive(root, p)
    identities, training = {}, {}
    for _, r, label in labels(p, include_original=True):
        _, identities[label] = load_model(root, p, label)
        if label.startswith('mappo-original-'):
            continue
        data = cached_json(root / 'models' / label / 'learning.json', p['fingerprint'])
        if (not data or data['label'] != label or
            data['parent_sha256'] != p['starting_models']['checkpoints'][f'mappo-original-r{r}']['sha256'] or
            [h['episodes'] for h in data['history']] != list(range(1, p['episodes'] + 1))):
            raise ValueError('Incomplete training history: ' + label)
        expected = [(e, s) for e in p['validation_episodes'] for s in p['validation_seeds']]
        if [(v['episodes'], v['seed']) for v in data['validations']] != expected:
            raise ValueError('Incomplete validation: ' + label)
        for val in data['validations']:
            if val['scenario'] != 'vmedium-pmedium':
                raise ValueError('Wrong validation scenario')
            reference(root, p, val['scenario'], val['seed'], val)
            event_path = root / 'events' / f'validation-{label}-{val["episodes"]}-{val["seed"]}.json.gz'
            if digest(event_path) != val['events_sha256']:
                raise ValueError('Validation event archive changed')
        for h in data['history']:
            e = h['episodes']
            expected_seed = p['training_seed_base'] + r * p['training_seed_stride'] + e - 1
            if h['seed'] != expected_seed or h['scenario'] != p['training_scenarios'][(e - 1) % len(p['training_scenarios'])]:
                raise ValueError('Wrong training demand schedule')
            key = (r, e)
            pair = (h['demand_sha256'], h['network_sha256'])
            if key in training and training[key] != pair:
                raise ValueError('Unmatched training demand across arms')
            training[key] = pair
            if digest(root / 'teacher-data' / label / f'{e:03d}.json.gz') != h['dataset_sha256']:
                raise ValueError('Training demonstration data changed')
    scenarios = {}
    for scenario in p['evaluation_scenarios']:
        baseline = {name: [] for name in p['baselines']}
        for seed in p['evaluation_seeds']:
            rows = cached_json(root / 'baselines' / f'{scenario}-{seed}.json', p['fingerprint'])
            if not rows or [r['controller'] for r in rows['records']] != p['baselines']:
                raise ValueError('Missing baseline coverage')
            for row in rows['records']:
                event_path = root / 'events' / f'baseline-{row["controller"]}-{scenario}-{seed}.json.gz'
                if digest(event_path) != row['events_sha256']:
                    raise ValueError('Baseline event archive changed')
                baseline[row['controller']].append(row)
        learned = {}
        for arm in ['original'] + p['arms']:
            matrix = []
            for r in range(p['replicas']):
                label, rows = f'mappo-{arm}-r{r}', []
                for seed in p['evaluation_seeds']:
                    row = cached_json(root / 'evaluation' / f'{label}-{scenario}-{seed}.json', p['fingerprint'])
                    if (not row or row['label'] != label or row['scenario'] != scenario or row['seed'] != seed or
                        row['checkpoint_sha256'] != identities[label]):
                        raise ValueError('Missing/wrong evaluation: ' + label)
                    if digest(root / 'events' / f'{label}-{scenario}-{seed}.json.gz') != row['events_sha256']:
                        raise ValueError('Evaluation event archive changed')
                    reference(root, p, scenario, seed, row)
                    rows.append(row)
                matrix.append(rows)
            learned[f'mappo-{arm}'] = matrix
        scenarios[scenario] = aggregate_scenario(baseline, learned, draws=p['bootstrap_draws'], contrasts=p['contrasts'])
    coverage = dict(complete=True, checkpoint_sha256=identities, simulation_totals=p['simulation_totals'],
                    training_demands_matched=True, events_and_demonstrations_sha256_verified=True)
    write_report(root, p, scenarios, coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pedestrian-finetune')
    parser.add_argument('--parent-archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--stage', choices=('all', 'baselines', 'train', 'evaluate', 'report'), default='all')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.status:
        print((root / 'status.json').read_text() if (root / 'status.json').exists() else 'Not started')
        return
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with exclusive_lock(root):
        p = make_protocol(args.quick, args.parent_archive)
        old = cached_json(root / 'protocol.json', p['fingerprint'])
        if old and old != p:
            raise ValueError('Protocol changed; use a fresh output directory')
        atomic_json(root / 'protocol.json', p)
        for name in ('baselines', 'models', 'evaluation', 'events', 'teacher-data'):
            (root / name).mkdir(exist_ok=True)
        if not (root / 'source.zip').exists():
            with zipfile.ZipFile(root / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                for name in p['source']:
                    z.write(ROOT / name, name)
        verify_source_archive(root, p)
        prepare_parents(root, p, args.parent_archive)
        jobs = labels(p)
        try:
            resource_check(root, p)
            if args.stage in ('all', 'baselines'):
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
                     for d in p['evaluation_seeds'] for job in labels(p, include_original=True)])
            if args.stage in ('all', 'report'):
                progress(root, p, stage='report')
                report(root, p)
                progress(root, p, state='complete', stage='complete')
                print('Complete:', root / 'REPORT.md', flush=True)
            else:
                progress(root, p, state='stage_complete', stage=args.stage)
        except BaseException as exc:
            progress(root, p, state='stopped' if isinstance(exc, InterruptedError) else 'failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
