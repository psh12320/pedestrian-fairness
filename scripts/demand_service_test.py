"""Frozen-model demand factorial and first-legal pedestrian-service diagnostic.

No training, simulator changes, or final-test seeds. Run as a module from root.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
import zipfile

from scripts.pedestrian_finetune import pilot as shared
from src.traffic_control.fairness.demand import scenarios
from src.traffic_control.audit_trial.reporting import validate_result

ROOT = shared.ROOT
PREFIX = 'runs/pedestrian-finetune-cluster/'
ARCHIVE = ROOT / 'runs/pedestrian-finetune-cluster-results.zip'
ARCHIVE_SHA = '255b82536b3a341c8964ef2338b5bd0a5e54fbe73dfd8e89fd9315b18c46354b'
SCENARIOS = ('vlow-plow', 'vlow-pmedium', 'vmedium-plow', 'vmedium-pmedium')
ARMS = ('original', 'continue', 'tail', 'guided', 'early')


class EarlyServicePolicy:
    """Override requests only; unchanged SafeSignal owns physical transitions.

    Use the SAME categorical draw at each free decision as the plain policy.
    This changes the distribution to a point mass when PED is legal and a
    local crossing has a pending request. No oldest-age threshold is applied.
    Empty PED requests are deliberately unchanged to isolate early service.
    """
    def __init__(self, model):
        self.model = model
        self.features = model.features
        self.observation_schema = model.observation_schema
        self.indices = {j: [i for i, n in enumerate(names)
                           if n.startswith(':' + j + '_c') and n.endswith('/count')]
                        for j, names in self.features.items()}
        if any(len(indices) != 4 for indices in self.indices.values()):
            raise ValueError('Expected four local crossing counts per junction')

    def distribution(self, junction, observations, masks):
        torch = shared.torch
        dist = self.model.distribution(junction, observations, masks)
        obs = torch.as_tensor(observations, dtype=dist.probs.dtype, device=dist.probs.device)
        legal = torch.as_tensor(masks, dtype=torch.bool, device=dist.probs.device)
        pending = obs[..., self.indices[junction]].gt(0).any(dim=-1)
        force = pending & legal[..., 2]
        point = torch.zeros_like(dist.probs)
        point[..., 2] = 1.
        return torch.distributions.Categorical(probs=torch.where(force[..., None], point, dist.probs))


def make_protocol(quick=False):
    if shared.digest(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Expected the independently reviewed complete fine-tuning archive')
    with zipfile.ZipFile(ARCHIVE) as z:
        manifest = json.loads(z.read('manifest.json'))
        raw = z.read(PREFIX + 'protocol.json')
        if hashlib.sha256(raw).hexdigest() != manifest[PREFIX + 'protocol.json']:
            raise ValueError('Source protocol hash mismatch')
        parent = json.loads(raw)
        shared.verify_source(parent)
        models = {}
        for r in range(1 if quick else 3):
            for arm in ARMS[:-1]:
                label = f'mappo-{arm}-r{r}'
                member = PREFIX + f'models/{label}/final.pt'
                if hashlib.sha256(z.read(member)).hexdigest() != manifest[member]:
                    raise ValueError('Model archive hash mismatch')
                models[label] = dict(member=member, sha256=manifest[member])
                if arm != 'original':
                    done_member = PREFIX + f'models/{label}/complete.json'
                    done_raw = z.read(done_member)
                    if hashlib.sha256(done_raw).hexdigest() != manifest[done_member]:
                        raise ValueError('Completion hash mismatch')
                    models[label]['complete'] = json.loads(done_raw)
    sources = dict(parent['source'])
    for file in ('scripts/demand_service_test.py', 'scripts/demand_service_report.py'):
        sources[file] = shared.digest(ROOT / file)
    seeds = [1011001] if quick else list(range(1010001, 1010011))
    used = (set(parent['evaluation_seeds']) | set(parent['validation_seeds']) |
            set(parent['reserved_final_seeds_not_used']) |
            {parent['training_seed_base'] + r * parent['training_seed_stride'] + e
             for r in range(3) for e in range(64)})
    if used & set(seeds):
        raise ValueError('Demand seed overlap')
    p = dict(schema='demand-early-service-v1', quick=quick, source=sources,
        archive_sha256=ARCHIVE_SHA, starting_protocol=parent, models=models,
        arms=list(ARMS), baselines=['fixed', 'actuated', 'pressure'],
        replicas=1 if quick else 3, evaluation_seeds=seeds,
        evaluation_scenarios=['vlow-plow'] if quick else list(SCENARIOS),
        demand_rates={s: scenarios()[s] for s in SCENARIOS},
        horizon_seconds=180 if quick else 2400, bootstrap_draws=0 if quick else 1000,
        seed_policy='New development seeds; reserved 58001-58020 untouched',
        selection='All frozen final checkpoints and all replicas; no training or outcome-based selection',
        intervention='early reuses continue weights: pending local request + legal PED => PED; otherwise unchanged categorical policy',
        inference='Same per-demand/per-junction categorical RNG streams as archived evaluation',
        safety='Unchanged simulator, minimum greens, yellow, all-red and occupancy clearance for every controller',
        contrasts=[['mappo-early', 'mappo-continue']],
        limitations=['Synthetic network-wide Poisson arrival rates, not calibrated Singapore counts',
            'Demand matches across controllers within a cell; changing vehicle rate changes shared-RNG pedestrian draws across cells',
            'Three trained replicas; new development analysis, not a confirmatory test',
            'Early service is a hybrid rule plus MAPPO, not a newly trained MARL algorithm or a p95 guarantee',
            'Normal walking speed only in this factorial; existing slow-walker results remain part of the project',
            '600-second empty-reset training versus 2400-second evaluation remains unchanged'],
        output_limit_bytes=512 * 1024**2, minimum_free_bytes=512 * 1024**2,
        python=platform.python_version(), torch=str(shared.torch.__version__))
    p['total_simulations'] = len(jobs(p))
    p['fingerprint'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    return p


def jobs(p):
    # Complete matched seeds across all settings early, without selecting outcomes.
    return [(s, seed, label, r)
            for seed in p['evaluation_seeds'] for s in p['evaluation_scenarios']
            for label, r in [(b, None) for b in p['baselines']] +
                [(f'mappo-{a}', r) for r in range(p['replicas']) for a in p['arms']]]


def job_name(job):
    scenario, seed, group, replica = job
    label = group if replica is None else f'{group}-r{replica}'
    return f'{label}-{scenario}-{seed}'


def load_model(p, group, replica):
    label = f'{"mappo-continue" if group == "mappo-early" else group}-r{replica}'
    info = p['models'][label]
    with zipfile.ZipFile(ARCHIVE) as z:
        raw = z.read(info['member'])
    if hashlib.sha256(raw).hexdigest() != info['sha256']:
        raise ValueError('Checkpoint hash changed')
    saved = shared.torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    parent = p['starting_protocol']
    if group == 'mappo-original':
        source = parent['starting_models']['checkpoints'][label]
        expected = (parent['starting_models']['fingerprint'], source['parent_label'], 128)
    else:
        expected = (parent['fingerprint'], label, 64)
        done = info['complete']
        original = parent['starting_models']['checkpoints'][f'mappo-original-r{replica}']['sha256']
        if (done['checkpoint_sha256'] != info['sha256'] or done['parent_sha256'] != original
                or saved['parent_sha256'] != original):
            raise ValueError('Model ancestry mismatch')
    if (saved['fingerprint'], saved['label'], saved['episodes']) != expected:
        raise ValueError('Model identity mismatch')
    model = shared.RepairNetworks.from_snapshot(saved['model'])
    if model.method != 'mappo' or model.variant != 'normalized-clock':
        raise ValueError('Unexpected architecture')
    model.observation_schema = 'legacy-v2'
    model.eval()
    return model, info['sha256']


def audit_outcome(out, early=False):
    validate_result(out['result'])
    count = Counter()
    decisions = []
    horizon = out['rollout']['horizon']
    for row in out['rollout']['trace']:
        if row['action'] is None or row['time'] >= horizon:
            continue
        if not row['mask'][row['action']]:
            raise AssertionError('Illegal action')
        count['legal_actions'] += 1
        if sum(row['mask']) <= 1:
            continue
        values = dict(zip(out['features'][row['junction']], row['observation']))
        oldest = max((v * 60 for n, v in values.items() if n.endswith('/oldest')), default=0.)
        opportunity = row['pending_count'] > 0 and bool(row['mask'][2])
        count['free_decisions'] += 1
        count['pedestrian_service_opportunities'] += opportunity
        count['pedestrian_opportunities_served'] += opportunity and row['action'] == 2
        count['empty_pedestrian_requests'] += row['pending_count'] == 0 and row['action'] == 2
        if early and opportunity and row['action'] != 2:
            raise AssertionError('Early-service intervention failed')
        decisions.append(dict(time=row['time'], junction=row['junction'], action=row['action'],
            mask=row['mask'], pending=row['pending_count'], oldest_seconds=oldest,
            observation=row['observation']))
    history = out['phase_history']
    for e in out['events']:
        if e['entered_at'] is None:
            continue
        tls = e['crossing_id'].split('_')[0][1:]
        stages = history[tls]
        # Movement at t occurs before applying the transition stamped t.
        i = max(0, bisect_left([h[0] for h in stages], e['entered_at']) - 1)
        if stages[i][1] != 'PED':
            raise AssertionError('Crossing entry outside prior admission phase')
        count['legal_crossing_entries'] += 1
        if e['reached_at'] is not None and abs(e['entered_at'] - e['reached_at'] - e['wait_seconds']) > 1e-7:
            raise AssertionError('Event wait arithmetic mismatch')
    return dict(count), decisions


def run_job(root, p, job, replay=False):
    started = time.perf_counter()
    root = Path(root)
    name = job_name(job)
    path = root / 'records' / (name + '.json')
    cached = shared.cached_json(path, p['fingerprint'])
    if cached and not replay:
        for field, directory in [('events_sha256', 'events'), ('decisions_sha256', 'decisions')]:
            if shared.digest(root / directory / (name + '.json.gz')) != cached[field]:
                raise ValueError('Cached artifact hash mismatch')
        return name
    scenario, seed, group, replica = job
    model, sha = load_model(p, group, replica) if replica is not None else (None, None)
    before = model.snapshot()['weights'] if model else None
    policy = EarlyServicePolicy(model) if group == 'mappo-early' else model
    out = shared.episode(root, p, scenario, seed, group, policy)
    if model and any(not shared.torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise AssertionError('Inference changed weights')
    audit, decisions = audit_outcome(out, group == 'mappo-early')
    record = dict(fingerprint=p['fingerprint'], scenario=scenario, seed=seed, group=group,
        replica=replica, checkpoint_sha256=sha, audit=audit,
        **shared.compact(out), crossing_completion=shared.crossing_completion(out))
    if replay:
        # Signal histories contain tuples in memory and lists after JSON storage.
        canonical = json.loads(json.dumps(record, allow_nan=False))
        changed = [k for k, value in canonical.items() if not cached or cached[k] != value]
        if changed:
            raise AssertionError('Deterministic outcome replay mismatch: ' + ', '.join(changed))
        for key, data in [('events_sha256', out['events']), ('decisions_sha256', decisions)]:
            import gzip
            raw = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if hashlib.sha256(gzip.compress(raw, mtime=0)).hexdigest() != cached[key]:
                raise AssertionError('Deterministic event/decision replay mismatch')
        return name
    record['events_sha256'] = shared.save_gzip(root / 'events' / (name + '.json.gz'), out['events'])
    record['decisions_sha256'] = shared.save_gzip(root / 'decisions' / (name + '.json.gz'), decisions)
    record['elapsed_seconds'] = time.perf_counter() - started
    shared.atomic_json(path, record)
    return name


def status(root, p, state='running', error=None):
    records = list((root / 'records').glob('*.json'))
    data = dict(updated=shared.utc(), fingerprint=p['fingerprint'], state=state,
                completed=len(records), total=p['total_simulations'], error=error)
    shared.atomic_json(root / 'status.json', data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/demand-service-test')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--verify-replay', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.status:
        print((root / 'status.json').read_text() if (root / 'status.json').exists() else 'Not started')
        return
    root.mkdir(parents=True, exist_ok=True)
    with shared.exclusive_lock(root):
        p = make_protocol(args.quick)
        shared.cached_json(root / 'protocol.json', p['fingerprint'])
        shared.atomic_json(root / 'protocol.json', p)
        for name in ('records', 'events', 'decisions'):
            (root / name).mkdir(exist_ok=True)
        if not (root / 'source.zip').exists():
            with zipfile.ZipFile(root / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                for name in p['source']:
                    z.write(ROOT / name, name)
        try:
            pending = [job for job in jobs(p) if not (root / 'records' / (job_name(job) + '.json')).exists()]
            status(root, p)
            if not args.report_only and args.workers == 1:
                for job in pending:
                    print('Completed', run_job(str(root), p, job), flush=True)
                    status(root, p)
            elif not args.report_only:
                with ProcessPoolExecutor(args.workers) as pool:
                    iterator = iter(pending)
                    futures = {}
                    for _ in range(args.workers):
                        job = next(iterator, None)
                        if job is not None:
                            futures[pool.submit(run_job, str(root), p, job)] = job
                    while futures:
                        finished, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                        for future in finished:
                            job = futures.pop(future)
                            print('Completed', future.result(), flush=True)
                            status(root, p)
                            shared.resource_check(root, p)
                            job = next(iterator, None)
                            if job is not None:
                                futures[pool.submit(run_job, str(root), p, job)] = job
            if args.verify_replay:
                selected = [j for j in jobs(p) if j[2] in ('actuated', 'mappo-early')]
                for job in selected[:2]:
                    run_job(root, p, job, replay=True)
                shared.atomic_json(root / 'replay.json', dict(fingerprint=p['fingerprint'], jobs=selected[:2], passed=True))
            from scripts.demand_service_report import report
            report(root, p)
            status(root, p, 'complete')
        except BaseException as exc:
            status(root, p, 'stopped' if isinstance(exc, InterruptedError) else 'failed', repr(exc))
            raise


if __name__ == '__main__':
    main()
