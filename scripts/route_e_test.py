"""Route E: unified service rule (frozen model, NO training).

mappo-unified = continued-MAPPO weights
              + larger-stopped-queue vehicle rule (Stage 1)
              + per-crossing pedestrian DEADLINE: if any local crossing's oldest
                pending request exceeds that crossing's threshold AND PED is legal,
                force PED -- overriding both the model and the vehicle-axis rule.

Measured headroom (early+vehrule, med/med): PED is legal-but-unserved in ~56
decisions/run. The deadline targets exactly those for crossings nearing their
actuated p95, to protect the tails the rule alone leaves at 6/12.

Thresholds default to each crossing's actuated p95 (so "no worse than actuated" is
the literal trigger). Per-crossing /oldest feature is in the observation
(value*60 = seconds). All physical safety unchanged; the rule only redirects among
already-legal actions (PED must be in the mask to be forced).

    python -m scripts.route_e_test --quick --output runs/route-e-smoke
    python -m scripts.route_e_test --workers 5 --output runs/route-e
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path

from src.traffic_control.connected import incoming_groups
import scripts.demand_service_test as base
import scripts.vehicle_rule_test as vr
from scripts.demand_service_test import ROOT, load_model, audit_outcome, status

shared = base.shared
ARMS = ('original', 'continue', 'tail', 'guided', 'early', 'early+vehrule', 'unified')
QUEUE_TIE = vr.QUEUE_TIE


class UnifiedPolicy(vr.VehicleAxisPolicy):
    """VehicleAxisPolicy + per-crossing pedestrian deadline override.

    Deadline wins over everything: if a local crossing's oldest pending wait
    (observation '<c>/oldest' * 60 seconds) exceeds thresholds[c] and PED is legal,
    put a point mass on PED. Otherwise fall through to the vehicle-axis + early
    pedestrian rule of the parent.
    """
    def __init__(self, model, thresholds):
        super().__init__(model)
        self.thresholds = thresholds
        self.oldest_idx = {}
        for junction, names in self.features.items():
            index = {n: i for i, n in enumerate(names)}
            self.oldest_idx[junction] = [
                (c, index[c + '/oldest'])
                for c in sorted(thresholds) if c.startswith(':' + junction + '_c')
                and c + '/oldest' in index]

    def distribution(self, junction, observations, masks):
        torch = shared.torch
        dist = super().distribution(junction, observations, masks)  # vehicle + early rules
        probs = dist.probs
        obs = torch.as_tensor(observations, dtype=probs.dtype, device=probs.device)
        legal = torch.as_tensor(masks, dtype=torch.bool, device=probs.device)
        overdue = torch.zeros(probs.shape[:-1], dtype=torch.bool, device=probs.device)
        for crossing, idx in self.oldest_idx[junction]:
            seconds = obs[..., idx] * 60.0
            overdue = overdue | (seconds > self.thresholds[crossing])
        force_ped = overdue & legal[..., 2]
        point = torch.zeros_like(probs)
        point[..., 2] = 1.0
        return torch.distributions.Categorical(probs=torch.where(force_ped[..., None], point, probs))


def audit_unified(out, thresholds):
    """When a crossing is overdue (oldest > threshold) and PED is legal, the executed
    action must be PED. Symmetric to the Stage-1 audits; fails loudly on a bug."""
    horizon = out['rollout']['horizon']
    for row in out['rollout']['trace']:
        if row['action'] is None or row['time'] >= horizon or sum(row['mask']) <= 1:
            continue
        if not row['mask'][2]:
            continue
        values = dict(zip(out['features'][row['junction']], row['observation']))
        overdue = any(values.get(c + '/oldest', 0) * 60.0 > thresholds[c]
                      for c in thresholds if c.startswith(':' + row['junction'] + '_c'))
        if overdue and row['action'] != 2:
            raise AssertionError(
                f"Unified rule failed at t={row['time']} {row['junction']}: "
                f"overdue crossing but action={row['action']} (expected PED)")


def actuated_thresholds(root, p):
    """Per-crossing threshold = actuated p95 on vmedium-pmedium, pooled over the
    evaluation seeds. Fallback 20 s where a crossing has no p95."""
    from src.traffic_control.fairness.analysis import quantile
    cache = Path(root) / 'actuated_thresholds.json'
    if cache.exists():
        return json.loads(cache.read_text())
    pooled = {}
    for seed in p['evaluation_seeds']:
        out = shared.episode(Path(root), p, 'vmedium-pmedium', seed, 'actuated', None)
        for c, d in out['result']['crossings'].items():
            pooled.setdefault(c, []).extend(d['waits'])
    thr = {c: (quantile(v) if v else 20.0) or 20.0 for c, v in pooled.items()}
    cache.write_text(json.dumps(thr, indent=2))
    return thr


def make_protocol(quick=False):
    p = base.make_protocol(quick)
    p['arms'] = list(ARMS)
    p['schema'] = 'route-e-unified-v1'
    p['intervention'] = ('unified = continue weights + larger-queue vehicle rule + '
                         'per-crossing PED deadline (oldest > actuated p95 and PED legal => PED)')
    p['contrasts'] = [['mappo-unified', 'mappo-early+vehrule'],
                      ['mappo-unified', 'mappo-continue']]
    p['source']['scripts/route_e_test.py'] = shared.digest(ROOT / 'scripts/route_e_test.py')
    p['source']['scripts/vehicle_rule_test.py'] = shared.digest(ROOT / 'scripts/vehicle_rule_test.py')
    p['total_simulations'] = len(jobs(p))
    p['fingerprint'] = hashlib.sha256(json.dumps(
        {k: v for k, v in p.items() if k != 'thresholds'}, sort_keys=True).encode()).hexdigest()
    return p


def jobs(p):
    return [(s, seed, label, r)
            for seed in p['evaluation_seeds'] for s in p['evaluation_scenarios']
            for label, r in [(b, None) for b in p['baselines']] +
                [(f'mappo-{a}', r) for r in range(p['replicas']) for a in p['arms']]]


def policy_for(p, group, replica, thresholds):
    if replica is None:
        return None, None, None
    base_group = 'mappo-continue' if group in ('mappo-early', 'mappo-early+vehrule', 'mappo-unified') else group
    model, sha = load_model(p, base_group, replica)
    if group == 'mappo-unified':
        return UnifiedPolicy(model, thresholds), model, sha
    if group == 'mappo-early+vehrule':
        return vr.VehicleAxisPolicy(model), model, sha
    if group == 'mappo-early':
        return vr.EarlyServicePolicy(model), model, sha
    return model, model, sha


def run_job(root, p, job, thresholds):
    root = Path(root)
    name = base.job_name(job)
    path = root / 'records' / (name + '.json')
    cached = shared.cached_json(path, p['fingerprint'])
    if cached:
        return name
    scenario, seed, group, replica = job
    policy, model, sha = policy_for(p, group, replica, thresholds)
    before = model.snapshot()['weights'] if model else None
    out = shared.episode(root, p, scenario, seed, group, policy)
    if model and any(not shared.torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise AssertionError('Inference changed weights')
    audit, decisions = audit_outcome(out, group == 'mappo-early')
    if group == 'mappo-early+vehrule':
        audit_outcome(out, False); vr.audit_vehicle_rule(out)
    if group == 'mappo-unified':
        audit_unified(out, thresholds)
    record = dict(fingerprint=p['fingerprint'], scenario=scenario, seed=seed, group=group,
        replica=replica, checkpoint_sha256=sha, audit=audit,
        **shared.compact(out), crossing_completion=shared.crossing_completion(out))
    record['events_sha256'] = shared.save_gzip(root / 'events' / (name + '.json.gz'), out['events'])
    record['decisions_sha256'] = shared.save_gzip(root / 'decisions' / (name + '.json.gz'), decisions)
    shared.atomic_json(path, record)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'runs/route-e')
    ap.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with shared.exclusive_lock(root):
        p = make_protocol(args.quick)
        shared.cached_json(root / 'protocol.json', p['fingerprint'])
        thresholds = actuated_thresholds(root, p)
        p['thresholds'] = thresholds
        shared.atomic_json(root / 'protocol.json', p)
        for name in ('records', 'events', 'decisions'):
            (root / name).mkdir(exist_ok=True)
        pending = [j for j in jobs(p) if not (root / 'records' / (base.job_name(j) + '.json')).exists()]
        status(root, p)
        if args.workers == 1:
            for job in pending:
                print('Completed', run_job(str(root), p, job, thresholds), flush=True)
                status(root, p)
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                it = iter(pending); futures = {}
                for _ in range(args.workers):
                    j = next(it, None)
                    if j is not None:
                        futures[pool.submit(run_job, str(root), p, j, thresholds)] = j
                while futures:
                    done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                    for f in done:
                        futures.pop(f); print('Completed', f.result(), flush=True); status(root, p)
                        j = next(it, None)
                        if j is not None:
                            futures[pool.submit(run_job, str(root), p, j, thresholds)] = j
        from scripts.vehicle_rule_report import report
        report(root, p)
        status(root, p, 'complete')


if __name__ == '__main__':
    main()
