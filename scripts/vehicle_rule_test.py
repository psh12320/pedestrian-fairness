"""Stage 1: frozen-model vehicle-axis service rule (NO training).

Adds one arm to the E15 demand/early-service comparison:
  mappo-early+vehrule = continued-MAPPO weights + early-service pedestrian rule
                        + a post-clearance vehicle-axis rule that, when EW and NS
                        are both legal and PED is not being forced, requests the
                        LARGER stopped-queue axis (actuated's rule; see
                        fairness/signals.py::baseline_action 'actuated').

Hypothesis (from results/demand-service-review): the hybrid's residual vehicle
cost is concentrated in ~13% of post-clearance choices that pick the SMALLER
queue. Steering those toward the larger queue should recover vehicle delay while
keeping the early-service pedestrian protection. All physical safety is owned by
the unchanged SafeSignal; this wrapper only redirects among already-legal actions.

Reuses demand_service_test wholesale (archive loading, hashing, audit, report).
Writes to a NEW output dir with its own fingerprint; the frozen-source guard is
unchanged. Reserved seeds 58001-58020 remain untouched.

    python -m scripts.vehicle_rule_test --quick --workers 1 --verify-replay \
        --output runs/vehicle-rule-smoke
    python -m scripts.vehicle_rule_test --workers 5 --output runs/vehicle-rule-test
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path

from src.traffic_control.connected import incoming_groups
import scripts.demand_service_test as base
from scripts.demand_service_test import (ROOT, SCENARIOS, EarlyServicePolicy,
                                         load_model, audit_outcome, status)

shared = base.shared
ARMS = ('original', 'continue', 'tail', 'guided', 'early', 'early+vehrule')
# axis index in the (EW, NS, PED) action space
AXES = {'EW': 0, 'NS': 1}
# Queue features are halted-vehicle counts / 10, so one vehicle = 0.1. Treat
# differences below a fraction of one vehicle as a tie, with the SAME tolerance in
# the policy and the audit so float summation order can never make them disagree.
QUEUE_TIE = 1e-6


def _axis_preference(ew, ns):
    """Return 0 (EW) / 1 (NS) / None (tie) using the shared tolerance. Pure Python
    floats; the policy and audit both route their comparison through this."""
    if ew - ns > QUEUE_TIE:
        return 0
    if ns - ew > QUEUE_TIE:
        return 1
    return None


class VehicleAxisPolicy(EarlyServicePolicy):
    """Early-service pedestrian rule PLUS a larger-stopped-queue vehicle rule.

    Composition order matters: the pedestrian point-mass (inherited) wins whenever
    a local request is pending and PED is legal. Only when PED is NOT forced and
    both EW and NS are legal do we collapse onto the larger-queue axis. Ties and
    the single-legal-axis case fall through to the unchanged categorical policy,
    so we never override a decision the actor had no freedom in anyway.
    """
    def __init__(self, model):
        super().__init__(model)
        groups = incoming_groups(3)
        # Per junction, the observation-feature indices of each axis's queue edges.
        self.queue_indices = {}
        for junction, names in self.features.items():
            index = {name: i for i, name in enumerate(names)}
            self.queue_indices[junction] = {
                axis: [index[e + '/queue'] for e in sorted(groups[junction][axis])]
                for axis in AXES}

    def distribution(self, junction, observations, masks):
        torch = shared.torch
        dist = super().distribution(junction, observations, masks)  # pedestrian rule applied
        probs = dist.probs
        obs = torch.as_tensor(observations, dtype=probs.dtype, device=probs.device)
        legal = torch.as_tensor(masks, dtype=torch.bool, device=probs.device)
        qi = self.queue_indices[junction]
        ew = obs[..., qi['EW']].sum(dim=-1)
        ns = obs[..., qi['NS']].sum(dim=-1)
        ped_forced = probs[..., 2] >= 1.  # pedestrian point-mass already set upstream
        both_veh_legal = legal[..., 0] & legal[..., 1]
        # Larger queue beyond a one-vehicle-fraction tolerance; ties fall through to
        # the categorical policy (no gratuitous switch). Same tolerance as the audit.
        want_ew = both_veh_legal & ~ped_forced & (ew - ns > QUEUE_TIE)
        want_ns = both_veh_legal & ~ped_forced & (ns - ew > QUEUE_TIE)
        point_ew = torch.zeros_like(probs); point_ew[..., 0] = 1.
        point_ns = torch.zeros_like(probs); point_ns[..., 1] = 1.
        out = torch.where(want_ew[..., None], point_ew, probs)
        out = torch.where(want_ns[..., None], point_ns, out)
        return torch.distributions.Categorical(probs=out)


def audit_vehicle_rule(out):
    """Symmetric to the early-service assertion: when the vehicle rule must fire
    (both axes legal, pedestrian rule not firing, larger queue beyond tolerance),
    the executed action is the larger-queue axis. Uses the SAME _axis_preference
    as the policy so float summation order cannot cause a spurious failure."""
    groups = incoming_groups(3)
    horizon = out['rollout']['horizon']
    for row in out['rollout']['trace']:
        if row['action'] is None or row['time'] >= horizon or sum(row['mask']) <= 1:
            continue
        values = dict(zip(out['features'][row['junction']], row['observation']))
        # Pedestrian rule fires exactly when a local request is pending AND PED is
        # legal; those rows are the early-service path, audited separately.
        if row['pending_count'] > 0 and row['mask'][2]:
            continue
        if not (row['mask'][0] and row['mask'][1]):
            continue
        ew = sum(values[e + '/queue'] for e in groups[row['junction']]['EW'])
        ns = sum(values[e + '/queue'] for e in groups[row['junction']]['NS'])
        prefer = _axis_preference(ew, ns)
        if prefer is not None and row['action'] != prefer:
            raise AssertionError(
                f"Vehicle rule failed at t={row['time']} {row['junction']}: "
                f"EW_q={ew:.4f} NS_q={ns:.4f} prefer={prefer} action={row['action']}")


# ---- protocol/jobs: a thin re-wrap of demand_service_test with the extra arm ----

def make_protocol(quick=False):
    p = base.make_protocol(quick)
    p['arms'] = list(ARMS)
    p['schema'] = 'vehicle-axis-rule-v1'
    p['intervention'] = ('early+vehrule reuses continue weights: pending local request + legal PED => PED; '
                         'else if EW and NS both legal, request the strictly larger stopped-queue axis; '
                         'otherwise unchanged categorical policy')
    p['contrasts'] = [['mappo-early+vehrule', 'mappo-continue'],
                      ['mappo-early+vehrule', 'mappo-early'],
                      ['mappo-early', 'mappo-continue']]
    # stamp the new source file into the fingerprint
    p['source']['scripts/vehicle_rule_test.py'] = shared.digest(ROOT / 'scripts/vehicle_rule_test.py')
    p['total_simulations'] = len(jobs(p))
    p['fingerprint'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    return p


def jobs(p):
    return [(s, seed, label, r)
            for seed in p['evaluation_seeds'] for s in p['evaluation_scenarios']
            for label, r in [(b, None) for b in p['baselines']] +
                [(f'mappo-{a}', r) for r in range(p['replicas']) for a in p['arms']]]


def policy_for(p, group, replica):
    model, sha = load_model(p, 'mappo-continue' if group == 'mappo-early+vehrule'
                            else group, replica) if replica is not None else (None, None)
    if group == 'mappo-early+vehrule':
        return VehicleAxisPolicy(model), model, sha
    if group == 'mappo-early':
        return EarlyServicePolicy(model), model, sha
    return model, model, sha


def run_job(root, p, job, replay=False):
    root = Path(root)
    name = base.job_name(job)
    path = root / 'records' / (name + '.json')
    cached = shared.cached_json(path, p['fingerprint'])
    if cached and not replay:
        return name
    scenario, seed, group, replica = job
    policy, model, sha = policy_for(p, group, replica)
    before = model.snapshot()['weights'] if model else None
    out = shared.episode(root, p, scenario, seed, group, policy)
    if model and any(not shared.torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise AssertionError('Inference changed weights')
    audit, decisions = audit_outcome(out, group == 'mappo-early')
    if group == 'mappo-early+vehrule':
        audit_outcome(out, False)          # pedestrian-rule path still legal
        audit_vehicle_rule(out)            # new vehicle-rule assertion
    record = dict(fingerprint=p['fingerprint'], scenario=scenario, seed=seed, group=group,
        replica=replica, checkpoint_sha256=sha, audit=audit,
        **shared.compact(out), crossing_completion=shared.crossing_completion(out))
    if replay:
        canonical = json.loads(json.dumps(record, allow_nan=False))
        if cached and any(cached.get(k) != v for k, v in canonical.items()):
            raise AssertionError('Deterministic replay mismatch')
        return name
    record['events_sha256'] = shared.save_gzip(root / 'events' / (name + '.json.gz'), out['events'])
    record['decisions_sha256'] = shared.save_gzip(root / 'decisions' / (name + '.json.gz'), decisions)
    shared.atomic_json(path, record)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/vehicle-rule-test')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--status', action='store_true')
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
        pending = [j for j in jobs(p) if not (root / 'records' / (base.job_name(j) + '.json')).exists()]
        status(root, p)
        if args.workers == 1:
            for job in pending:
                print('Completed', run_job(str(root), p, job), flush=True)
                status(root, p)
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                it = iter(pending)
                futures = {}
                for _ in range(args.workers):
                    job = next(it, None)
                    if job is not None:
                        futures[pool.submit(run_job, str(root), p, job)] = job
                while futures:
                    done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        print('Completed', future.result(), flush=True)
                        status(root, p)
                        job = next(it, None)
                        if job is not None:
                            futures[pool.submit(run_job, str(root), p, job)] = job
        if args.verify_replay:
            for job in [j for j in jobs(p) if j[2] == 'mappo-early+vehrule'][:2]:
                run_job(root, p, job, replay=True)
        from scripts.vehicle_rule_report import report
        report(root, p)
        status(root, p, 'complete')


if __name__ == '__main__':
    main()
