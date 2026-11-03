"""Read-only protocol diagnostics; never selects a better-performing model."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def p95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None


def service_components(a, b):
    av, bv = a['vehicles'], b['vehicles']
    return {
        'vehicle_completion_worse': av['completed_count'] < bv['completed_count'],
        'vehicle_unfinished_worse': av['unfinished_count'] > bv['unfinished_count'],
        'vehicle_uninserted_worse': av['uninserted_count'] > bv['uninserted_count'],
        'person_trip_service_worse': any(a['persons'].get(k, 0) * sign > b['persons'].get(k, 0) * sign
            for k, sign in [('completed', -1), ('unfinished', 1), ('uninserted', 1)]),
        'crossing_pending_at_cutoff_worse': any(a['crossings'][c]['pending_at_cutoff'] >
            b['crossings'][c]['pending_at_cutoff'] for c in a['crossings']),
        'crossing_unresolved_at_end_worse': any(a['crossings'][c]['unresolved_after_followup'] >
            b['crossings'][c]['unresolved_after_followup'] for c in a['crossings']),
        'measurement_unknown_or_lost': any(x['unknown_cohort_entries'] or x['lost']
            for row in (a, b) for x in row['crossings'].values()),
    }


def main():
    archive = ROOT / 'runs/ppo-repair-results.zip'
    with zipfile.ZipFile(archive) as z:
        prefix = 'runs/ppo-repair-pilot/'
        read = lambda name: json.loads(z.read(prefix + name))
        protocol = read('protocol.json')
        result = {'input': str(archive), 'fingerprint': protocol['fingerprint'],
                  'training': {}, 'medium_comparisons': {}, 'validation_trajectories': {}}
        names = [n for n in z.namelist() if n.endswith('/learning.json')]
        histories = defaultdict(list)
        validations = defaultdict(lambda: defaultdict(list))
        for name in names:
            data = json.loads(z.read(name)); label = data['label'].rsplit('-r', 1)[0]
            rollouts = [r for h in data['history'] for r in h['rollouts']]
            if len(rollouts) != protocol['episodes']:
                raise AssertionError('Missing training episodes')
            histories[label].append({
                'model': data['label'], 'episodes': len(rollouts),
                'scenario_counts': dict(Counter(r['scenario'] for r in rollouts)),
                'free_decisions_per_junction': {
                    t: sum(r['diagnostics'][t]['free_decisions'] for r in rollouts)
                    for t in ('C0', 'C1', 'C2')},
                'update_batches': len(data['history']),
            })
            for row in data['validations']:
                validations[label][row['episodes']].append(row['result'])
        result['training'] = dict(histories)
        seeds = protocol['evaluation_seeds']
        baseline = {s: next(x['result'] for x in read(f'baselines/vmedium-pmedium-{s}.json')['records']
                             if x['controller'] == 'actuated') for s in seeds}
        # Validate actual generated demand across Windows/Linux serialization.
        # This audit does not rewrite the original demand generator or old hashes.
        sys.path.insert(0, str(ROOT))
        from src.traffic_control.fairness.demand import make_demand, scenarios
        config = json.loads((ROOT / 'config/fairness_v2.json').read_text())
        config.update(scenarios()['vmedium-pmedium'])
        demand_audit = []
        with TemporaryDirectory(prefix='protocol-demand-audit-', dir=ROOT / 'runs') as tmp:
            for seed in seeds:
                path = Path(tmp) / 'demand.rou.xml'
                generated = make_demand(path, seed, config, 3)
                normalized_hash = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                if normalized_hash != baseline[seed]['demand_sha256']:
                    raise AssertionError('Semantic demand replay differs after newline normalization')
                cohort = [v for v in generated['vehicles'] if 300 <= v['depart'] < 2100]
                demand_audit.append({'seed': seed, 'cohort_vehicles': len(cohort),
                    'three_junction_routes': sum(len(v['route']) == 4 for v in cohort),
                    'raw_hash_matches_cluster': generated['sha256'] == baseline[seed]['demand_sha256'],
                    'lf_hash_matches_cluster': True})
        result['demand_replay'] = demand_audit
        result['corridor_route_fraction'] = sum(x['three_junction_routes'] for x in demand_audit) / sum(
            x['cohort_vehicles'] for x in demand_audit)
        for label in sorted(histories):
            pairs = []
            models = []
            for replica in range(protocol['replicas']):
                rows = []
                for seed in seeds:
                    row = read(f'evaluation/{label}-r{replica}-vmedium-pmedium-{seed}.json')['result']
                    if row['demand_sha256'] != baseline[seed]['demand_sha256']:
                        raise AssertionError('Unmatched demand')
                    for data in (row, baseline[seed]):
                        v = data['vehicles']
                        assert v['scheduled_count'] == sum(v[k] for k in
                            ('completed_count', 'unfinished_count', 'uninserted_count'))
                        for c in data['crossings'].values():
                            assert c['entries'] == len(c['waits']) and c['p95'] == p95(c['waits'])
                    pairs.append(service_components(row, baseline[seed])); rows.append(row)
                models.append(rows)
            delta = {c: st.mean(p95([w for row in rows for w in row['crossings'][c]['waits']])
                                  for rows in models) - p95([w for s in seeds for w in baseline[s]['crossings'][c]['waits']])
                     for c in baseline[seeds[0]]['crossings']}
            result['medium_comparisons'][label] = {
                'pairs': len(pairs), 'failure_component_pairs': dict(Counter(k for row in pairs for k,v in row.items() if v)),
                'failure_pairs': sum(any(row.values()) for row in pairs),
                'cutoff_only_failure_pairs': sum(row['crossing_pending_at_cutoff_worse'] and
                    not any(v for k,v in row.items() if k != 'crossing_pending_at_cutoff_worse') for row in pairs),
                'vehicle_delta': st.mean(row['vehicles']['completed_mean_time_loss_seconds'] -
                    baseline[seed]['vehicles']['completed_mean_time_loss_seconds']
                    for rows in models for seed,row in zip(seeds, rows)),
                'crossing_p95_deltas': delta,
                'crossings_no_worse': sum(d <= 0 for d in delta.values()),
            }
            result['validation_trajectories'][label] = {ep: {
                'vehicle_loss': st.mean(row['vehicles']['completed_mean_time_loss_seconds'] for row in rows),
                'completed_vehicles': st.mean(row['vehicles']['completed_count'] for row in rows),
                'pending_crossings': st.mean(sum(c['pending_at_cutoff'] for c in row['crossings'].values()) for row in rows),
                'mean_crossing_run_p95_DIAGNOSTIC_ONLY': st.mean(c['p95'] for row in rows for c in row['crossings'].values()),
            } for ep,rows in validations[label].items()}
    local = ROOT / 'runs/local-timing-study/records'
    fixed = {}
    for path in local.glob('*fixed.json.gz'):
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        if data['scenario'] in ('vmedium-pmedium', 'pedestrian-burst', 'slow-walkers'):
            fixed[(data['scenario'], data['seed'], data['admission_seconds'])] = data['result']['vehicles']
    if fixed:
        result['fixed_vehicle_invariance_checks'] = {
            scenario: all(value == fixed[('vmedium-pmedium', seed, duration)]
                for (s, seed, duration), value in fixed.items() if s == scenario)
            for scenario in ('pedestrian-burst', 'slow-walkers')}
    destination = ROOT / 'runs/deep-protocol-audit.json'
    destination.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(destination)
    for key,row in result['medium_comparisons'].items():
        print(key, json.dumps({k:v for k,v in row.items() if k != 'crossing_p95_deltas'}))
    for key,rows in result['training'].items():
        free = [n for row in rows for n in row['free_decisions_per_junction'].values()]
        print(key, 'free decisions/actor: min/mean/max', min(free), round(st.mean(free),1), max(free))


if __name__ == '__main__':
    main()
