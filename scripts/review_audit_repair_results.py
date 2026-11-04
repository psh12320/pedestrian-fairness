"""Independent, standard-library checks of the completed local learning audit.

Reads existing results only. Does not load checkpoints, import the learner or
rerun SUMO. Bootstrap intervals remain those of the archived reporting code.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None


def close(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b) < 1e-7), (a, b)


def main():
    root = ROOT / 'runs/audit-repair-trial'
    p, s, status = (read(root / n) for n in ['protocol.json', 'summary.json', 'status.json'])
    assert status['state'] == 'complete' and status['completed_simulations'] == 1396
    assert s['fingerprint'] == status['fingerprint'] == p['fingerprint']
    checks = dict(source_files=0, models=0, training_episodes=0, validations=0,
                  traffic_runs=0, crossing_event_files=0, crossing_rows=0, aggregate_rows=0)
    with zipfile.ZipFile(root / 'source.zip') as z:
        for name, expected in p['source'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
            assert sha(ROOT / name) == expected
            checks['source_files'] += 1
    training, validation_trends = {}, {}
    model_hashes = {}
    for method in p['methods']:
        for arm in p['variants']:
            label = f'{method}-{arm}'
            validation_trends[label] = {e: [] for e in p['validation_episodes']}
            for r in range(p['replicas']):
                full = f'{label}-r{r}'; folder = root / 'models' / full
                done, curve = read(folder / 'complete.json'), read(folder / 'learning.json')
                assert done['fingerprint'] == curve['fingerprint'] == p['fingerprint']
                assert done['episodes'] == p['episodes'] == 64
                assert sha(folder / 'final.pt') == done['checkpoint_sha256']
                model_hashes[full] = done['checkpoint_sha256']
                rollouts = [v for h in curve['history'] for v in h['rollouts']]
                expected = [(i+1, p['training_scenarios'][i % len(p['training_scenarios'])],
                             p['training_seed_base'] + r*p['training_seed_stride'] + i) for i in range(64)]
                assert [(v['episode'],v['scenario'],v['seed']) for v in rollouts] == expected
                identity = [(v['demand_sha256'],v['network_sha256']) for v in rollouts]
                assert training.setdefault(r, identity) == identity
                assert len(curve['validations']) == 3
                for v in curve['validations']:
                    validation_trends[label][v['episodes']].append(v['result']['vehicles']['completed_mean_time_loss_seconds'])
                checks['models'] += 1; checks['training_episodes'] += len(rollouts)
                checks['validations'] += len(curve['validations'])
    seen, records, anomalies = set(), {}, []

    def verify(row, label, scenario, seed):
        key = (label, scenario, seed)
        if key in seen:
            return
        seen.add(key); checks['traffic_runs'] += 1
        result = row['result']; v = result['vehicles']; person = result['persons']
        for a in [v, row['vehicles_end']]:
            assert a['scheduled_count'] == sum(a[k] for k in ['completed_count','unfinished_count','uninserted_count'])
        assert person['scheduled'] == sum(person[k] for k in ['completed','unfinished','uninserted'])
        close(v['completed_mean_time_loss_seconds'] + v['completed_mean_depart_delay_seconds'],
              v['completed_mean_time_loss_plus_depart_delay_seconds'])
        for diag in row['diagnostics'].values():
            close(sum(diag['phase_seconds'].values()), 2400)
        filename = f'baseline-{label}-{scenario}-{seed}' if label in p['baselines'] else f'{label}-{scenario}-{seed}'
        events_path = root / 'events' / (filename + '.json.gz')
        assert sha(events_path) == row['events_sha256']
        events = json.loads(gzip.decompress(events_path.read_bytes()))
        checks['crossing_event_files'] += 1
        assert len(result['crossings']) == 12
        for c, data in result['crossings'].items():
            visits = [e for e in events if e['crossing_id']==c and e['reached_at'] is not None and 300<=e['reached_at']<2100]
            waits = [e['entered_at']-e['reached_at'] for e in visits if e['entered_at'] is not None]
            assert sorted(waits) == sorted(data['waits'])
            assert len(waits) == data['entries']
            close(tail(waits), data['p95'])
            close(st.mean(waits) if waits else None, data['mean'])
            assert sum(e['entered_at'] is None for e in visits) == data['unresolved_after_followup']
            assert sum(e['entered_at'] is None or e['entered_at']>=2100 for e in visits) == data['pending_at_cutoff']
            if data['lost'] or data['unknown_cohort_entries'] or data['unresolved_after_followup']:
                anomalies.append([label,scenario,seed,c,data['lost'],data['unknown_cohort_entries'],data['unresolved_after_followup']])
            checks['crossing_rows'] += 1
        records[key] = row

    for path in (root / 'baselines').glob('*.json'):
        group = read(path); assert group['fingerprint'] == p['fingerprint']
        for row in group['records']:
            verify(row,row['controller'],group['scenario'],group['seed'])
    for path in (root / 'evaluation').glob('*.json'):
        row = read(path); assert row['fingerprint'] == p['fingerprint']
        assert row['checkpoint_sha256'] == model_hashes[row['label']]
        ref = records['actuated',row['scenario'],row['seed']]
        assert (row['network_sha256'],row['result']['demand_sha256']) == (ref['network_sha256'],ref['result']['demand_sha256'])
        verify(row,row['label'],row['scenario'],row['seed'])
    summary_rows = {}
    for scenario, data in s['scenarios'].items():
        summary_rows[scenario] = {}
        for group in ['baselines','learned']:
            for label, rec in data[group].items():
                matrix = [[records[(f'{label}-r{r}' if group=='learned' else label),scenario,seed]['result']
                           for seed in p['evaluation_seeds']] for r in range(p['replicas'] if group=='learned' else 1)]
                rows = [x for replica in matrix for x in replica]
                a = rec['absolute']
                close(st.mean(x['vehicles']['completed_mean_time_loss_seconds'] for x in rows),a['vehicle_mean_time_loss'])
                for short, field in [('scheduled','scheduled_count'),('completed','completed_count'),('unfinished','unfinished_count'),('uninserted','uninserted_count')]:
                    close(st.mean(x['vehicles'][field] for x in rows), a[f'vehicle_{short}_count'])
                deltas = {}
                for c, value in a['crossings'].items():
                    waits = [[w for r in replica for w in r['crossings'][c]['waits']] for replica in matrix]
                    close(st.mean(tail(w) for w in waits),value['mean_replica_pooled_p95'])
                    close(st.mean(st.mean(w) for w in waits),value['mean_wait'])
                    assert [len(w) for w in waits] == value['entries_per_replica']
                    ref = [w for seed in p['evaluation_seeds'] for w in records['actuated',scenario,seed]['result']['crossings'][c]['waits']]
                    deltas[c] = st.mean(tail(w) for w in waits) - tail(ref)
                comparison = rec.get('versus_actuated',rec.get('comparisons',{}).get('actuated'))
                for c,v in deltas.items(): close(v,comparison['point']['crossing_p95_delta_seconds'][c])
                no_worse = sum(v<=1e-9 for v in deltas.values())
                summary_rows[scenario][label] = dict(vehicle_loss=a['vehicle_mean_time_loss'],completed=a['vehicle_completed_count'],
                    uninserted=a['vehicle_uninserted_count'],worst_crossing_delta=max(deltas.values()),crossings_no_worse=no_worse,
                    descriptive_pass=comparison['descriptive_pass'])
                checks['aggregate_rows'] += 1
    output = dict(fingerprint=p['fingerprint'],checks=checks,measurement_anomalies=anomalies,scenarios=summary_rows,
        validation_vehicle_loss={label:{e:st.mean(values) for e,values in by_episode.items()} for label,by_episode in validation_trends.items()},
        limitations=['No fresh SUMO replay or checkpoint deserialization.','Bootstrap intervals inspected in code, not independently regenerated.',
                     'These are two-replica development results, not a final method ranking.'])
    dest = ROOT / 'results/audit-repair-review';dest.mkdir(parents=True,exist_ok=True)
    (dest / 'audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2));print('Measurement anomalies:',len(anomalies))


if __name__ == '__main__':
    main()
