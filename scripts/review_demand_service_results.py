"""Independent saved-artifact audit; no Torch, simulation, or source mutation."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/demand-service-test'
DEST = ROOT / 'results/demand-service-review'
ORDER = ['vlow-plow', 'vlow-pmedium', 'vmedium-plow', 'vmedium-pmedium']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def q(values, probability=.95):
    return sorted(values)[max(0, math.ceil(probability * len(values)) - 1)] if values else None


def same(a, b, path=''):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) <= set(b), path
        for k, v in a.items():
            same(v, b[k], path + '/' + k)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + '/' + str(i))
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7), (path, a, b)
    else:
        assert a == b, (path, a, b)


def component_flags(a, b):
    x, y = a['vehicles'], b['vehicles']
    p, r = a['persons'], b['persons']
    return {
        'fewer_completed_vehicles': x['completed_count'] < y['completed_count'],
        'more_unfinished_vehicles': x['unfinished_count'] > y['unfinished_count'],
        'more_uninserted_vehicles': x['uninserted_count'] > y['uninserted_count'],
        'more_crossing_requests_pending_at_cutoff': any(a['crossings'][c]['pending_at_cutoff'] > b['crossings'][c]['pending_at_cutoff'] for c in a['crossings']),
        'more_crossing_requests_unresolved_after_followup': any(a['crossings'][c]['unresolved_after_followup'] > b['crossings'][c]['unresolved_after_followup'] for c in a['crossings']),
        'unknown_or_lost_measurements': any(v['lost'] or v['unknown_cohort_entries'] for z in (a, b) for v in z['crossings'].values()),
        'worse_person_trip_service': p['completed'] < r['completed'] or p['unfinished'] > r['unfinished'] or p['uninserted'] > r['uninserted'],
    }


def verify_comparison(left, right, stored, checks):
    nr, ns = len(left), len(left[0])
    pairs = [(left[r][s]['result'], right[r][s]['result']) for r in range(nr) for s in range(ns)]
    tails = {c: st.mean(q([w for record in left[r] for w in record['result']['crossings'][c]['waits']])
                       - q([w for record in right[r] for w in record['result']['crossings'][c]['waits']]) for r in range(nr))
             for c in left[0][0]['result']['crossings']}
    delta = st.mean(a['vehicles']['completed_mean_time_loss_seconds'] - b['vehicles']['completed_mean_time_loss_seconds'] for a, b in pairs)
    same(dict(vehicle_delta_seconds=delta, crossing_p95_delta_seconds=tails,
              worst_crossing_delta_seconds=max(tails.values()), crossings_no_worse=sum(v <= 1e-9 for v in tails.values())), stored['point'])
    flags = [component_flags(a, b) for a, b in pairs]
    same(sum(any(f.values()) for f in flags), stored['service_failure_pairs'])
    same({k: sum(f[k] for f in flags) for k in flags[0]}, stored['service_failure_components'])
    same(sum(f['more_crossing_requests_pending_at_cutoff'] and not any(v for k, v in f.items() if k != 'more_crossing_requests_pending_at_cutoff') for f in flags), stored['cutoff_only_failure_pairs'])
    minimum = min(sum(x['result']['crossings'][c]['entries'] for x in row) for m in (left, right) for row in m for c in tails)
    same(minimum, stored['minimum_pooled_crossing_entries'])
    same(minimum < 100, stored['sparse'])
    same(delta < 0 and all(v <= 1e-9 for v in tails.values()) and not any(any(f.values()) for f in flags), stored['descriptive_pass'])
    for c, point in tails.items():
        same(point, stored['crossings'][c]['p95_delta_seconds'])
    if stored['bootstrap_requested_draws']:
        rng = random.Random(424693)
        values = []
        for _ in range(stored['bootstrap_requested_draws']):
            rs, ss = rng.choices(range(nr), k=nr), rng.choices(range(ns), k=ns)
            values.append(st.mean(left[r][s]['result']['vehicles']['completed_mean_time_loss_seconds']
                                 - right[r][s]['result']['vehicles']['completed_mean_time_loss_seconds'] for r in rs for s in ss))
        same([q(values, .025), q(values, .975)], stored['exploratory_intervals']['vehicle_delta_seconds'])
        checks['vehicle_bootstrap_intervals_reproduced'] += 1
    checks['paired_point_and_service_comparisons'] += 1


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    p = json.loads((RUN / 'protocol.json').read_text())
    summary = json.loads((RUN / 'summary.json').read_text())
    status = json.loads((RUN / 'status.json').read_text())
    assert p == summary['protocol']
    assert status['state'] == 'complete' and status['completed'] == status['total'] == 720
    assert summary['coverage'] == {'complete': True, 'simulations': 720}
    identity = dict(p); identity.pop('fingerprint')
    assert sha(json.dumps(identity, sort_keys=True).encode()) == p['fingerprint'] == status['fingerprint']
    checks = Counter()
    with zipfile.ZipFile(RUN / 'source.zip') as z:
        assert set(z.namelist()) == set(p['source'])
        for name, expected in p['source'].items():
            assert sha(z.read(name)) == sha((ROOT / name).read_bytes()) == expected
            checks['frozen_source_files'] += 1
    archive = ROOT / 'runs/pedestrian-finetune-cluster-results.zip'
    assert sha(archive.read_bytes()) == p['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        for label, info in p['models'].items():
            assert sha(z.read(info['member'])) == info['sha256']
            checks['model_hashes'] += 1
    expected = {(scenario, seed, label, r)
                for scenario in p['evaluation_scenarios'] for seed in p['evaluation_seeds']
                for label, r in [(b, None) for b in p['baselines']] + [(f'mappo-{a}', r) for a in p['arms'] for r in range(3)]}
    assert len(expected) == 720 and not set(p['evaluation_seeds']) & set(range(58001, 58021))
    records, demands, mechanism = {}, {}, defaultdict(Counter)
    incomplete, errors = [], Counter()
    for file in sorted((RUN / 'records').glob('*.json')):
        row = json.loads(file.read_text())
        key = row['scenario'], row['seed'], row['group'], row['replica']
        assert key in expected and key not in records
        assert row['fingerprint'] == p['fingerprint']
        pair = row['result']['demand_sha256'], row['network_sha256']
        assert demands.setdefault(key[:2], pair) == pair
        if row['replica'] is not None:
            label = ('mappo-continue' if row['group'] == 'mappo-early' else row['group']) + f'-r{row["replica"]}'
            assert row['checkpoint_sha256'] == p['models'][label]['sha256']
        data = {}
        for directory, field in [('events', 'events_sha256'), ('decisions', 'decisions_sha256')]:
            raw = (RUN / directory / (file.stem + '.json.gz')).read_bytes()
            assert sha(raw) == row[field]
            data[directory] = json.loads(gzip.decompress(raw))
            checks['compressed_artifact_hashes'] += 1
        v, person = row['result']['vehicles'], row['result']['persons']
        assert v['scheduled_count'] == sum(v[k] for k in ('completed_count', 'unfinished_count', 'uninserted_count'))
        assert person['scheduled'] == person['completed'] + person['unfinished'] + person['uninserted']
        same(v['completed_mean_time_loss_seconds'] + v['completed_mean_depart_delay_seconds'], v['completed_mean_time_loss_plus_depart_delay_seconds'])
        ve = row['vehicles_end']
        assert ve['scheduled_count'] == ve['completed_count'] + ve['unfinished_count'] + ve['uninserted_count']
        if ve['unfinished_count'] or ve['uninserted_count'] or person['unfinished'] or person['uninserted']:
            incomplete.append(dict(run=file.stem, vehicles=ve, persons=person))
        by_cross = defaultdict(list)
        entries = defaultdict(list)
        for event in data['events']:
            by_cross[event['crossing_id']].append(event)
            if event['entered_at'] is None:
                continue
            tls = event['crossing_id'].split('_')[0][1:]
            entries[tls].append(event['entered_at'])
            history = row['phase_history'][tls]
            index = max(0, bisect_left([h[0] for h in history], event['entered_at']) - 1)
            assert history[index][1] == 'PED'
            if event['reached_at'] is not None:
                same(event['entered_at'] - event['reached_at'], event['wait_seconds'])
            checks['legal_crossing_entries'] += 1
        for tls in entries:
            entries[tls].sort()
        for crossing, value in row['result']['crossings'].items():
            cohort = [e for e in by_cross[crossing] if e['reached_at'] is not None and 300 <= e['reached_at'] < 2100]
            waits = [e['entered_at'] - e['reached_at'] for e in cohort if e['entered_at'] is not None]
            same(waits, value['waits']); same(q(waits), value['p95']); same(st.mean(waits), value['mean'])
            same(len(waits), value['entries'])
            same(sum(e['entered_at'] is None or e['entered_at'] >= 2100 for e in cohort), value['pending_at_cutoff'])
            same(sum(e['entered_at'] is None for e in cohort), value['unresolved_after_followup'])
            errors.update({k: value[k] for k in ('lost', 'unknown_cohort_entries', 'unresolved_after_followup')})
            same(sum(e['entered_at'] is not None and e['exited_at'] is None for e in by_cross[crossing]),
                 row['crossing_completion'][crossing]['entered_without_observed_exit'])
            checks['crossing_records'] += 1
        # Completed phase durations and sequence. Tail phase is right-censored.
        mech = mechanism[row['scenario'], row['group']]
        for tls, history in row['phase_history'].items():
            for (start, phase), (end, _) in zip(history, history[1:] + [[2400., 'END']]):
                mech['measurement_phase_' + phase] += max(0., min(end, 2100.) - max(start, 300.))
            for (start, phase), (end, following) in zip(history, history[1:]):
                minimum = 12 if phase in ('EW', 'NS', 'PED') else 3 if phase.endswith('_YELLOW') else 2 if phase == 'ALL_RED' else 0
                assert end - start + 1e-7 >= minimum
                if phase in ('EW', 'NS'):
                    assert following == phase + '_YELLOW'
                if phase.endswith('_YELLOW'):
                    assert following == 'ALL_RED'
                if phase == 'PED':
                    assert following == 'PED_CLEARANCE'
                if phase == 'PED_CLEARANCE':
                    assert following == 'READY'
                checks['completed_phase_transitions'] += 1
        counter = Counter()
        mech['runs'] += 1
        for decision in data['decisions']:
            assert decision['mask'][decision['action']] and sum(decision['mask']) > 1
            values = dict(zip(row['features'][decision['junction']], decision['observation']))
            pending = sum(round(v * 5) for n, v in values.items() if n.startswith(':') and n.endswith('/count'))
            assert pending == decision['pending']
            opportunity = pending > 0 and decision['mask'][2]
            window = 'warmup' if decision['time'] < 300 else 'measurement' if decision['time'] < 2100 else 'followup'
            mech[window + '_empty_pedestrian_requests'] += pending == 0 and decision['action'] == 2
            mech[window + '_pedestrian_service_opportunities'] += opportunity
            mech[window + '_pedestrian_opportunities_served'] += opportunity and decision['action'] == 2
            counter['free_decisions'] += 1
            counter['pedestrian_service_opportunities'] += opportunity
            counter['pedestrian_opportunities_served'] += opportunity and decision['action'] == 2
            counter['empty_pedestrian_requests'] += pending == 0 and decision['action'] == 2
            if row['group'] == 'mappo-early' and opportunity:
                assert decision['action'] == 2
            checks['legal_free_decisions'] += 1
            if decision['action'] == 2 and pending == 0:
                tls = decision['junction']
                history = row['phase_history'][tls]
                index = next((i for i, (t, stage) in enumerate(history) if t >= decision['time'] and stage == 'PED'), None)
                if index is None or index + 1 >= len(history):
                    mech['empty_request_censored_admission'] += 1
                    continue
                start, end = history[index][0], history[index+1][0]
                assert history[index+1][1] == 'PED_CLEARANCE'
                assert start <= decision['time'] + 5.01  # Same frozen yellow/all-red.
                arrivals = bisect_right(entries[tls], end) - bisect_right(entries[tls], start)
                mech['empty_request_complete_admissions'] += 1
                mech['empty_request_admissions_with_zero_entries'] += arrivals == 0
                mech['entries_during_empty_request_admissions'] += arrivals
                mech[window + '_empty_request_complete_admissions'] += 1
                mech[window + '_empty_request_admissions_with_zero_entries'] += arrivals == 0
                if window == 'measurement' and end <= 2100:
                    mech['measurement_fully_observed_empty_admissions'] += 1
                    mech['measurement_fully_observed_empty_admissions_zero_entries'] += arrivals == 0
        same(dict(counter), row['audit'])
        mech.update(counter)
        checks['records'] += 1
        records[key] = row
    assert set(records) == expected
    assert not any(errors.values())
    same(checks['legal_crossing_entries'], summary['audit']['legal_crossing_entries'])
    comparisons = []
    for scenario in ORDER:
        data = summary['scenarios'][scenario]
        matrices = {b: [[records[scenario, seed, b, None] for seed in p['evaluation_seeds']]] for b in p['baselines']}
        matrices.update({f'mappo-{a}': [[records[scenario, seed, f'mappo-{a}', r] for seed in p['evaluation_seeds']]
                                     for r in range(3)] for a in p['arms']})
        for group, matrix in matrices.items():
            stored = data['baselines'].get(group) or data['learned'][group]
            flat = [row for replica in matrix for row in replica]
            a = stored['absolute']
            same(st.mean(r['result']['vehicles']['completed_mean_time_loss_seconds'] for r in flat), a['vehicle_mean_time_loss'])
            for key in ('scheduled_count', 'completed_count', 'unfinished_count', 'uninserted_count'):
                same(st.mean(r['result']['vehicles'][key] for r in flat), a['vehicle_' + key])
                same(st.mean(r['vehicles_end'][key] for r in flat), stored['vehicles_end'][key])
            for c, value in a['crossings'].items():
                samples = [[w for record in replica for w in record['result']['crossings'][c]['waits']] for replica in matrix]
                same([len(w) for w in samples], value['entries_per_replica'])
                same([q(w) for w in samples], value['p95_per_replica'])
                same(st.mean(q(w) for w in samples), value['mean_replica_pooled_p95'])
                same(st.mean(st.mean(w) for w in samples), value['mean_wait'])
            if group in p['baselines']:
                comparisons.append((matrix, matrices['actuated'], stored['versus_actuated']))
            else:
                for reference, c in stored['comparisons'].items():
                    ref = matrices[reference]
                    comparisons.append((matrix, ref * len(matrix) if reference in p['baselines'] else ref, c))
            checks['aggregate_rows'] += 1
        comparisons.append((matrices['mappo-early'], matrices['mappo-continue'], data['paired_contrasts']['mappo-early-vs-mappo-continue']))
    for left, right, stored in comparisons:
        verify_comparison(left, right, stored, checks)
    output = dict(fingerprint=p['fingerprint'], checks=dict(checks), measurement_errors=dict(errors),
        incomplete_whole_run_trips=incomplete, completed_at=status['updated'],
        mechanism={s: {g: dict(mechanism[s, g]) for g in p['baselines'] + ['mappo-' + a for a in p['arms']]} for s in ORDER},
        limitations=['Stored event/decision audit, not a fresh SUMO replay',
            'All comparison point estimates, service components and vehicle bootstrap intervals independently reproduced; crossing bootstrap intervals retained from frozen report, not independently regenerated',
            'Phase audit covers minimum green, yellow/all-red, admission duration and order plus observed legal crossing entry; clearance occupancy/physical conflict checks rely on the unchanged runtime safety implementation',
            'An empty request is empty when chosen; a later pedestrian may use that admission. Zero-entry counts exclude horizon-censored admissions. No counterfactual vehicle-time saving is inferred.'])
    (DEST / 'audit.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(dict(checks=checks, measurement_errors=errors, incomplete_whole_run_trips=len(incomplete)), indent=2))


if __name__ == '__main__':
    main()
