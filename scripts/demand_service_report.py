"""Complete factorial report, paired seed/replica inference and service audit."""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import statistics as st

from scripts.pedestrian_finetune import pilot as shared
from scripts.pedestrian_finetune.reporting import aggregate_scenario

ROUNDING = 1e-9  # Arithmetic ties only, not a noninferiority margin.


def correct_rounding(comparison):
    point = comparison['point']
    deltas = point['crossing_p95_delta_seconds']
    count = sum(v is not None and v <= ROUNDING for v in deltas.values())
    comparison['exact_arithmetic_crossings_no_worse'] = point['crossings_no_worse']
    point['crossings_no_worse'] = count
    for c, row in comparison['crossings'].items():
        row['p95_no_higher_point_estimate'] = deltas[c] is not None and deltas[c] <= ROUNDING
    vehicle_gain = point['vehicle_delta_seconds'] is not None and point['vehicle_delta_seconds'] < 0
    service = comparison['service_failure_pairs'] == 0
    comparison['descriptive_pass'] = bool(vehicle_gain and count == 12 and service)
    comparison['original_target_descriptively_met'] = bool(
        comparison['comparison_reference'] == 'actuated' and comparison['descriptive_pass'])
    comparison['partial_crossing_protection_with_vehicle_gain'] = bool(vehicle_gain and 0 < count < 12)
    comparison['partial_evidence_with_no_strict_service_deterioration'] = bool(
        vehicle_gain and 0 < count < 12 and service and not comparison['sparse'])
    comparison['evidence_label'] = ('Sparse/incomplete: inconclusive' if comparison['sparse'] or point['missing_crossings']
        else ('Joint point estimates and strict service gate passed; no confirmatory statistical claim'
              if comparison['descriptive_pass'] else 'Joint target not met'))
    comparison['roundoff_tolerance_seconds'] = ROUNDING


def load_records(root, p):
    from scripts.demand_service_test import jobs, job_name
    records, audit = {}, Counter()
    for job in jobs(p):
        name = job_name(job)
        row = shared.cached_json(root / 'records' / (name + '.json'), p['fingerprint'])
        if row is None:
            raise ValueError('Incomplete study: ' + name)
        if (row['scenario'], row['seed'], row['group'], row['replica']) != job:
            raise ValueError('Wrong run identity: ' + name)
        for field, directory in [('events_sha256', 'events'), ('decisions_sha256', 'decisions')]:
            path = root / directory / (name + '.json.gz')
            if shared.digest(path) != row[field]:
                raise ValueError('Artifact changed: ' + name)
        if row['replica'] is not None:
            label = ('mappo-continue' if row['group'] == 'mappo-early' else row['group']) + f'-r{row["replica"]}'
            if row['checkpoint_sha256'] != p['models'][label]['sha256']:
                raise ValueError('Wrong model identity')
        if row['group'] == 'mappo-early' and (row['audit'].get('pedestrian_service_opportunities', 0)
                != row['audit'].get('pedestrian_opportunities_served', 0)):
            raise ValueError('Hybrid did not implement early service')
        events = json.loads(gzip.decompress((root / 'events' / (name + '.json.gz')).read_bytes()))
        # Independent cohort reconstruction from arrival and entry timestamps.
        start, end = (0, 120) if p['quick'] else (300, 2100)
        for c, result in row['result']['crossings'].items():
            cohort = [e for e in events if e['crossing_id'] == c and e['reached_at'] is not None
                      and start <= e['reached_at'] < end]
            waits = [e['entered_at'] - e['reached_at'] for e in cohort if e['entered_at'] is not None]
            if len(waits) != result['entries'] or any(abs(a - b) > 1e-7 for a, b in zip(waits, result['waits'])):
                raise ValueError('Crossing wait reconstruction failed')
            if sum(e['entered_at'] is None for e in cohort) != result['unresolved_after_followup']:
                raise ValueError('Unresolved request reconstruction failed')
        audit.update(row['audit'])
        records[job] = row
    return records, dict(audit)


def report(root, p):
    root = Path(root)
    records, audit = load_records(root, p)
    scenarios = {}
    for scenario in p['evaluation_scenarios']:
        baselines = {b: [records[scenario, seed, b, None] for seed in p['evaluation_seeds']] for b in p['baselines']}
        learned = {f'mappo-{a}': [[records[scenario, seed, f'mappo-{a}', r] for seed in p['evaluation_seeds']]
                                for r in range(p['replicas'])] for a in p['arms']}
        data = aggregate_scenario(baselines, learned, draws=p['bootstrap_draws'], contrasts=p['contrasts'])
        comparisons = [row['versus_actuated'] for row in data['baselines'].values()]
        comparisons += [c for row in data['learned'].values() for c in row['comparisons'].values()]
        comparisons += list(data['paired_contrasts'].values())
        for c in comparisons:
            correct_rounding(c)
        for group, matrix in ({b: [rows] for b, rows in baselines.items()} | learned).items():
            rows = [row for replica in matrix for row in replica]
            counts = Counter()
            for row in rows:
                counts.update(row['audit'])
            target = data['baselines' if group in baselines else 'learned'][group]
            target['decision_audit'] = dict(counts)
            durations = Counter()
            for row in rows:
                for junction in row['diagnostics'].values():
                    durations.update(junction['phase_seconds'])
            total = sum(durations.values())
            target['phase_fraction'] = {k: v / total for k, v in durations.items()}
            target['mean_runtime_seconds'] = st.mean(r['elapsed_seconds'] for r in rows)
        scenarios[scenario] = data
    summary = dict(protocol=p, coverage=dict(complete=True, simulations=len(records)), audit=audit, scenarios=scenarios)
    shared.atomic_json(root / 'summary.json', summary)
    fmt = lambda x: 'NA' if x is None else f'{x:.2f}'
    lines = ['# Demand and early pedestrian service: frozen-model evaluation', '',
        '**Smoke test only; not research evidence.**' if p['quick'] else
        '**Complete development experiment: 720 evaluations, no retraining.**', '',
        'Three saved MAPPO replicas per learned arm. Early service uses the continued-policy weights and '
        'requests PED whenever a local request is pending and PED is legal. Vehicle decisions and empty '
        'pedestrian requests otherwise follow the saved policy. It is a hybrid controller, not a newly trained algorithm.', '',
        'All controllers share the same geometry and physical safety rules. Demand and networks are hash-matched within '
        'each setting and seed. Final reserved test seeds are untouched. Earlier unfavorable studies remain unchanged.', '',
        'Vehicle time loss is conditional on completion. Vehicle cohort: scheduled departures in [300,2100) seconds, '
        'status at 2100. Pedestrians reaching crossings in that window are followed to 2400. Whole-run service is also '
        'retained separately in summary.json. Crossing p95 pools seeds within each replica, then averages replica tails. '
        'Intervals resample paired whole seeds and replicas; they are exploratory, not a simultaneous noninferiority claim. '
        'Fewer than 100 entries at any pooled replica/reference crossing is flagged as sparse. Only 1e-9 second arithmetic ties are ignored.', '']
    for scenario, data in scenarios.items():
        lines += [f'## {scenario}', '', '| Controller | Vehicle loss s | Change vs actuated s | Paired 95% interval | Crossings no worse /12 | Worst crossing excess s | Sparse? |',
                  '|---|---:|---:|---|---:|---:|---|']
        rows = {**data['baselines'], **data['learned']}
        for label, row in rows.items():
            c = row['versus_actuated'] if label in data['baselines'] else row['comparisons']['actuated']
            point, ci = c['point'], c.get('exploratory_intervals')
            interval = ci['vehicle_delta_seconds'] if ci else None
            interval = '[' + ', '.join(fmt(v) for v in interval) + ']' if interval else 'NA'
            lines.append(f'| {label} | {fmt(row["absolute"]["vehicle_mean_time_loss"])} | {fmt(point["vehicle_delta_seconds"])} | {interval} | '
                f'{point["crossings_no_worse"]} | {fmt(point["worst_crossing_delta_seconds"])} | {c["sparse"]} |')
        lines += ['', '| Controller | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup | Insertion delay s | Strict service failures / pairs |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for label, row in rows.items():
            a = row['absolute']
            c = row['versus_actuated'] if label in data['baselines'] else row['comparisons']['actuated']
            values = [a[k] for k in ('vehicle_scheduled_count', 'vehicle_completed_count', 'vehicle_unfinished_count',
                'vehicle_uninserted_count', 'pedestrian_pending_cutoff', 'pedestrian_unresolved', 'vehicle_mean_insertion_delay')]
            lines.append('| ' + label + ' | ' + ' | '.join(fmt(v) for v in values) + f' | {c["service_failure_pairs"]}/{c["pairs"]} |')
        lines += ['', '| Crossing | Actuated p95 | Original | Continue | Tail | Guided | Early service | Minimum entries across all replicas/controllers |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for crossing in sorted(rows['actuated']['absolute']['crossings']):
            names = ['actuated'] + [f'mappo-{a}' for a in p['arms']]
            vals = [rows[n]['absolute']['crossings'][crossing] for n in names]
            minimum = min(n for v in vals for n in v['entries_per_replica'])
            lines.append('| ' + crossing + ' | ' + ' | '.join(fmt(v['mean_replica_pooled_p95']) for v in vals) + f' | {minimum} |')
        contrast = data['paired_contrasts']['mappo-early-vs-mappo-continue']
        lines += ['', 'Early-service minus unchanged continued MAPPO: ' + json.dumps(contrast['point'], sort_keys=True), '',
            'The complete summary includes crossing means, per-replica tails, pointwise intervals, count differences, '
            'whole-run vehicle/person service, phase occupancy, decision audits, and the strict service-gate components.', '']
    lines += ['## Limitations', ''] + ['- ' + v for v in p['limitations']]
    lines += ['', 'No sparse or missing crossing is counted as zero wait. A partial crossing count or favorable setting does '
              'not replace the original every-crossing, no-service-deterioration target. An unchanged model at low demand '
              'tests transfer; it is not evidence about a model retrained for low demand.', '',
              f'Artifact verification: {len(records)} complete records and both compressed event/decision hashes per record. '
              f'{audit.get("legal_crossing_entries", 0)} recorded entries checked against the signal active during movement.', '']
    shared.atomic_bytes(root / 'REPORT.md', '\n'.join(lines).encode())
    return summary
