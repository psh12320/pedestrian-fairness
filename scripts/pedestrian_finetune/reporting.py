"""Paired warm-start results with explicit partial crossing protection.

The original zero-margin every-crossing criterion is unchanged. A count such
as 6/12 is a descriptive trade-off result, never a substitute success claim.
Only reporting helpers are added here; frozen simulator/training code is reused.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import statistics as st

from src.traffic_control.audit_trial.reporting import (
    absolute_metrics, comparison, end_vehicle_metrics, paired_estimator, validate_result,
)
from src.traffic_control.fairness.analysis import quantile


def detailed_comparison(models, references, draws=1000, reference_name='actuated'):
    """Extend the verified comparison without changing its strict service gate."""
    for matrix in (models, references):
        for replica in matrix:
            for result in replica:
                validate_result(result)
    out = comparison(models, references, draws=draws)
    point = out['point']
    crossings = sorted(models[0][0]['crossings'])
    nr, ns = len(models), len(models[0])
    out['crossings'] = {}
    intervals = {c: None for c in crossings}
    if draws and ns >= 2:
        # Same paired draws as the base comparison: models and their references
        # always receive the same replica and demand resample.
        rng = random.Random(424693)
        estimate = paired_estimator(models, references)
        samples = [estimate(rng.choices(range(nr), k=nr), rng.choices(range(ns), k=ns))
                   for _ in range(draws)]
        for c in crossings:
            values = [s['crossing_p95_delta_seconds'][c] for s in samples]
            if all(v is not None for v in values):
                intervals[c] = [quantile(values, .025), quantile(values, .975)]
    for c in crossings:
        counts = [[sum(result['crossings'][c]['entries'] for result in replica)
                   for replica in matrix] for matrix in (models, references)]
        zero_entries = [[r, s] for r, replica in enumerate(models)
                        for s, row in enumerate(replica) if row['crossings'][c]['entries'] == 0]
        delta = point['crossing_p95_delta_seconds'][c]
        out['crossings'][c] = {
            'p95_delta_seconds': delta,
            'p95_no_higher_point_estimate': delta is not None and delta <= 0,
            'pointwise_exploratory_95_interval_seconds': intervals[c],
            'model_entries_per_replica': counts[0],
            'reference_entries_per_replica': counts[1],
            'sparse': min(counts[0] + counts[1]) < 100,
            'model_zero_entry_runs_replica_seed_index': zero_entries,
            'model_unresolved_requests_mean': st.mean(row['crossings'][c]['unresolved_after_followup']
                                                    for replica in models for row in replica),
            'reference_unresolved_requests_mean': st.mean(row['crossings'][c]['unresolved_after_followup']
                                                        for replica in references for row in replica),
        }
    vehicle_gain = point['vehicle_delta_seconds'] is not None and point['vehicle_delta_seconds'] < 0
    k = point['crossings_no_worse']
    service_ok = out['service_failure_pairs'] == 0
    out['partial_crossing_protection_with_vehicle_gain'] = bool(vehicle_gain and 0 < k < 12)
    out['partial_evidence_with_no_strict_service_deterioration'] = bool(
        vehicle_gain and 0 < k < 12 and service_ok and not out['sparse'])
    vehicle_ci = (out.get('exploratory_intervals') or {}).get('vehicle_delta_seconds')
    supported = [c for c, value in out['crossings'].items()
                 if value['pointwise_exploratory_95_interval_seconds'] is not None
                 and value['pointwise_exploratory_95_interval_seconds'][1] <= 0
                 and not value['sparse']]
    out['pointwise_interval_supported_crossings'] = supported
    out['exploratory_vehicle_gain_interval_supported'] = bool(vehicle_ci and vehicle_ci[1] < 0)
    out['exploratory_partial_evidence'] = bool(
        out['exploratory_vehicle_gain_interval_supported'] and supported and 0 < k < 12
        and service_ok and not out['sparse'])
    out['comparison_reference'] = reference_name
    out['original_target_descriptively_met'] = bool(reference_name == 'actuated' and out['descriptive_pass'])
    target = 'original target' if reference_name == 'actuated' else 'joint comparison gate'
    if point['missing_crossings'] or point['vehicle_delta_seconds'] is None or out['sparse']:
        label = 'inconclusive: missing or sparse measurements'
    elif out['descriptive_pass']:
        label = f'{target} descriptively met; statistical confirmation not established'
    elif vehicle_gain and k > 0:
        label = f'vehicle gain with {k}/12 crossing p95 point estimates no higher; {target} unmet'
        if not service_ok:
            label += '; strict service deterioration also present'
    elif vehicle_gain:
        label = f'vehicle gain with all 12 crossing p95 point estimates worse; {target} unmet'
    else:
        label = f'no vehicle time-loss improvement; {target} unmet'
    if reference_name != 'actuated':
        label += '; original target is assessed against actuated only'
    out['evidence_label'] = label
    out['crossing_interval_scope'] = ('Pointwise exploratory intervals; not simultaneous non-inferiority guarantees. '
        'Counting interval-supported crossings after observing them is descriptive and does not establish a simultaneous subset claim.')
    out['partial_metric_scope'] = 'Count of crossing point estimates; does not redefine the original every-crossing target'
    return out


def _records_to_results(matrix):
    for replica in matrix:
        for row in replica:
            if 'network_sha256' not in row:
                raise ValueError('Evaluation record lacks network identity')
            validate_result(row['result'])
    return [[row['result'] for row in replica] for replica in matrix]


def _crossing_completion(matrix):
    rows = [row for replica in matrix for row in replica]
    # Caller can retain per-crossing counts computed from event entry/exit times.
    # These are supplementary horizon diagnostics, not queue-wait imputations.
    values = [r.get('crossing_completion') for r in rows]
    if any(v is None for v in values):
        return {'available': False, 'runs': len(rows),
                'note': 'Entered-but-not-exited diagnostics were not recorded for every run'}
    crossings = sorted(values[0])
    if any(sorted(value) != crossings for value in values):
        raise ValueError('Mismatched crossing completion diagnostics')
    fields = ('entered_without_observed_exit', 'completed_visits')
    if any(not isinstance(value[c][field], int) or value[c][field] < 0
           for value in values for c in crossings for field in fields):
        raise ValueError('Invalid crossing completion counts')
    return {'available': True, 'runs': len(rows), 'per_run': values,
            'scope': 'All recorded visits at the simulation horizon, including warmup; not the primary cutoff cohort',
            'crossings': {c: {field: st.mean(value[c][field] for value in values) for field in fields}
                          for c in crossings},
            'network_means': {field: st.mean(sum(value[c][field] for c in crossings) for value in values)
                              for field in fields}}


def aggregate_scenario(baselines, learned, draws=1000, contrasts=None, original_group='mappo-original'):
    """Accept record matrices; caller is responsible for frozen model identities.

    baselines: {controller: [record by demand seed]}
    learned: {group: [[record by demand seed] by training replica]}
    Each record contains result and network_sha256, optionally vehicles_end and
    crossing_completion. All groups share the same ordered demand seeds.
    """
    if 'actuated' not in baselines or not baselines['actuated']:
        raise ValueError('A complete actuated reference is required')
    ns = len(baselines['actuated'])
    if not learned:
        raise ValueError('No learned controller results')
    nr = len(next(iter(learned.values())))
    if nr < 1:
        raise ValueError('No training replicas')
    matrices = {k: [v] for k, v in baselines.items()} | learned
    for label, matrix in matrices.items():
        if not matrix or any(len(replica) != ns for replica in matrix):
            raise ValueError('Incomplete evaluation demand coverage: ' + label)
        if label in learned and len(matrix) != nr:
            raise ValueError('Unequal training replica coverage')
        for replica in matrix:
            for s, row in enumerate(replica):
                ref = baselines['actuated'][s]
                if row.get('network_sha256') != ref.get('network_sha256'):
                    raise ValueError('Unmatched evaluation network: ' + label)
                if row['result']['demand_sha256'] != ref['result']['demand_sha256']:
                    raise ValueError('Unmatched evaluation demand: ' + label)
    base = {k: _records_to_results([v])[0] for k, v in baselines.items()}
    policies = {k: _records_to_results(v) for k, v in learned.items()}
    out = {'baselines': {}, 'learned': {}, 'paired_contrasts': {}}
    for label, rows in base.items():
        matrix = [rows]
        out['baselines'][label] = {
            'absolute': absolute_metrics(matrix),
            'versus_actuated': detailed_comparison(matrix, [base['actuated']], draws),
            'vehicles_end': end_vehicle_metrics([[r.get('vehicles_end') for r in baselines[label]]]),
            'crossing_completion': _crossing_completion([baselines[label]]),
        }
    for label, matrix in policies.items():
        comparisons = {c: detailed_comparison(matrix, [rows] * nr, draws if c == 'actuated' else 0, c)
                       for c, rows in base.items()}
        if original_group in policies:
            comparisons[original_group] = detailed_comparison(matrix, policies[original_group], draws, original_group)
        out['learned'][label] = {
            'absolute': absolute_metrics(matrix), 'comparisons': comparisons,
            'vehicles_end': end_vehicle_metrics([[r.get('vehicles_end') for r in replica]
                                                for replica in learned[label]]),
            'crossing_completion': _crossing_completion(learned[label]),
        }
    for left, right in contrasts or []:
        if left not in policies or right not in policies:
            raise ValueError('Missing predeclared contrast arm')
        out['paired_contrasts'][left + '-vs-' + right] = detailed_comparison(policies[left], policies[right], draws, right)
    return out


def write_report(root, protocol, scenarios, coverage):
    """Write only a complete study, after caller verifies training/checkpoint hashes."""
    if coverage.get('complete') is not True:
        raise ValueError('Incomplete fine-tuning study cannot produce a final report')
    if set(scenarios) != set(protocol['evaluation_scenarios']):
        raise ValueError('Missing planned scenario')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary = {'protocol': protocol, 'fingerprint': protocol['fingerprint'], 'coverage': coverage,
               'scenarios': scenarios,
               'interpretation': ('Quick smoke validation only; not evidence of algorithm improvement.' if protocol.get('quick') else
                    'Exploratory warm-start study; partial crossing protection is not the original all-crossing target.')}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    fmt = lambda v: 'NA' if v is None else f'{v:.2f}'
    lines = ['# Pedestrian-focused MAPPO fine-tuning', '',
        ('**Quick smoke test only.** The reduced horizon, one starting replica and one evaluation seed verify execution and accounting. These outputs are not research evidence of improvement.\n\n' if protocol.get('quick') else '') +
        'The starting policies are vehicle-efficient MAPPO-control checkpoints trained with pedestrian reward weight 3. This is a full model warm start, including actor, critic and value-normalisation state; these are not vehicle-only-reward checkpoints.', '',
        'All planned starting replicas and final-budget checkpoints are included. This experiment supplements the original proposal comparison. It does not replace earlier failures or use a best-run headline.', '',
        'Vehicle loss is completed-vehicle SUMO time loss. Count tables show means per simulation, not pooled totals. Vehicle service counts and insertion delay remain visible. Crossing tails pool demand seeds within each training replica and then average replica tails. Intervals are exploratory paired bootstrap intervals over whole replicas/demand seeds. Per-crossing intervals are pointwise, not simultaneous guarantees.', '',
        'A result such as vehicle improvement with 6/12 crossing p95s no higher is a partial trade-off result. The original target still requires all 12 crossing p95s no higher and no strict service deterioration. Sparse estimates (<100 entries in any pooled replica/reference crossing) are inconclusive.', '']
    for scenario, data in scenarios.items():
        lines += [f'## {scenario}', '',
                  '| Controller | Vehicle loss s | Insertion delay s | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label, row in {**data['baselines'], **data['learned']}.items():
            a = row['absolute']
            fields = ('vehicle_mean_time_loss', 'vehicle_mean_insertion_delay', 'vehicle_scheduled_count',
                      'vehicle_completed_count', 'vehicle_unfinished_count', 'vehicle_uninserted_count',
                      'pedestrian_pending_cutoff', 'pedestrian_unresolved')
            lines.append('| ' + label + ' | ' + ' | '.join(fmt(a[k]) for k in fields) + ' |')
        lines += ['', 'All-horizon vehicle service below includes warmup and is a different cohort from the primary measurements above.', '',
                  '| Controller | Horizon scheduled | Horizon completed | Horizon unfinished | Horizon uninserted |',
                  '|---|---:|---:|---:|---:|']
        for label, row in {**data['baselines'], **data['learned']}.items():
            v = row['vehicles_end']
            lines.append('| ' + label + ' | ' + ' | '.join(fmt(v.get(k)) for k in
                ('scheduled_count', 'completed_count', 'unfinished_count', 'uninserted_count')) + ' |')
        lines += ['', '| Controller | Persons scheduled | Persons completed | Persons unfinished | Persons uninserted | Entered crossing without observed exit (all horizon) |',
                  '|---|---:|---:|---:|---:|---:|']
        for label, row in {**data['baselines'], **data['learned']}.items():
            counts = row['absolute']['persons']
            completion = row['crossing_completion']
            entered = completion.get('network_means', {}).get('entered_without_observed_exit')
            lines.append('| ' + label + ' | ' + ' | '.join(fmt(counts[k]) for k in
                ('scheduled', 'completed', 'unfinished', 'uninserted')) + ' | ' + fmt(entered) + ' |')
        lines += ['', '| Policy / reference | Vehicle delta s (exploratory 95% interval) | Crossings no worse | Worst crossing delta s | Strict service failures | Interpretation |',
                  '|---|---|---:|---:|---:|---|']
        for label, row in data['learned'].items():
            for reference in ('actuated', 'mappo-original'):
                if reference not in row['comparisons']:
                    continue
                c = row['comparisons'][reference]; p = c['point']
                ci = (c.get('exploratory_intervals') or {}).get('vehicle_delta_seconds')
                interval = '[' + ', '.join(map(fmt, ci)) + ']' if ci else 'not estimated'
                lines.append(f"| {label} / {reference} | {fmt(p['vehicle_delta_seconds'])} ({interval}) | {p['crossings_no_worse']}/12 | {fmt(p['worst_crossing_delta_seconds'])} | {c['service_failure_pairs']}/{c['pairs']} | {c['evidence_label']} |")
        for label, row in data['learned'].items():
            for reference in ('actuated', 'mappo-original'):
                if reference not in row['comparisons']:
                    continue
                c = row['comparisons'][reference]
                lines += ['', f'### {label}: all crossings versus {reference}', '',
                          '| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |',
                          '|---|---:|---|---|---|---|']
                for crossing, value in c['crossings'].items():
                    interval = value['pointwise_exploratory_95_interval_seconds']
                    text = '[' + ', '.join(map(fmt, interval)) + ']' if interval else 'not estimated'
                    lines.append(f"| {crossing} | {fmt(value['p95_delta_seconds'])} | {text} | {value['model_entries_per_replica']} | {value['reference_entries_per_replica']} | {fmt(value['model_unresolved_requests_mean'])}/{fmt(value['reference_unresolved_requests_mean'])} |")
        lines += ['', 'The full JSON also retains person-trip service, slower-walker groups where recorded, zero-entry runs, cutoff-only failures, whole-horizon vehicle counts, and entered-but-not-exited diagnostics where available. Missing diagnostics remain unavailable rather than zero.', '']
    lines += ['## Limits', '',
              'These results concern one synthetic corridor and the frozen demands/budgets. Teacher guidance, reward shaping and extra experience must be interpreted through their matched contrasts. A vehicle gain with some protected crossings does not imply fairness at other crossings, successful service of unserved users, or an improvement to deployed traffic lights.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary
