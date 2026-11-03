"""Complete, paired development reports for the local audit-repair experiment.

Replica-level crossing tails pool pedestrians over demand seeds, then average
the replica tails. Neither individual-run p95s nor replicas' pedestrians are
pooled indiscriminately. Bootstrap draws preserve paired models/references.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
from collections import Counter
from bisect import bisect_right

from ..fairness.analysis import quantile, selection_score
from ..fairness.pipeline import atomic_json


VEHICLE_COUNTS = ('scheduled_count', 'completed_count', 'unfinished_count', 'uninserted_count')
DELAY_FIELDS = ('completed_mean_time_loss_seconds', 'completed_mean_depart_delay_seconds',
                'completed_mean_time_loss_plus_depart_delay_seconds')


def _mean(values):
    values = list(values)
    return st.mean(values) if values and all(v is not None for v in values) else None


def validate_result(result):
    v = result['vehicles']
    if any(v[k] < 0 for k in VEHICLE_COUNTS):
        raise ValueError('Negative vehicle count')
    if v['scheduled_count'] != sum(v[k] for k in VEHICLE_COUNTS[1:]):
        raise ValueError('Vehicle accounting failed')
    if len(result['crossings']) != 12:
        raise ValueError('Expected all 12 crossings')
    for value in result['crossings'].values():
        waits = value['waits']
        if (value['entries'] != len(waits) or value['p95'] != quantile(waits)
                or any(not math.isfinite(w) or w < 0 for w in waits)):
            raise ValueError('Invalid crossing measurements')
    p = result.get('persons')
    if p and p['scheduled'] != p['completed'] + p['unfinished'] + p['uninserted']:
        raise ValueError('Person accounting failed')
    for key in DELAY_FIELDS:
        x = v.get(key)
        if x is not None and (not math.isfinite(x) or x < 0):
            raise ValueError('Invalid vehicle delay')


def service_components(a, b):
    """Boolean pair-level reasons; these are not additive disjoint categories."""
    av, bv = a['vehicles'], b['vehicles']
    result = {
        'fewer_completed_vehicles': av['completed_count'] < bv['completed_count'],
        'more_unfinished_vehicles': av['unfinished_count'] > bv['unfinished_count'],
        'more_uninserted_vehicles': av['uninserted_count'] > bv['uninserted_count'],
        'more_crossing_requests_pending_at_cutoff': any(
            a['crossings'][c]['pending_at_cutoff'] > b['crossings'][c]['pending_at_cutoff']
            for c in a['crossings']),
        'more_crossing_requests_unresolved_after_followup': any(
            a['crossings'][c]['unresolved_after_followup'] > b['crossings'][c]['unresolved_after_followup']
            for c in a['crossings']),
        'unknown_or_lost_measurements': any(v['unknown_cohort_entries'] or v['lost']
            for row in (a, b) for v in row['crossings'].values()),
    }
    if 'persons' in a and 'persons' in b:
        p, q = a['persons'], b['persons']
        result['worse_person_trip_service'] = (p['completed'] < q['completed']
            or p['unfinished'] > q['unfinished'] or p['uninserted'] > q['uninserted'])
    return result


def paired_estimator(models, references):
    """Weighted nearest-rank CDFs make the paired bootstrap inexpensive."""
    crossings = sorted(models[0][0]['crossings'])

    def prepare(matrix):
        prepared = []
        for row in matrix:
            local = {}
            for c in crossings:
                waits = [sorted(r['crossings'][c]['waits']) for r in row]
                levels = sorted({w for values in waits for w in values})
                local[c] = (levels, [[bisect_right(values, level) for level in levels] for values in waits],
                            [len(values) for values in waits])
            prepared.append(local)
        return prepared

    left, right = prepare(models), prepare(references)

    def tail(prepared, weights):
        levels, cdfs, lengths = prepared
        count = sum(lengths[s] * weight for s, weight in weights)
        if not count:
            return None
        rank = math.ceil(.95 * count)
        lo, hi = 0, len(levels) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if sum(cdfs[s][mid] * weight for s, weight in weights) >= rank:
                hi = mid
            else:
                lo = mid + 1
        return levels[lo]

    def estimate(replicas, seeds):
        weights = tuple(Counter(seeds).items())
        deltas = {}
        for c in crossings:
            pairs = [(tail(left[r][c], weights), tail(right[r][c], weights)) for r in replicas]
            deltas[c] = (_mean(a - b for a, b in pairs) if all(a is not None and b is not None for a, b in pairs)
                         else None)
        delay = {}
        for field in DELAY_FIELDS:
            pairs = [(models[r][s]['vehicles'].get(field), references[r][s]['vehicles'].get(field))
                     for r in replicas for s in seeds]
            delay[field] = (_mean(a - b for a, b in pairs)
                            if all(a is not None and b is not None for a, b in pairs) else None)
        return {
            'vehicle_delta_seconds': delay[DELAY_FIELDS[0]],
            'insertion_delay_delta_seconds': delay[DELAY_FIELDS[1]],
            'time_loss_plus_insertion_delay_delta_seconds': delay[DELAY_FIELDS[2]],
            'crossing_p95_delta_seconds': deltas,
            'worst_crossing_delta_seconds': max(deltas.values()) if all(v is not None for v in deltas.values()) else None,
            'crossings_no_worse': sum(v is not None and v <= 0 for v in deltas.values()),
            'missing_crossings': [c for c, v in deltas.items() if v is None],
        }
    return estimate


def comparison(models, references, draws=1000):
    if not models or not models[0]:
        raise ValueError('Empty comparison')
    nr, ns = len(models), len(models[0])
    if len(references) != nr or any(len(row) != ns for row in models + references):
        raise ValueError('Unpaired result dimensions')
    cells = [(models[r][s], references[r][s]) for r in range(nr) for s in range(ns)]
    crossing_order = sorted(models[0][0]['crossings'])
    for a, b in cells:
        if a['demand_sha256'] != b['demand_sha256']:
            raise ValueError('Unmatched demand')
        if sorted(a['crossings']) != crossing_order or sorted(b['crossings']) != crossing_order:
            raise ValueError('Mismatched crossings')
    estimate = paired_estimator(models, references)
    point = estimate(list(range(nr)), list(range(ns)))
    reasons = [service_components(a, b) for a, b in cells]
    failures = sum(selection_score(a, b)[0] > 0 for a, b in cells)
    minimum = min(sum(row[s]['crossings'][c]['entries'] for s in range(ns))
                  for matrix in (models, references) for row in matrix for c in crossing_order)
    output = {
        'point': point, 'pairs': nr * ns, 'replicas': nr, 'demand_seeds': ns,
        'service_failure_pairs': failures,
        'service_failure_components': {k: sum(row.get(k, False) for row in reasons)
                                       for k in sorted({key for row in reasons for key in row})},
        'cutoff_only_failure_pairs': sum(row['more_crossing_requests_pending_at_cutoff']
            and not any(value for key, value in row.items() if key != 'more_crossing_requests_pending_at_cutoff')
            for row in reasons),
        'vehicle_count_mean_deltas': {k: st.mean(a['vehicles'][k] - b['vehicles'][k] for a, b in cells)
                                      for k in VEHICLE_COUNTS},
        'crossing_service_mean_deltas': {c: {field: st.mean(a['crossings'][c][field] - b['crossings'][c][field]
            for a, b in cells) for field in ('pending_at_cutoff', 'unresolved_after_followup')}
            for c in crossing_order},
        'minimum_pooled_crossing_entries': minimum, 'sparse': minimum < 100,
        'descriptive_pass': bool(point['vehicle_delta_seconds'] is not None and point['vehicle_delta_seconds'] < 0
            and point['worst_crossing_delta_seconds'] is not None and point['worst_crossing_delta_seconds'] <= 0
            and failures == 0),
        'exploratory_intervals': None, 'bootstrap_requested_draws': draws, 'bootstrap_valid_draws': 0,
        'confirmatory_statistical_claim': False,
    }
    if draws and ns >= 2:
        rng = random.Random(424693)
        samples = [estimate(rng.choices(range(nr), k=nr), rng.choices(range(ns), k=ns)) for _ in range(draws)]
        output['bootstrap_valid_draws'] = sum(s['vehicle_delta_seconds'] is not None for s in samples)
        output['bootstrap_valid_crossing_draws'] = sum(s['worst_crossing_delta_seconds'] is not None for s in samples)
        keys = ('vehicle_delta_seconds', 'insertion_delay_delta_seconds',
                'time_loss_plus_insertion_delay_delta_seconds', 'worst_crossing_delta_seconds')
        output['exploratory_intervals'] = {k: ([quantile([s[k] for s in samples], .025),
            quantile([s[k] for s in samples], .975)] if all(s[k] is not None for s in samples) else None) for k in keys}
        output['bootstrap_unit'] = ('paired training replicas and demand seeds' if nr > 1
                                     else 'paired demand seeds only; no training variability estimate')
    return output


def absolute_metrics(models):
    rows = [result for replica in models for result in replica]
    output = {'vehicle_' + k: _mean(r['vehicles'][k] for r in rows) for k in VEHICLE_COUNTS}
    for short, field in zip(('vehicle_mean_time_loss', 'vehicle_mean_insertion_delay',
                            'vehicle_mean_time_loss_plus_insertion_delay'), DELAY_FIELDS):
        output[short] = _mean(r['vehicles'].get(field) for r in rows)
    output['delay_coverage_runs'] = sum(r['vehicles'].get('completed_depart_delay_observed_count')
                                      == r['vehicles']['completed_count'] for r in rows)
    output['runs'] = len(rows)
    output['pedestrian_pending_cutoff'] = _mean(sum(c['pending_at_cutoff'] for c in r['crossings'].values()) for r in rows)
    output['pedestrian_unresolved'] = _mean(sum(c['unresolved_after_followup'] for c in r['crossings'].values()) for r in rows)
    output['persons'] = {k: _mean(r.get('persons', {}).get(k) for r in rows)
                         for k in ('scheduled', 'completed', 'unfinished', 'uninserted')}
    output['crossings'] = {}
    for c in sorted(rows[0]['crossings']):
        waits = [[w for result in replica for w in result['crossings'][c]['waits']] for replica in models]
        value = {
            'mean_wait': _mean(_mean(w) for w in waits),
            'mean_replica_pooled_p95': _mean(quantile(w) for w in waits),
            'p95_per_replica': [quantile(w) for w in waits],
            'entries_per_replica': [len(w) for w in waits],
            'pending_cutoff_mean_per_run': _mean(r['crossings'][c]['pending_at_cutoff'] for r in rows),
            'unresolved_mean_per_run': _mean(r['crossings'][c]['unresolved_after_followup'] for r in rows),
            'speed_groups': {},
        }
        speeds = sorted({speed for r in rows for speed in r['crossings'][c].get('speed_groups', {})})
        for speed in speeds:
            samples = [[w for r in replica for w in r['crossings'][c].get('speed_groups', {}).get(speed, {}).get('waits', [])]
                       for replica in models]
            value['speed_groups'][speed] = {'mean_wait': _mean(_mean(w) for w in samples),
                'mean_replica_pooled_p95': _mean(quantile(w) for w in samples),
                'entries_per_replica': [len(w) for w in samples]}
        output['crossings'][c] = value
    return output


def end_vehicle_metrics(matrices):
    """All scheduled trips at the simulation horizon; a different cohort/window."""
    rows = [v for replica in matrices for v in replica]
    if not rows or any(v is None for v in rows):
        return {'available': False, 'runs': len(rows), 'note': 'Whole-horizon vehicle summary not recorded for every run'}
    for v in rows:
        if v['scheduled_count'] != v['completed_count'] + v['unfinished_count'] + v['uninserted_count']:
            raise ValueError('Whole-horizon vehicle accounting failed')
    return {'available': True, 'runs': len(rows),
            'note': 'All scheduled departures including warmup, measured at simulation end; not the primary cutoff cohort',
            **{k: _mean(v[k] for v in rows) for k in VEHICLE_COUNTS},
            **{k: _mean(v.get(k) for v in rows) for k in DELAY_FIELDS}}


def _load(path, fingerprint):
    if not path.is_file():
        raise ValueError('Incomplete experiment; missing ' + str(path))
    data = json.loads(path.read_text())
    if data.get('fingerprint') != fingerprint:
        raise ValueError('Wrong result protocol: ' + str(path))
    return data


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(protocol):
    return [(method, variant, replica, f'{method}-{variant}-r{replica}')
            for method in protocol['methods'] for variant in protocol['variants']
            for replica in range(protocol['replicas'])]


def report(root, protocol, draws=1000):
    """Fail closed on incomplete models or evaluations; write no partial final report."""
    root = Path(root)
    fp, nr, seeds = protocol['fingerprint'], protocol['replicas'], protocol['evaluation_seeds']
    baseline_names = protocol['baselines']
    completed, learning, demand_by_replica = {}, {}, {}
    for method, variant, replica, label in labels(protocol):
        folder = root / 'models' / label
        done = _load(folder / 'complete.json', fp)
        if done['episodes'] != protocol['episodes'] or done['checkpoint_sha256'] != _digest(folder / 'final.pt'):
            raise ValueError('Training incomplete or checkpoint changed: ' + label)
        curve = _load(folder / 'learning.json', fp)
        rollouts = [r for history in curve['history'] for r in history['rollouts']]
        expected = [(i + 1, protocol['training_scenarios'][i % len(protocol['training_scenarios'])],
                     protocol['training_seed_base'] + replica * protocol.get('training_seed_stride', 1000) + i)
                    for i in range(protocol['episodes'])]
        if [(r['episode'], r['scenario'], r['seed']) for r in rollouts] != expected:
            raise ValueError('Training coverage mismatch: ' + label)
        matched = [(r['demand_sha256'], r['network_sha256']) for r in rollouts]
        if replica in demand_by_replica and matched != demand_by_replica[replica]:
            raise ValueError('Unmatched training demands: ' + label)
        demand_by_replica[replica] = matched
        validations = curve.get('validations', [])
        expected_validation = [(episode, 'vmedium-pmedium', seed)
            for episode in protocol.get('validation_episodes', []) for seed in protocol.get('validation_seeds', [])]
        if [(v['episodes'], v['scenario'], v['seed']) for v in validations] != expected_validation:
            raise ValueError('Validation coverage mismatch: ' + label)
        for validation in validations:
            validate_result(validation['result'])
            group = _load(root / 'baselines' / f"{validation['scenario']}-{validation['seed']}.json", fp)
            refs = group['records']
            if len(refs) != len(baseline_names) or {v['controller'] for v in refs} != set(baseline_names):
                raise ValueError('Wrong validation baseline coverage')
            expected_pair = (validation['result']['demand_sha256'], validation['network_sha256'])
            if any((v['result']['demand_sha256'], v['network_sha256']) != expected_pair for v in refs):
                raise ValueError('Unmatched validation demand/network')
        completed[label] = done
        learning[label] = {'episodes': len(rollouts), 'update_batches': len(curve['history']),
            'free_decisions_per_junction': {t: sum(r['diagnostics'][t]['free_decisions'] for r in rollouts)
                for t in ('C0', 'C1', 'C2')},
            'final_update': curve['history'][-1].get('update'), 'validations': validations}
    summaries = {}
    for scenario in protocol['evaluation_scenarios']:
        baselines = {c: [] for c in baseline_names}
        end_baselines = {c: [] for c in baseline_names}
        hashes, records, end_records = {}, {}, {}
        for seed in seeds:
            group = _load(root / 'baselines' / f'{scenario}-{seed}.json', fp)
            if len(group['records']) != len(baseline_names) or {r['controller'] for r in group['records']} != set(baseline_names):
                raise ValueError('Wrong baseline coverage')
            for record in group['records']:
                validate_result(record['result'])
                pair = (record['result']['demand_sha256'], record['network_sha256'])
                if seed in hashes and pair != hashes[seed]:
                    raise ValueError('Unmatched baseline demand/network')
                hashes[seed] = pair
                baselines[record['controller']].append(record['result'])
                end_baselines[record['controller']].append(record.get('vehicles_end'))
                if 'events_sha256' in record:
                    if _digest(root / 'events' / f"baseline-{record['controller']}-{scenario}-{seed}.json.gz") != record['events_sha256']:
                        raise ValueError('Baseline pedestrian events changed')
        for method, variant, replica, label in labels(protocol):
            rows, end_rows = [], []
            for seed in seeds:
                record = _load(root / 'evaluation' / f'{label}-{scenario}-{seed}.json', fp)
                if record['checkpoint_sha256'] != completed[label]['checkpoint_sha256']:
                    raise ValueError('Evaluation checkpoint changed')
                if (record['result']['demand_sha256'], record['network_sha256']) != hashes[seed]:
                    raise ValueError('Unmatched evaluation demand/network')
                validate_result(record['result'])
                rows.append(record['result'])
                end_rows.append(record.get('vehicles_end'))
                if 'events_sha256' in record:
                    if _digest(root / 'events' / f'{label}-{scenario}-{seed}.json.gz') != record['events_sha256']:
                        raise ValueError('Evaluation pedestrian events changed')
            records[label] = rows
            end_records[label] = end_rows
        summary = {'baselines': {}, 'learned': {}, 'paired_ablation': {}}
        for controller in baseline_names:
            summary['baselines'][controller] = {'absolute': absolute_metrics([baselines[controller]]),
                'whole_horizon_vehicles': end_vehicle_metrics([end_baselines[controller]]),
                'versus_actuated': comparison([baselines[controller]], [baselines['actuated']], draws=draws)}
        for method in protocol['methods']:
            for variant in protocol['variants']:
                label = f'{method}-{variant}'
                matrix = [records[f'{label}-r{r}'] for r in range(nr)]
                summary['learned'][label] = {'absolute': absolute_metrics(matrix),
                    'whole_horizon_vehicles': end_vehicle_metrics([end_records[f'{label}-r{r}'] for r in range(nr)]),
                    'comparisons': {c: comparison(matrix, [baselines[c]] * nr, draws=draws if c == 'actuated' else 0)
                                    for c in baseline_names}}
            for left, right in [('memory', 'legacy'), ('separate', 'legacy'),
                                ('memory-separate', 'memory'), ('memory-separate', 'separate'),
                                ('memory-separate', 'legacy')]:
                if left in protocol['variants'] and right in protocol['variants']:
                    summary['paired_ablation'][f'{method}:{left}-vs-{right}'] = comparison(
                        [records[f'{method}-{left}-r{r}'] for r in range(nr)],
                        [records[f'{method}-{right}-r{r}'] for r in range(nr)], draws=draws)
        summaries[scenario] = summary
    result = {'fingerprint': fp, 'protocol': protocol, 'completed_models': len(completed),
        'all_training_demands_matched': True, 'learning': learning, 'scenarios': summaries,
        'interpretation': 'Development ablation, not final testing. Few training replicas limit uncertainty; no confirmatory claim.'}
    atomic_json(root / 'summary.json', result)
    _write_markdown(root, result)
    return result


def _write_markdown(root, result):
    def fmt(x):
        return 'NA' if x is None else f'{x:.2f}'
    def interval(comparison, key):
        value = (comparison['exploratory_intervals'] or {}).get(key)
        return '[' + ', '.join(fmt(v) for v in value) + ']' if value else 'not estimated'
    protocol = result['protocol']
    lines = ['# Local SUMO audit-repair ablation', '',
        f"Engineering smoke only: **{protocol.get('quick', False)}**. Completed models: **{result['completed_models']}**, "
        f"each trained for **{protocol['episodes']} episodes**. Every final checkpoint and prescribed evaluation is present. "
        'No best-run selection. All paired demand and network hashes match.', '',
        'Legacy = previous normalized-clock PPO; memory = known controller state added; separate = actor/critic gradients clipped separately; '
        'memory-separate = both changes. Pedestrian reward weight remains 3. All controllers share the corrected collision checks, phase safety and measurements. '
        'This isolates learning changes; the earlier study and original four-controller proposal comparison remain separate.', '',
        f"Uncertainty is exploratory: {protocol['replicas']} training replicas and {len(protocol['evaluation_seeds'])} demand seeds. "
        'Resample whole paired replicas and demand seeds. Crossing p95 pools events across seeds within each replica, then averages replica tails. '
        'Sparse estimates (<100 entries for any crossing/replica/reference) are inconclusive. Baseline intervals resample demand only. '
        'Vehicle delay entries are means of run means, not traffic-volume-weighted means. Action randomness has one controlled stream per demand.', '']
    for scenario, summary in result['scenarios'].items():
        lines += [f'## {scenario}', '',
            '| Controller | Time loss s | Insertion delay s | Combined s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label, row in {**summary['baselines'], **summary['learned']}.items():
            a = row['absolute']
            fields = ('vehicle_mean_time_loss', 'vehicle_mean_insertion_delay', 'vehicle_mean_time_loss_plus_insertion_delay',
                      'vehicle_scheduled_count', 'vehicle_completed_count', 'vehicle_unfinished_count', 'vehicle_uninserted_count',
                      'pedestrian_pending_cutoff', 'pedestrian_unresolved')
            lines.append('| ' + label + ' | ' + ' | '.join(fmt(a[k]) for k in fields) + ' |')
        if all(row['whole_horizon_vehicles']['available'] for row in {**summary['baselines'], **summary['learned']}.values()):
            lines += ['', 'Whole-horizon service below includes **all scheduled departures, including warmup**. '
                'It is a different cohort and time window from the primary table above.', '',
                '| Controller | Scheduled | Completed at end | Unfinished at end | Uninserted at end |',
                '|---|---:|---:|---:|---:|']
            for label, row in {**summary['baselines'], **summary['learned']}.items():
                a = row['whole_horizon_vehicles']
                lines.append('| ' + label + ' | ' + ' | '.join(fmt(a[k]) for k in VEHICLE_COUNTS) + ' |')
        lines += ['', '| Learned vs actuated | Vehicle delta s (95% exploratory interval) | Worst crossing p95 delta s | Crossings no worse | Strict service failures | Cutoff-only failures | Min entries | Descriptive pass |',
                  '|---|---|---:|---:|---:|---:|---:|---|']
        for label, row in summary['learned'].items():
            c = row['comparisons']['actuated']; p = c['point']
            lines.append(f"| {label} | {fmt(p['vehicle_delta_seconds'])} {interval(c, 'vehicle_delta_seconds')} | "
                f"{fmt(p['worst_crossing_delta_seconds'])} | {p['crossings_no_worse']}/12 | {c['service_failure_pairs']}/{c['pairs']} | "
                f"{c['cutoff_only_failure_pairs']} | {c['minimum_pooled_crossing_entries']} | {c['descriptive_pass']} "
                f"{'(sparse: inconclusive)' if c['sparse'] else ''} |")
        lines += ['', '| Paired correction | Vehicle delta s (95% exploratory interval) | Worst crossing delta s (95% exploratory interval) | Crossings no worse | Strict service failures |',
                  '|---|---|---|---:|---:|']
        for label, c in summary['paired_ablation'].items():
            p = c['point']
            lines.append(f"| {label} | {fmt(p['vehicle_delta_seconds'])} {interval(c, 'vehicle_delta_seconds')} | "
                f"{fmt(p['worst_crossing_delta_seconds'])} {interval(c, 'worst_crossing_delta_seconds')} | "
                f"{p['crossings_no_worse']}/12 | {c['service_failure_pairs']}/{c['pairs']} |")
        lines += ['', 'A negative paired vehicle difference means the correction reduced completed-trip time loss relative to its named control. '
            'That alone does not establish the proposal target. Service-failure categories overlap; cutoff-only counts identify timing-sensitive failures '
            'without dropping the original criterion. Full crossing means/p95s, replica entry counts, slower-walker groups, '
            'service components and insertion-delay intervals are in summary.json; raw per-run waits remain in evaluation JSON.', '']
    lines += ['## Interpretation limits', '',
        'No result here proves deployed benefit, safety outside the checked model, or guaranteed PPO superiority. '
        'The joint criterion remains lower vehicle time loss, no higher p95 at any of 12 crossings, and no strict service deterioration. '
        'Sparse samples and the small number of training replicas limit inference; even a descriptive pass is not a confirmatory finding. '
        'The unchanged 600-second training versus 2400-second evaluation mismatch, local rewards, limited corridor-through traffic, '
        'and unobserved external backlog remain separate hypotheses. These ablations do not claim to fix them.', '',
        'Inspect effective free decisions, entropy, KL and gradients in learning.json and summary.json. '
        'Both improvements and failures must be retained. Reserved final test seeds remain unused.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
