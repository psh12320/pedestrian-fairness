"""Stage-1 report: reuses the frozen aggregate_scenario/correct_rounding core but
knows about the mappo-early+vehrule arm. Writes REPORT.md + summary.json.

Kept separate from demand_service_report.py so the frozen E15 report and its
hashes stay untouched. Only the arm set, label->checkpoint mapping and the
printed contrasts differ.
"""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

from scripts.pedestrian_finetune.reporting import aggregate_scenario
from scripts.demand_service_report import correct_rounding
import scripts.vehicle_rule_test as vr

shared = vr.shared


def _checkpoint_label(group, replica):
    base = 'mappo-continue' if group in ('mappo-early', 'mappo-early+vehrule', 'mappo-unified') else group
    return f'{base}-r{replica}'


def load_records(root, p):
    records = {}
    for job in vr.jobs(p):
        name = vr.base.job_name(job)
        row = shared.cached_json(root / 'records' / (name + '.json'), p['fingerprint'])
        if row is None:
            raise ValueError('Incomplete study: ' + name)
        if (row['scenario'], row['seed'], row['group'], row['replica']) != job:
            raise ValueError('Wrong run identity: ' + name)
        if row['replica'] is not None:
            expect = p['models'][_checkpoint_label(row['group'], row['replica'])]['sha256']
            if row['checkpoint_sha256'] != expect:
                raise ValueError('Wrong model identity: ' + name)
        records[job] = row
    return records


def report(root, p):
    root = Path(root)
    records = load_records(root, p)
    scenarios = {}
    for scenario in p['evaluation_scenarios']:
        seeds = sorted(p['evaluation_seeds'])
        baselines = {b: [records[(scenario, s, b, None)] for s in seeds] for b in p['baselines']}
        learned = {}
        for arm in p['arms']:
            group = f'mappo-{arm}'
            learned[group] = [[records[(scenario, s, group, r)] for s in seeds]
                              for r in range(p['replicas'])]
        data = aggregate_scenario(baselines, learned, draws=p['bootstrap_draws'],
                                  contrasts=p['contrasts'])
        for arm_data in data['learned'].values():
            for comp in arm_data['comparisons'].values():
                correct_rounding(comp)
        for comp in data['paired_contrasts'].values():
            correct_rounding(comp)
        scenarios[scenario] = data

    summary = {'fingerprint': p['fingerprint'], 'schema': p['schema'],
               'arms': p['arms'], 'contrasts': p['contrasts'], 'scenarios': scenarios,
               'interpretation': ('Quick smoke validation only; not research evidence.'
                   if p['quick'] else
                   'Frozen-model development evaluation. descriptive_pass is a point estimate, '
                   'not a confirmatory statistical claim; reserved seeds 58001-58020 untouched.')}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')

    fmt = lambda v: 'NA' if v is None else f'{v:.2f}'
    lines = ['# Stage 1: vehicle-axis service rule (frozen model)', '',
        ('**Quick smoke only** — reduced horizon/one seed; execution check, not evidence.\n' if p['quick'] else ''),
        'New arm `mappo-early+vehrule` = continued-MAPPO + early-service pedestrian rule + '
        'larger-stopped-queue vehicle rule. No training. Compared vs. actuated and vs. the '
        'continued/early arms on matched demand. `descriptive_pass` = point estimates meet '
        'vehicle<0 AND all 12 crossings p95 no worse AND strict service gate clean.', '']
    for scenario, data in scenarios.items():
        lines += [f'## {scenario}', '',
            '| Arm | Vehicle loss s | Δ vs actuated s | Crossings no-worse /12 | Service fail pairs | descriptive_pass |',
            '|---|---:|---:|---:|---:|---|']
        ref = data['learned']
        for arm in p['arms']:
            g = f'mappo-{arm}'
            comp = ref[g]['comparisons']['actuated']
            correct_rounding(comp)
            pt = comp['point']
            lines.append(f"| {g} | {fmt(ref[g]['absolute']['vehicle_mean_time_loss'])} | "
                         f"{fmt(pt['vehicle_delta_seconds'])} | {pt['crossings_no_worse']} | "
                         f"{comp['service_failure_pairs']} | {comp['descriptive_pass']} |")
        lines.append('')
        for key, c in data['paired_contrasts'].items():
            lines.append(f"- **{key}**: Δvehicle {fmt(c['point']['vehicle_delta_seconds'])} s, "
                         f"crossings no-worse {c['point']['crossings_no_worse']}/12, "
                         f"service-fail-pairs {c['service_failure_pairs']}")
        lines.append('')
    (root / 'REPORT.md').write_text('\n'.join(lines))
    return summary
