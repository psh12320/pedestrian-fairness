"""Read-only analysis of the downloaded study; never imports the archived code."""
from pathlib import Path
import collections
import hashlib
import json
import math
import random
import statistics as st
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'runs/guided-exploration-results.zip'
PREFIX = 'runs/guided-exploration/'
z = zipfile.ZipFile(ARCHIVE)
read = lambda name: json.loads(z.read(PREFIX + name))
summary = read('summary.json')
p = summary['protocol']
seeds = p['evaluation_seeds']
groups = [f'{m}-{v}' for m in p['methods'] for v in p['variants']] + ['imitation']
checks = collections.Counter()


def check(ok, label):
    assert ok, label
    checks[label] += 1


def same(a, b, label):
    check(a == b if a is None or b is None else math.isclose(a, b, abs_tol=1e-8), label)


def q(values, fraction=.95):
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)] if values else None


def validate(row):
    v = row['vehicles']
    check(v['scheduled_count'] == v['completed_count'] + v['unfinished_count'] + v['uninserted_count'], 'vehicle_accounting')
    check(v['inserted_count'] == v['completed_count'] + v['unfinished_count'], 'inserted_accounting')
    check(len(row['crossings']) == 12, 'crossing_coverage')
    for c in row['crossings'].values():
        check(len(c['waits']) == c['entries'], 'crossing_entries')
        same(q(c['waits']), c['p95'], 'crossing_p95')
        same(st.mean(c['waits']) if c['waits'] else None, c['mean'], 'crossing_mean')
        check(all(w >= 0 for w in c['waits']), 'nonnegative_waits')
        check(c['unresolved_after_followup'] <= c['pending_at_cutoff'], 'pending_accounting')


def service_flags(a, b):
    va, vb = a['vehicles'], b['vehicles']
    return {
        'vehicle': va['completed_count'] < vb['completed_count'] or any(va[k] > vb[k] for k in ['unfinished_count', 'uninserted_count']),
        'person_trip': a['persons']['completed'] < b['persons']['completed'] or any(a['persons'][k] > b['persons'][k] for k in ['unfinished', 'uninserted']),
        'crossing_cutoff': any(c['pending_at_cutoff'] > b['crossings'][k]['pending_at_cutoff'] for k, c in a['crossings'].items()),
        'crossing_followup': any(c['unresolved_after_followup'] > b['crossings'][k]['unresolved_after_followup'] for k, c in a['crossings'].items()),
        'measurement': any(c['lost'] or c['unknown_cohort_entries'] or b['crossings'][k]['lost'] or b['crossings'][k]['unknown_cohort_entries'] for k, c in a['crossings'].items()),
    }


def verify_absolute(matrix, expected):
    rows = [x for rep in matrix for x in rep]
    for key in ['scheduled_count', 'completed_count', 'unfinished_count', 'uninserted_count']:
        same(st.mean(x['vehicles'][key] for x in rows), expected['vehicle_' + key], 'aggregate_vehicle_counts')
    same(st.mean(x['vehicles']['completed_mean_time_loss_seconds'] for x in rows), expected['vehicle_mean_time_loss'], 'aggregate_vehicle_loss')
    for key, source in [('pedestrian_pending_cutoff', 'pending_at_cutoff'), ('pedestrian_unresolved', 'unresolved_after_followup')]:
        same(st.mean(sum(c[source] for c in x['crossings'].values()) for x in rows), expected[key], 'aggregate_pedestrian_service')
    for crossing, value in expected['crossings'].items():
        waits = [[w for row in rep for w in row['crossings'][crossing]['waits']] for rep in matrix]
        check([len(w) for w in waits] == value['entries_per_replica'], 'pooled_crossing_entries')
        same(st.mean(q(w) for w in waits), value['mean_replica_pooled_p95'], 'pooled_crossing_p95')
        same(st.mean(st.mean(w) for w in waits), value['mean_wait'], 'pooled_crossing_mean')


def verify_comparison(a, b, expected):
    nr, ns = len(a), len(a[0])
    differences = [[a[r][s]['vehicles']['completed_mean_time_loss_seconds'] - b[r][s]['vehicles']['completed_mean_time_loss_seconds'] for s in range(ns)] for r in range(nr)]
    same(st.mean(x for row in differences for x in row), expected['point']['vehicle_delta_seconds'], 'paired_vehicle_delta')
    for c, value in expected['point']['crossing_p95_delta_seconds'].items():
        tails = [st.mean(q([w for row in rep for w in row['crossings'][c]['waits']]) for rep in matrix) for matrix in [a, b]]
        same(tails[0] - tails[1], value, 'paired_crossing_delta')
    flags = [service_flags(a[r][s], b[r][s]) for r in range(nr) for s in range(ns)]
    check(sum(any(f.values()) for f in flags) == expected['service_failure_pairs'], 'service_gate')
    for r in range(nr):
        for s in range(ns):
            check(a[r][s]['demand_sha256'] == b[r][s]['demand_sha256'], 'paired_demand')
    if expected['exploratory_intervals']:
        rng = random.Random(424672)
        draws = []
        for _ in range(expected['bootstrap_requested_draws']):
            reps = rng.choices(range(nr), k=nr)
            samples = rng.choices(range(ns), k=ns)
            draws.append(st.mean(differences[r][s] for r in reps for s in samples))
        for got, want in zip([q(draws, .025), q(draws, .975)], expected['exploratory_intervals']['vehicle_delta_seconds']):
            same(got, want, 'vehicle_bootstrap_interval')
    return {k: sum(f[k] for f in flags) for k in flags[0]} | {'cutoff_only': sum(f['crossing_cutoff'] and not any(v for k, v in f.items() if k != 'crossing_cutoff') for f in flags)}


manifest = json.loads(z.read('manifest.json'))
check(set(z.namelist()) == set(manifest) | {'manifest.json'}, 'manifest_coverage')
for name, digest in manifest.items():
    check(hashlib.sha256(z.read(name)).hexdigest() == digest, 'manifest_hash')
for name, digest in p['source'].items():
    check(hashlib.sha256(z.read(name)).hexdigest() == digest, 'archived_source_hash')
source_changes = [name for name, digest in p['source'].items() if not (ROOT / name).exists() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
for name in z.namelist():
    if name.startswith(PREFIX) and name.endswith('.json'):
        check(json.loads(z.read(name))['fingerprint'] == p['fingerprint'], 'result_fingerprint')

training = {}
model_hashes = {}
training_demands = {}
validations = {}
for group in groups:
    for replica in range(p['replicas']):
        label = f'{group}-r{replica}'
        if group == 'imitation':
            data = read(f'teacher/final-r{replica}.json')
            model_path = PREFIX + f'teacher/final-r{replica}.pt'
            for row in data['validation']:
                validate(row['result'])
        else:
            data = read(f'models/{label}/complete.json')
            model_path = PREFIX + f'models/{label}/final.pt'
            curve = read(f'models/{label}/learning.json')
            history = curve['history']
            check(len(history) == data['episodes'] == data['update_batches'] == p['episodes'], 'training_budget')
            expected = [(i + 1, p['training_scenarios'][i % 4], p['training_seed_base'] + replica * 1000 + i) for i in range(p['episodes'])]
            check([(h['episodes'], h['scenario'], h['seed']) for h in history] == expected, 'training_schedule')
            demand = [(h['demand_sha256'], h['network_sha256']) for h in history]
            if replica in training_demands:
                check(demand == training_demands[replica], 'matched_training_demands')
            training_demands[replica] = demand
            for h in history:
                for stats in h['update'].values():
                    check(stats['normalizer_count'] == h['episodes'] * 599, 'normalizer_sample_count')
            check({(v['episodes'], v['seed']) for v in curve['validations']} == {(e, s) for e in p['validation_episodes'] for s in p['validation_seeds']}, 'validation_coverage')
            for row in curve['validations']:
                validate(row['result'])
            validations[label] = curve['validations']
            training[label] = {
                'free_decisions_per_junction': {c: sum(h['diagnostics'][c]['free_decisions'] for h in history) for c in ['C0', 'C1', 'C2']},
                'recorded_training_wall_seconds': sum(h['seconds'] for h in history),
                'mean_actor_gradient_norm': st.mean(v['actor_gradient_norm'] for h in history for v in h['update'].values()),
                'mean_critic_gradient_norm': st.mean(v['critic_gradient_norm'] for h in history for v in h['update'].values()),
            }
        digest = hashlib.sha256(z.read(model_path)).hexdigest()
        check(digest == data['checkpoint_sha256'], 'model_hash')
        model_hashes[label] = digest

evaluations = {}
service = {}
problem_runs = []
diagnostics = {}
person_metrics = {}
for scenario in p['evaluation_scenarios']:
    bases = {c: [] for c in p['baselines']}
    network = {}
    for seed in seeds:
        data = read(f'baselines/{scenario}-{seed}.json')
        check(sorted(x['controller'] for x in data['records']) == sorted(bases), 'baseline_coverage')
        for row in data['records']:
            validate(row['result'])
            bases[row['controller']].append(row['result'])
            if seed in network:
                check(network[seed] == row['network_sha256'], 'baseline_network')
            network[seed] = row['network_sha256']
    sc = summary['scenarios'][scenario]
    matrices = {}
    for name, rows in bases.items():
        matrices[name] = [rows]
        verify_absolute([rows], sc['baselines'][name]['absolute'])
        verify_comparison([rows], [bases['actuated']], sc['baselines'][name]['versus_actuated'])
    service[scenario] = {}
    diagnostics[scenario] = {}
    person_metrics[scenario] = {}
    for group in groups:
        matrix = []
        diag_rows = []
        for replica in range(p['replicas']):
            rows = []
            for seed in seeds:
                label = f'{group}-r{replica}'
                data = read(f'evaluation/{label}-{scenario}-{seed}.json')
                check(data['checkpoint_sha256'] == model_hashes[label], 'evaluation_checkpoint')
                check(data['network_sha256'] == network[seed], 'evaluation_network')
                validate(data['result'])
                rows.append(data['result'])
                diag_rows.extend(data['diagnostics'].values())
                for d in data['diagnostics'].values():
                    same(sum(d['phase_seconds'].values()), 2400, 'phase_duration_total')
                unresolved = sum(c['unresolved_after_followup'] for c in data['result']['crossings'].values())
                if unresolved:
                    problem_runs.append({'label': label, 'scenario': scenario, 'seed': seed, 'unresolved': unresolved,
                                         'persons': data['result']['persons'], 'diagnostics': data['diagnostics'],
                                         'crossings': {k: {field: c[field] for field in ['entries', 'p95', 'pending_at_cutoff', 'unresolved_after_followup']} for k, c in data['result']['crossings'].items()}})
            matrix.append(rows)
        matrices[group] = matrix
        verify_absolute(matrix, sc['learned'][group]['absolute'])
        for baseline in bases:
            flags = verify_comparison(matrix, [bases[baseline]] * p['replicas'], sc['learned'][group]['comparisons'][baseline])
            if baseline == 'actuated':
                service[scenario][group] = flags
        diagnostics[scenario][group] = {k: st.mean(d[k] for d in diag_rows) for k in ['mean_entropy', 'sampled_teacher_agreement', 'teacher_probability']}
        person_metrics[scenario][group] = {k: st.mean(row['persons'][k] for rep in matrix for row in rep) for k in ['scheduled', 'completed', 'unfinished', 'uninserted']}
    for label, expected in sc['paired_ablation'].items():
        a, b = label.split(':')[1].split('-vs-')
        verify_comparison(matrices['mappo-' + a], matrices[b if b == 'imitation' else 'mappo-' + b], expected)
    evaluations[scenario] = matrices

validation_trend = {}
for variant in p['variants']:
    group = 'mappo-' + variant
    validation_trend[group] = {}
    for episode in p['validation_episodes']:
        rows = [v for r in range(p['replicas']) for v in validations[f'{group}-r{r}'] if v['episodes'] == episode]
        validation_trend[group][episode] = {
            'vehicle_time_loss': st.mean(v['result']['vehicles']['completed_mean_time_loss_seconds'] for v in rows),
            'entropy': st.mean(d['mean_entropy'] for v in rows for d in v['diagnostics'].values()),
            'teacher_agreement': st.mean(d['sampled_teacher_agreement'] for v in rows for d in v['diagnostics'].values()),
        }

audit = {'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), 'checks': dict(checks),
         'check_count': sum(checks.values()), 'source_changed_since_run': source_changes,
         'training': training, 'validation_trend': validation_trend, 'evaluation_diagnostics': diagnostics,
         'service_failure_components_vs_actuated': service, 'person_trip_means': person_metrics,
         'unresolved_learned_runs': problem_runs,
         'limitations': ['Original simulator traces and demonstration tensors are absent; no physical-safety replay was performed.',
                         'Recomputed point estimates, service gates and vehicle bootstrap intervals; crossing intervals were inspected, not independently regenerated.',
                         'No SLURM job log is included; artifact completeness is verified independently of scheduler status.']}
(OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
for filename in ['summary.json', 'protocol.json', 'REPORT.md']:
    (OUT / ('archived_' + filename)).write_bytes(z.read(PREFIX + filename))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
names = ['fixed', 'pressure', 'mappo-control', 'mappo-age', 'mappo-rnd', 'mappo-guided', 'imitation']
labels = ['Fixed time', 'Pressure', 'MAPPO control', '+ Waiting-age penalty', '+ Age and RND', '+ Age and guidance', 'Imitation only']
colors = ['#94a3b8', '#64748b', '#2563eb', '#d97706', '#7c3aed', '#059669', '#0d9488']
medium = summary['scenarios']['vmedium-pmedium']
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={'width_ratios': [1, 1.05]})
for idx, (name, label, color) in enumerate(zip(names, labels, colors)):
    comparison = medium['baselines'][name]['versus_actuated'] if name in medium['baselines'] else medium['learned'][name]['comparisons']['actuated']
    for ax, key in zip(axes, ['vehicle_delta_seconds', 'worst_crossing_delta_seconds']):
        value = comparison['point'][key]
        label_edge = value
        ax.barh(idx, value, color=color, height=.58, alpha=.88)
        if key == 'vehicle_delta_seconds' and comparison['exploratory_intervals']:
            lo, hi = comparison['exploratory_intervals'][key]
            ax.plot([lo, hi], [idx, idx], color='#172033', linewidth=1.7)
            ax.scatter([lo, hi], [idx, idx], marker='|', color='#172033', s=55)
            label_edge = max(value, hi) if value >= 0 else min(value, lo)
        ax.text(label_edge + (.6 if value >= 0 else -.6), idx, f'{value:+.2f}', va='center', ha='left' if value >= 0 else 'right', fontsize=9)
axes[0].set_yticks(range(len(labels)), labels)
axes[1].set_yticks(range(len(labels)), [''] * len(labels))
for ax in axes:
    ax.invert_yaxis()
    ax.axvline(0, color='#172033', linewidth=1)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    ax.set_xlabel('Difference from pedestrian-actuated control (seconds)')
axes[0].set_title('Completed-vehicle time loss\nLower is better', loc='left', fontweight='bold')
axes[1].set_title('Largest crossing p95 increase\nZero or below required at every crossing', loc='left', fontweight='bold')
axes[0].set_xlim(-24, 6)
axes[1].set_xlim(-1, 50)
fig.suptitle('Medium demand: guidance closes the pedestrian gap, but gives up vehicle gains', fontsize=13, fontweight='bold', x=.03, ha='left')
fig.text(.03, .025, '3 training initialisations × 6 demand seeds. Vehicle bars show paired exploratory 95% intervals where available.\nCrossing p95s are pooled within each replica, then averaged. Service counts are assessed separately. Development study.', fontsize=8.5, color='#475569')
fig.tight_layout(rect=(0, .09, 1, .9))
for suffix in ['png', 'svg']:
    fig.savefig(OUT / f'medium_tradeoff.{suffix}', dpi=170, facecolor='white')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for variant, color in zip(p['variants'], ['#2563eb', '#d97706', '#059669', '#7c3aed']):
    trend = validation_trend['mappo-' + variant]
    xs = list(trend)
    for ax, key in zip(axes, ['vehicle_time_loss', 'teacher_agreement']):
        ax.plot(xs, [trend[x][key] for x in xs], marker='o', label=variant, color=color)
for ax in axes:
    ax.set_xlabel('Training episode')
    ax.set_xticks([32, 64, 128])
    ax.grid(alpha=.15)
axes[0].set_ylabel('Mean completed-vehicle time loss (seconds)')
axes[1].set_ylabel('Sampled agreement with actuated teacher')
axes[1].set_ylim(.35, 1)
axes[1].legend(frameon=False)
fig.suptitle('Medium-demand validation: unguided learning changes late; guidance stays near its teacher', fontsize=12)
fig.text(.04, .015, 'Descriptive means: 3 initialisations × 2 validation seeds. Final checkpoints used; these curves do not select a winner.', fontsize=8)
fig.tight_layout(rect=(0, .055, 1, .92))
for suffix in ['png', 'svg']:
    fig.savefig(OUT / f'validation_trends.{suffix}', dpi=170, facecolor='white')
plt.close(fig)

print(json.dumps({'checks_passed': sum(checks.values()), 'models': len(model_hashes), 'evaluation_records': sum(1 for n in z.namelist() if '/evaluation/' in n), 'source_changed': source_changes, 'unresolved_runs': [{k: d[k] for k in ['label', 'scenario', 'seed', 'unresolved']} for d in problem_runs], 'output': str(OUT)}, indent=2))
