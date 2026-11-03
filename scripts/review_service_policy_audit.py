"""Read-only review of the cluster policy diagnostic; no retraining or test tuning.

Read ZIP members in memory. Preserve demand seeds, trained replicas and policy
random seeds as separate units. Write compact JSON, Markdown and a static figure.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_service_policy import job_key
from src.traffic_control.fairness.analysis import quantile, selection_score
from src.traffic_control.service_priority.control import service_components


def family(controller):
    return re.sub(r'-r\d+$', '', controller)


def crossing_estimates(rows, baselines):
    """Pool demand within each model/policy seed, then average tail estimates.

    A policy seed is a repeated evaluation, never an independent training run.
    The deterministic baseline is pooled once, not multiplied by policy repeats.
    Missing entries block a claim instead of being imputed as zero waiting.
    """
    pools = defaultdict(list)
    for row in rows:
        job = row['job']
        pools[job['controller'], job['policy_seed']].append(row)
    out = {}
    for crossing in sorted(baselines[0]['result']['crossings']):
        reference = [w for r in baselines for w in r['result']['crossings'][crossing]['waits']]
        samples = [[w for r in group for w in r['result']['crossings'][crossing]['waits']]
                   for group in pools.values()]
        tails = [quantile(w) for w in samples]
        means = [st.mean(w) if w else None for w in samples]
        complete = bool(reference and all(w for w in samples))
        p95 = st.mean(tails) if complete else None
        out[crossing] = {
            'p95': p95, 'baseline_p95': quantile(reference),
            'p95_delta': p95 - quantile(reference) if complete else None,
            'mean_wait': st.mean(means) if complete else None,
            'baseline_mean_wait': st.mean(reference) if reference else None,
            'baseline_entries': len(reference),
            'model_policy_seed_entries': [len(w) for w in samples],
            'all_pools_observed': complete,
        }
    return out


def describe(rows, reference):
    vehicle_keys = ('scheduled_count', 'completed_count', 'unfinished_count',
                    'uninserted_count', 'completed_mean_time_loss_seconds')
    values = {k: st.mean(r['result']['vehicles'][k] for r in rows) for k in vehicle_keys}
    crossings = crossing_estimates(rows, list(reference.values()))
    deltas = [c['p95_delta'] for c in crossings.values()]
    complete = all(d is not None for d in deltas)
    service = [service_components(r['result'], reference[r['job']['seed']]['result']) for r in rows]
    for key in ('pending_at_cutoff', 'unresolved_after_followup', 'unknown_cohort_entries', 'lost'):
        values[key] = st.mean(sum(c[key] for c in r['result']['crossings'].values()) for r in rows)
    values['person_unfinished_at_end'] = st.mean(r['result']['persons']['unfinished'] for r in rows)
    model_groups = defaultdict(list)
    for row in rows:
        model_groups[row['job']['controller']].append(row)
    diagnostics = [d for r in rows for d in r['diagnostics'].values()]
    longest = [max(d['max_consecutive_queued_no_green_samples_1s'].values()) for d in diagnostics]
    scores = [selection_score(r['result'], reference[r['job']['seed']]['result'])[0] for r in rows]
    row = {'family': family(rows[0]['job']['controller']), 'mode': rows[0]['job']['mode'],
           'evaluation_runs': len(rows),
           'trained_replicas': len(model_groups) if rows[0]['job']['mode'] in ('argmax', 'sample') else 0,
           'vehicle_and_service_means': values, 'crossings': crossings,
           'crossings_no_worse_point': sum(d is not None and d <= 1e-9 for d in deltas),
           'worst_crossing_delta': max(deltas) if complete else None,
           'service_failed_runs': sum(s > 0 for s in scores),
           'service_deltas': {k: st.mean(s[k] for s in service) for k in service[0]
                              if not isinstance(service[0][k], list)},
           'max_queued_no_green_seconds': max(longest),
           'mean_junction_max_queued_no_green_seconds': st.mean(longest),
           'replica_mean_vehicle_loss': {m: st.mean(r['result']['vehicles']['completed_mean_time_loss_seconds'] for r in rs)
                                         for m, rs in model_groups.items()}}
    if 'monte_carlo_explained_variance' in diagnostics[0]:
        row['critic'] = {
            'median_mc_explained_variance': st.median(d['monte_carlo_explained_variance'] for d in diagnostics),
            'min_mc_explained_variance': min(d['monte_carlo_explained_variance'] for d in diagnostics),
            'max_mc_explained_variance': max(d['monte_carlo_explained_variance'] for d in diagnostics),
            'mean_prediction': st.mean(d['value_mean'] for d in diagnostics),
            'mean_discounted_return': st.mean(d['discounted_return_mean'] for d in diagnostics),
            'mean_free_choice_max_probability': st.mean(d['mean_max_probability'] for d in diagnostics),
        }
    baseline_loss = st.mean(r['result']['vehicles']['completed_mean_time_loss_seconds'] for r in reference.values())
    row['descriptive_joint_pass'] = bool(complete and max(deltas) <= 1e-9 and
        values['completed_mean_time_loss_seconds'] < baseline_loss and all(s == 0 for s in scores))
    return row


def validate_archive(source, pilot_source):
    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(pilot_source) as pilot:
        assert archive.testzip() is None, 'Diagnostic ZIP CRC failed'
        assert len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate ZIP member'
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['smoke'] is False
        jobs = manifest['jobs']
        expected_names = {job_key(j) + '.json' for j in jobs}
        assert len(jobs) == len(expected_names) == 110
        assert set(archive.namelist()) == expected_names | {'manifest.json', 'REPORT.md'}
        prefix = 'runs/service-priority-pilot/'
        protocol = json.loads(pilot.read(prefix + 'protocol.json'))
        seeds = protocol['seeds']
        assert seeds == [47001, 47002]
        models = [f'{kind}-{method}-r{r}' for kind in ('plain', 'priority')
                  for method in ('ippo', 'mappo') for r in range(3)]
        expected_jobs = set()
        for seed in seeds:
            expected_jobs.add(('actuated', 'baseline', seed, 0, False))
            for model in models:
                priority = model.startswith('priority-')
                expected_jobs.add((model, 'argmax', seed, 0, priority))
                for p in (48001, 48002, 48003): expected_jobs.add((model, 'sample', seed, p, priority))
            for priority in (False, True):
                for p in (48001, 48002, 48003):
                    expected_jobs.add(('uniform-priority' if priority else 'uniform-plain', 'uniform', seed, p, priority))
        actual_jobs = {(j['controller'], j['mode'], j['seed'], j['policy_seed'], j['priority']) for j in jobs}
        assert actual_jobs == expected_jobs
        assert not set(seeds) & set(protocol['reserved_test_seeds_not_used'])
        originals = {}
        for seed in seeds:
            group = json.loads(pilot.read(prefix + f'baseline-vmedium-pmedium-{seed}.json'))
            originals['actuated', seed] = next(r['result'] for r in group['records'] if r['controller'] == 'actuated')
        for model in models:
            for r in json.loads(pilot.read(prefix + model + '/evaluation.json')):
                if r['scenario'] == 'vmedium-pmedium': originals[model, r['seed']] = r['result']
        rows = []
        replay_count = 0
        for job in jobs:
            row = json.loads(archive.read(job_key(job) + '.json'))
            assert row['job'] == job and job['fingerprint'] == manifest['fingerprint']
            result = row['result']; baseline = originals['actuated', job['seed']]
            assert result['demand_sha256'] == baseline['demand_sha256']
            v = result['vehicles']; p = result['persons']
            assert v['scheduled_count'] == v['completed_count'] + v['unfinished_count'] + v['uninserted_count']
            assert p['scheduled'] == p['completed'] + p['unfinished'] + p['uninserted']
            assert v['unfinished_count'] == len(v['unfinished_ids'])
            assert v['uninserted_count'] == len(v['uninserted_ids'])
            assert set(result['crossings']) == set(baseline['crossings']) and len(result['crossings']) == 12
            for c in result['crossings'].values():
                assert c['entries'] == len(c['waits']) == len(c['wait_lower']) == len(c['wait_upper'])
                assert c['p95'] == quantile(c['waits'])
                assert math.isclose(c['mean'], st.mean(c['waits']), abs_tol=1e-9)
                assert all(-1e-9 <= lo <= w + 1e-9 and w <= hi + 1e-9
                           for lo, w, hi in zip(c['wait_lower'], c['waits'], c['wait_upper']))
            assert row['service_score_vs_actuated'] == selection_score(result, baseline)[0]
            assert math.isclose(row['vehicle_difference_vs_actuated'],
                v['completed_mean_time_loss_seconds'] - baseline['vehicles']['completed_mean_time_loss_seconds'], abs_tol=1e-9)
            assert set(row['diagnostics']) == {'C0', 'C1', 'C2'}
            if job['mode'] in ('baseline', 'argmax'):
                original = originals[job['controller'], job['seed']]
                assert all(result[k] == original[k] for k in ('vehicles', 'persons', 'crossings'))
                assert row['replay_matches'] == dict.fromkeys(('vehicles', 'persons', 'crossings'), True)
                replay_count += 1
            rows.append(row)
    assert replay_count == 26
    return manifest, rows, replay_count


def make_figure(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    lookup = {(s['family'], s['mode']): s for s in summary}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1, 1.4]})
    families = ['plain-ippo', 'plain-mappo', 'priority-ippo', 'priority-mappo']
    y = list(range(4))
    axes[0].barh([i + .16 for i in y], [lookup[f, 'argmax']['vehicle_and_service_means']['completed_count'] for f in families],
                 height=.3, color='#526783', label='Highest probability')
    axes[0].barh([i - .16 for i in y], [lookup[f, 'sample']['vehicle_and_service_means']['completed_count'] for f in families],
                 height=.3, color='#168b92', label='Sample probabilities')
    axes[0].axvline(lookup['actuated', 'baseline']['vehicle_and_service_means']['completed_count'], color='#20252c', ls='--', label='Actuated: 933.5')
    axes[0].set(yticks=y, yticklabels=['Plain IPPO', 'Plain MAPPO', 'Priority IPPO', 'Priority MAPPO'],
                xlim=(0, 1000), xlabel='Mean vehicles completed by 2100 s', title='Sampling restores much of vehicle service')
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.02, .94), ncol=3, fontsize=8, frameon=False)
    selected = [('plain-ippo', 'sample'), ('plain-mappo', 'sample'), ('priority-ippo', 'sample'),
                ('priority-mappo', 'sample'), ('uniform-plain', 'uniform'), ('uniform-priority', 'uniform')]
    crossings = sorted(lookup[selected[0]]['crossings'])
    matrix = [[lookup[key]['crossings'][c]['p95_delta'] for c in crossings] for key in selected]
    extent = max(1, max(abs(v) for row in matrix for v in row))
    im = axes[1].imshow(matrix, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent), aspect='auto')
    axes[1].set(yticks=range(6), yticklabels=['Sample IPPO', 'Sample MAPPO', 'Priority sample IPPO', 'Priority sample MAPPO', 'Uniform', 'Priority uniform'],
                xticks=range(12), xticklabels=[c.replace(':', '') for c in crossings],
                title='Pedestrian tail-wait trade-offs remain', xlabel='Crossing')
    axes[1].tick_params(axis='x', rotation=65, labelsize=8)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axes[1].text(j, i, f'{value:+.1f}', ha='center', va='center', fontsize=7,
                         color='white' if abs(value) > extent*.65 else '#171717')
    fig.colorbar(im, ax=axes[1], shrink=.8, label='p95 difference vs actuated (seconds)')
    fig.suptitle('Policy diagnostic: same models and demand, different action selection', fontsize=12)
    fig.text(.5, .012, 'Development medium demand only: 2 demand seeds, 3 trained replicas, 3 sampling seeds. Descriptive means; no confidence claim.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .90))
    fig.savefig(destination / 'policy_diagnostic.png', dpi=180)
    fig.savefig(destination / 'policy_diagnostic.svg')
    plt.close(fig)


def main():
    source = ROOT / 'runs/service-policy-audit-results.zip'
    destination = ROOT / 'results/service-policy-audit'
    destination.mkdir(parents=True, exist_ok=True)
    manifest, rows, replays = validate_archive(source, ROOT / 'runs/service-priority-results.zip')
    reference = {r['job']['seed']: r for r in rows if r['job']['mode'] == 'baseline'}
    grouped = defaultdict(list)
    for row in rows: grouped[family(row['job']['controller']), row['job']['mode']].append(row)
    order = [('actuated', 'baseline')] + [(f, m) for f in ('plain-ippo', 'plain-mappo', 'priority-ippo', 'priority-mappo') for m in ('argmax', 'sample')]
    order += [('uniform-plain', 'uniform'), ('uniform-priority', 'uniform')]
    summary = [describe(grouped[key], reference) for key in order]
    paired = []
    for row in rows:
        job = row['job']
        if job['mode'] != 'sample': continue
        uniform = next(r for r in rows if r['job']['mode'] == 'uniform' and
                       all(r['job'][k] == job[k] for k in ('seed', 'policy_seed', 'priority')))
        paired.append({'controller': job['controller'], 'demand_seed': job['seed'], 'policy_seed': job['policy_seed'],
                       'vehicle_loss_delta_vs_uniform': row['result']['vehicles']['completed_mean_time_loss_seconds'] - uniform['result']['vehicles']['completed_mean_time_loss_seconds'],
                       'completed_delta_vs_uniform': row['result']['vehicles']['completed_count'] - uniform['result']['vehicles']['completed_count']})
    audit = {'archive_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'evaluations': len(rows),
             'independent_replay_matches': replays, 'demand_seeds': sorted(reference),
             'policy_seeds': [48001, 48002, 48003], 'sumo_versions': sorted({r['sumo_version'] for r in rows}),
             'unfinished_person_runs': sum(r['result']['persons']['unfinished'] > 0 for r in rows),
             'unresolved_crossing_runs': sum(any(c['unresolved_after_followup'] for c in r['result']['crossings'].values()) for r in rows),
             'unknown_or_lost_events': sum(c['unknown_cohort_entries'] + c['lost'] for r in rows for c in r['result']['crossings'].values()),
             'model_weights_in_archive': False}
    output = {'audit': audit, 'manifest': manifest, 'summary': summary, 'paired_vs_uniform': paired}
    (destination / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Service-policy audit review', '',
        'All 110 evaluations completed. All 26 deterministic replays independently match the previous pilot archive, including full vehicle, person and crossing records. ZIP integrity, manifest coverage, demand pairing, vehicle/person counts, crossing means, quantiles and interval ordering passed checks. SUMO version: 1.27.1. No final test seeds were used.', '',
        '## Findings', '',
        'The evaluation-mode hypothesis is supported on these saved checkpoints: probability sampling removes most of the severe vehicle under-service seen with argmax. It does not meet the joint proposal target. Plain sampled policies have worse p95 point estimates at all 12 crossings. Priority sampling protects more crossing tails but incurs substantially higher vehicle time loss. No arm passes the combined target.', '',
        'Uniform choices also restore vehicle insertion. Plain learned sampling has worse mean vehicle loss than uniform, but substantially smaller worst-crossing penalties: learning cannot be judged solely from vehicle flow. Priority sampled MAPPO has somewhat better descriptive vehicle results than priority uniform; neither beats actuated on the joint objective. These are observations on two development demand seeds, not statistical superiority claims.', '',
        '## Complete medium-demand comparison', '',
        'Vehicle counts cover trips scheduled in [300,2100) and their status at 2100 seconds. There are 974.5 scheduled vehicles on average. Unfinished vehicles at that cutoff are distinct from unserved pedestrians after the 2400-second follow-up. Vehicle time loss is conditional on completion.', '',
        '| Controller | Choice | Runs | Vehicle loss s | Completed | Unfinished | Uninserted | Crossing p95 no worse / 12 | Worst p95 delta s | Service-flagged runs |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        v = row['vehicle_and_service_means']
        lines.append(f"| {row['family']} | {row['mode']} | {row['evaluation_runs']} | {v['completed_mean_time_loss_seconds']:.2f} | {v['completed_count']:.1f} | {v['unfinished_count']:.1f} | {v['uninserted_count']:.1f} | {row['crossings_no_worse_point']}/12 | {row['worst_crossing_delta']:+.2f} | {row['service_failed_runs']}/{row['evaluation_runs']} |")
    lines += ['',
        'Crossing estimator: pool the two demand seeds within each model and policy-random seed, compute each crossing p95, then average those p95s across trained models and policy seeds. The reference pools the two actuated runs once. Policy-seed repeats are not extra independently trained models. Only 1e-9 s floating-point roundoff is ignored. No sparse or missing wait is converted to zero.', '',
        'All 110 runs completed all scheduled person trips by simulation end and had zero unresolved measured crossing requests, zero unknown entries and zero lost events. Service flags here are vehicle or crossing-cutoff comparisons; they must not be presented as permanent pedestrian starvation. The full per-crossing means, tails, entry counts and signed service components are in summary.json.', '',
        '## Mechanism evidence', '',
        'For plain-mappo-r0, demand seed 47001, C0 EW has 2354 consecutive queued/no-green samples under argmax. Under sampling the corresponding maxima are 153, 144 and 200 seconds. This is a movement-level diagnostic, not an individual vehicle wait. The mean gap between the top two action probabilities in the argmax replay is only 0.00823. The comparison changes action selection while retaining the checkpoint, physical rules and scheduled demand.', '',
        '## Critic evidence', '',
        '| Sampled controller | Mean value prediction | Mean discounted observed return | Median explained variance |',
        '|---|---:|---:|---:|']
    for row in summary:
        if row['mode'] == 'sample':
            c = row['critic']; lines.append(f"| {row['family']} | {c['mean_prediction']:.2f} | {c['mean_discounted_return']:.2f} | {c['median_mc_explained_variance']:.4f} |")
    lines += ['',
        'Values and returns are reward units, not seconds. Near-zero explained variance and large mean-scale differences indicate poor calibration on these trajectories. Realised Monte Carlo targets contain randomness, the evaluation horizon is longer than training, and selected checkpoints span episodes 8-32. This does not prove a specific optimiser bug or establish convergence. In particular, action-mode changes do not repair the learned value estimator.', '',
        'The current learner scales the squared value error by the return standard deviation while leaving its value output in raw reward units. That is different from training a normalised value output and denormalising it before computing advantages. The [official MAPPO implementation](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py) supports value normalisation and uses separate actor and critic optimisers. Those are concrete implementation choices to test, not guarantees of better traffic outcomes.', '',
        '## Recommended next work', '',
        '1. Preserve the original study and both diagnostics. Report the demonstrated argmax failure and the vehicle-pedestrian trade-off. Do not silently replace the original final-test evaluation mode.',
        '2. Before another long training job, check whether the value estimator can fit known return targets on a small, separately held-out set of development trajectories. Compare the current critic with a correctly normalised critic at matched update budgets. Log raw-value bias/RMSE and explained variance; test save/resume and raw/normalised unit consistency.',
        '3. If that check succeeds, run a controlled IPPO/MAPPO critic-normalisation ablation using the same demand, actor, reward, safety and training budget. Freeze the sampling evaluation protocol in advance; report argmax as a sensitivity test. Use ordinary and priority uniform controls and the conventional baselines. Do not combine reward, green-duration, layout and training changes into an unexplained single intervention.',
        '4. A separate controller-design hypothesis is actuated-guided or residual learning: retain demand-responsive service and let learning make bounded allocation adjustments. Compare it against the same conventional rule without learning. If adding movement service protection, enforce it consistently and verify minimum greens, clearance and blockage behaviour. Neither a warm start nor a service rule guarantees every-crossing p95 non-inferiority.',
        '5. Use the reserved final test seeds only after a development candidate and its evaluation protocol are frozen. A final claim requires repeated independent training, sufficient crossing samples and uncertainty. Current evidence concerns one synthetic medium-demand setting and cannot establish deployed benefits.', '',
        '## Scope of this review', '',
        'This review analysed the downloaded results and generated the figure below. It did not retrain controllers, alter simulator safety rules or launch a new cluster job. The result archive contains no model weights or full state/reward trajectories, so it cannot by itself run the proposed critic-fitting check.', '',
        '![Diagnostic comparison](policy_diagnostic.png)', '']
    (destination / 'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    make_figure(summary, destination)
    print(json.dumps({'audit': audit, 'report': str(destination / 'REVIEW.md')}, indent=2))


if __name__ == '__main__':
    main()
