"""Validate and review the four downloaded fixed-policy critic diagnostics.

No weight loading, extraction, training, or experiment selection. This verifies
reported sufficient statistics, not the absent raw trajectory tensors.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BEHAVIORS = ('plain-mappo-r0', 'priority-mappo-r0')
METHODS = ('ippo', 'mappo')
SEEDS = (49301, 49302, 49303)


def close(a, b, tolerance=1e-9):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)


def check_metrics(metric):
    assert isinstance(metric['n'], int) and metric['n'] > 0
    assert all(math.isfinite(v) for v in metric.values())
    assert metric['rmse'] >= abs(metric['bias']) - 1e-5
    close(metric['bias'], metric['prediction_mean'] - metric['target_mean'], 1e-4)
    close(metric['mse_ratio_to_training_mean'], (metric['rmse']/metric['constant_rmse'])**2)
    # Original statistics used float32 tensors; recombining summaries needs a
    # small floating-point tolerance, unlike report aggregations below.
    ev = 1 - (metric['rmse']**2 - metric['bias']**2) / metric['target_sd']**2
    close(metric['explained_variance'], ev, 1e-4)


def validate(z, behavior, stage):
    prefix = f'runs/critic-{stage}-{behavior}/'
    protocol = json.loads(z.read(prefix+'protocol.json'))
    summary = json.loads(z.read(prefix+'summary.json'))
    assert summary['protocol'] == protocol
    assert protocol['behavior'] == behavior
    assert protocol['initial_seeds'] == list(SEEDS)
    assert protocol['training_demand_seeds'] == [49101, 49102, 49103, 49104]
    checks = [49201, 49202] if stage == 'fit' else [49401, 49402]
    assert protocol['check_demand_seeds'] == checks
    assert protocol['budgets'] == [128, 512] and protocol['configuration']['end_seconds'] == 600
    assert protocol['gamma'] == .99 and protocol['reward_weight'] == 3
    for path, digest in protocol['source'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    unhashed = dict(protocol); unhashed.pop('fingerprint')
    assert hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode()).hexdigest() == protocol['fingerprint']
    if stage == 'horizon':
        parent = json.loads(z.read(f'runs/critic-fit-{behavior}/protocol.json'))
        assert protocol['parent_fingerprint'] == parent['fingerprint']
        assert protocol['checkpoint_sha256'] == parent['checkpoint_sha256']
        assert set(protocol['trajectory_hashes']) == {'49101', '49102', '49103', '49104'}
        assert hashlib.sha256((ROOT/'scripts/critic_horizon_check.py').read_bytes()).hexdigest() == protocol['horizon_script_sha256']
    groups = []
    for method in METHODS:
        for seed in SEEDS:
            group = json.loads(z.read(prefix+f'fit-{method}-{seed}.json'))
            assert group['fingerprint'] == protocol['fingerprint']
            assert group['method'] == method and group['initial_seed'] == seed
            option = 'mode' if stage == 'fit' else 'clock'
            variants = ('raw-scaled-loss', 'normalized-output') if stage == 'fit' else (False, True)
            expected = {(tls, mode, step) for tls in ('C0', 'C1', 'C2') for mode in variants for step in (0, 128, 512)}
            assert len(group['results']) == len(expected) == 18
            assert {(r['junction'], r[option], r['updates']) for r in group['results']} == expected
            for row in group['results']:
                a = row['assessments']
                assert a['training']['n'] == 2396 and a['heldout']['n'] == 1198
                for seed_id in checks: assert a[f'seed-{seed_id}']['n'] == 599
                for metric in a.values(): check_metrics(metric)
                per_seed = [a[f'seed-{s}'] for s in checks]
                for key in ('bias', 'prediction_mean', 'target_mean'):
                    close(a['heldout'][key], st.mean(m[key] for m in per_seed), 1e-4)
                close(a['heldout']['rmse']**2, st.mean(m['rmse']**2 for m in per_seed), 1e-4)
            if stage == 'horizon':
                for tls in ('C0', 'C1', 'C2'):
                    initial = [r['assessments'] for r in group['results'] if r['updates'] == 0 and r['junction'] == tls]
                    assert initial[0] == initial[1], 'Clock pair did not start with the same predictions'
            groups.append(group)
    fields = {'mean_holdout_mse_ratio': ('heldout', 'mse_ratio_to_training_mean'),
              'mean_train_mse_ratio': ('training', 'mse_ratio_to_training_mean'),
              'mean_holdout_ev': ('heldout', 'explained_variance'), 'mean_holdout_bias': ('heldout', 'bias')}
    if stage == 'fit':
        fields.update(mean_interior_mse_ratio=('interior', 'mse_ratio_to_training_mean'),
                      mean_free_decision_mse_ratio=('free_decisions', 'mse_ratio_to_training_mean'))
    for reported in summary['summary']:
        rs = [r for g in groups if g['method'] == reported['method'] for r in g['results']
              if r[option] == reported[option] and r['updates'] == reported['updates']]
        assert len(rs) == 9
        for name, (part, metric) in fields.items():
            close(reported[name], st.mean(r['assessments'][part][metric] for r in rs))
    individual = []
    for group in groups:
        final = [r for r in group['results'] if r['updates'] == 512]
        means = {mode: st.mean(r['assessments']['heldout']['mse_ratio_to_training_mean']
                              for r in final if r[option] == mode) for mode in variants}
        if stage == 'fit':
            reported = next(r for r in summary['per_initialization'] if r['method'] == group['method'] and r['seed'] == group['initial_seed'])
            close(means['normalized-output'], reported['mean_holdout_mse_ratio'])
            individual.append(means['normalized-output'] < 1)
        else:
            reported = next(r for r in summary['gates'] if r['method'] == group['method'] and r['seed'] == group['initial_seed'])
            close(means[False], reported['no_clock']); close(means[True], reported['clock'])
            passed = means[True] < min(1, means[False])
            assert passed == reported['beats_both_constant_and_no_clock']
            individual.append(passed)
    key = 'normalized_beats_constant_each_initialization' if stage == 'fit' else 'passed'
    assert summary[key] == all(individual)
    return protocol, summary, groups


def main():
    source = ROOT/'runs/critic-check-results.zip'
    destination = ROOT/'results/critic-checks-cluster'
    destination.mkdir(parents=True, exist_ok=True)
    detailed = {}; pairs = []; junctions = []; failures = []
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == 38
        expected_files = set()
        for behavior in BEHAVIORS:
            for stage in ('fit', 'horizon'):
                prefix = f'runs/critic-{stage}-{behavior}/'
                files = ['protocol.json', 'summary.json', 'REPORT.md'] + [f'fit-{m}-{s}.json' for m in METHODS for s in SEEDS]
                if stage == 'fit': files += ['toy.json']
                expected_files.update(prefix+f for f in files)
                protocol, summary, groups = validate(z, behavior, stage)
                detailed[f'{stage}/{behavior}'] = {'protocol': protocol, 'summary': summary, 'groups': groups}
                if stage != 'horizon': continue
                for method in METHODS:
                    means = {clock: next(r for r in summary['summary'] if r['method'] == method and r['clock'] == clock and r['updates'] == 512)
                             for clock in (False, True)}
                    cases = []; improvement_cases = []
                    for group in groups:
                        if group['method'] != method: continue
                        lookup = {(r['junction'], r['clock']): r for r in group['results'] if r['updates'] == 512}
                        for tls in ('C0', 'C1', 'C2'):
                            for seed in protocol['check_demand_seeds']:
                                m0 = lookup[tls, False]['assessments'][f'seed-{seed}']
                                m1 = lookup[tls, True]['assessments'][f'seed-{seed}']
                                cases.append(m1['mse_ratio_to_training_mean'] < 1)
                                improvement_cases.append(m1['rmse'] < m0['rmse'])
                                if m1['mse_ratio_to_training_mean'] >= 1:
                                    failures.append({'behavior': behavior, 'method': method, 'junction': tls,
                                                     'critic_seed': group['initial_seed'], 'check_seed': seed,
                                                     'mse_ratio': m1['mse_ratio_to_training_mean']})
                    gates = [g for g in summary['gates'] if g['method'] == method]
                    pairs.append({'behavior': behavior, 'method': method,
                        'without_clock': means[False]['mean_holdout_mse_ratio'], 'with_clock': means[True]['mean_holdout_mse_ratio'],
                        'relative_reduction': 1-means[True]['mean_holdout_mse_ratio']/means[False]['mean_holdout_mse_ratio'],
                        'initialization_gates_passed': sum(g['beats_both_constant_and_no_clock'] for g in gates),
                        'individual_cases_below_constant': sum(cases), 'individual_cases_clock_improved': sum(improvement_cases),
                        'individual_cases': len(cases), 'training_ratio_with_clock': means[True]['mean_train_mse_ratio'],
                        'bias_with_clock': means[True]['mean_holdout_bias']})
                    for tls in ('C0', 'C1', 'C2'):
                        rs = [r for g in groups if g['method'] == method for r in g['results']
                              if r['updates'] == 512 and r['clock'] and r['junction'] == tls]
                        junctions.append({'behavior': behavior, 'method': method, 'junction': tls,
                            'mean_mse_ratio': st.mean(r['assessments']['heldout']['mse_ratio_to_training_mean'] for r in rs),
                            'training_target_mean': rs[0]['assessments']['training']['target_mean'],
                            'check_target_mean': rs[0]['assessments']['heldout']['target_mean'],
                            'mean_bias': st.mean(r['assessments']['heldout']['bias'] for r in rs)})
        assert set(z.namelist()) == expected_files
    audit = {'archive_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'complete_diagnostics': 4, 'fit_job_files': 24, 'critic_fits': 144,
             'metric_rows': 432, 'source_and_protocol_hashes_match': True,
             'summaries_and_gates_recomputed': True,
             'raw_trajectories_or_model_weights_present': False,
             'limitation': 'Cannot recompute prediction errors from raw samples or replay learned actors from this summary-only archive.'}
    (destination/'summary.json').write_text(json.dumps({'audit': audit, 'horizon_comparisons': pairs,
        'junction_results': junctions, 'individual_failures': failures,
        'diagnostic_summaries': {k: v['summary'] for k,v in detailed.items()}}, indent=2)+'\n')
    lines = ['# Cluster critic-check review', '',
        'All four diagnostic reports and 24 fit-job files are present, covering 144 fitted critics. ZIP integrity, exact member coverage, source/protocol hashes, seed splits, metric identities and all reported averages/gates passed checks. The protocol describes 16 short simulations. Raw trajectory tensors and saved actor weights are absent, so this review verifies the supplied statistics and their aggregation rather than recomputing predictions from raw samples.', '',
        '## Decision', '',
        'Remaining-time information helps on both learned behavior policies. The local-input critics pass the predeclared average gate in all three initialisations for each policy. The network-wide critics improve relative to their no-clock controls but fail the constant-predictor comparison in every initialisation. Therefore neither complete horizon diagnostic passes its all-methods gate, and neither scaling-only diagnostic passes. The earlier local success on actuated-generated trajectories did not fully carry over.', '',
        '## Matched remaining-time comparison at 512 updates', '',
        'Both arms already use normalised value outputs. Error is MSE divided by the error of predicting the training-set mean on the same check data. Below 1 is better than that constant reference; these are reward-prediction errors, not traffic delays.', '',
        '| Frozen behavior | Critic inputs | No clock | Clock | Error reduction vs no clock | Initialisation gates | Individual cases beating constant |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        lines.append(f"| {r['behavior']} | {'Local' if r['method']=='ippo' else 'Network-wide'} | {r['without_clock']:.3f} | {r['with_clock']:.3f} | {100*r['relative_reduction']:.1f}% | {r['initialization_gates_passed']}/3 | {r['individual_cases_below_constant']}/18 |")
    lines += ['',
        'The 18 cases per row are three critic initialisations x three junctions x two check demand seeds. They share trajectories and are not 18 independent statistical trials. These critics were fitted afresh; both behavior actors are MAPPO replica 0. The table is not evidence that trained IPPO traffic control beats MAPPO.', '',
        '## Scaling-only experiment at 512 updates', '',
        '| Behavior | Inputs | Raw-output error ratio | Normalised-output error ratio |',
        '|---|---|---:|---:|']
    for behavior in BEHAVIORS:
        rows = detailed[f'fit/{behavior}']['summary']['summary']
        for method in METHODS:
            metrics = {r['mode']: r['mean_holdout_mse_ratio'] for r in rows if r['method']==method and r['updates']==512}
            lines.append(f"| {behavior} | {method} | {metrics['raw-scaled-loss']:.3f} | {metrics['normalized-output']:.3f} |")
    lines += ['',
        'Normalisation removes much of the raw-output bias, but it is not sufficient to learn a reliable state-dependent value function on these four training trajectories. The scaling and horizon experiments use different check seed pairs (49201/49202 and 49401/49402); do not combine their numbers into a single raw-versus-clock effect estimate.', '',
        '## Statistics hidden by the averages', '',
        '| Behavior | Inputs | Junction | Clock error ratio | Training target mean | Check target mean | Prediction bias |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in junctions:
        lines.append(f"| {r['behavior']} | {r['method']} | {r['junction']} | {r['mean_mse_ratio']:.3f} | {r['training_target_mean']:.2f} | {r['check_target_mean']:.2f} | {r['mean_bias']:.2f} |")
    lines += ['',
        'Local critics still have failures: C1 under plain behavior has a pooled error ratio of 1.066; C2 under priority behavior is 1.011. For network-wide critics, plain C1 is 1.455 and priority C0 is 1.808. All individual failures are retained in summary.json.', '',
        'The network-wide clock critic fits its training data closely (ratios about 0.050 and 0.049) while its check ratios are 1.123 and 1.236. This is a strong training/generalisation gap. More updates on the same four trajectories do not resolve it. The check trajectories also have different return distributions: at C1 the mean target changes from -96.15 to -140.29 for plain behavior, and -102.16 to -148.95 for priority behavior. Larger input/capacity, limited independent trajectories, stochastic outcomes and distribution differences are plausible contributors; these experiments do not identify one unique cause.', '',
        '## What follows', '',
        'The scale and remaining-time mechanisms justify a controlled repair, but the combined implementation is not yet validated for both critic types. Do not launch another unchanged 512-episode traffic study or claim the traffic target has been met.', '',
        'The next narrow prediction test should address generalisation: increase independent training trajectories while keeping the architecture, target definition and update budget matched, and evaluate on a newly reserved development check set. A simple constant or regularised predictor should remain a comparator. If simplifying the central critic is tested, separate that from the data-volume change. Improving the number of nearly adjacent states is not equivalent to adding independent traffic seeds.', '',
        'For an eventual on-policy PPO pilot, separate raw/no-clock, normalised/no-clock and normalised/clock variants under equal actor settings, rewards, demand and safety rules. These offline fits use only four trajectories, whereas original PPO generated new traffic episodes throughout training; do not infer that the original 512-episode study trained on only four trajectories. Improved offline values do not guarantee an actor improvement. All original vehicle/pedestrian results remain unchanged.', '',
        'Keep the deadline in view: preserve these results as diagnosis and avoid open-ended method searching. A final paper should distinguish demonstrated evaluation-mode effects, value-prediction findings, and actual traffic-control outcomes. No final evaluation seeds have been consumed by these checks.', '']
    (destination/'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'audit': audit, 'comparisons': pairs, 'review': str(destination/'REVIEW.md')}, indent=2))


if __name__ == '__main__':
    main()
