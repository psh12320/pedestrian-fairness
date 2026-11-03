"""Complete, paired reports; sparse/negative outcomes remain visible."""
from ..fairness.pipeline import atomic_json
from ..ppo_repair.pilot import cached_json, digest
from ..ppo_repair.reporting import comparison, absolute_metrics, validate_result
from .pilot import labels, BASELINES


def report(root, protocol):
    fp = protocol['fingerprint']; nr = protocol['replicas']; seeds = protocol['evaluation_seeds']
    complete = {}; demands = {}
    def required(path):
        data = cached_json(path, fp)
        if data is None: raise ValueError('Incomplete study: ' + str(path))
        return data
    demos = required(root / 'teacher/demonstrations.json')
    if len(demos['records']) != protocol['demonstration_episodes']: raise ValueError('Incomplete demonstrations')
    if digest(root / 'teacher/demonstrations.pt') != demos['data_sha256']: raise ValueError('Changed demonstrations')
    fidelity = {}
    for r in range(nr):
        data = required(root / 'teacher' / f'final-r{r}.json')
        if digest(root / 'teacher' / f'final-r{r}.pt') != data['checkpoint_sha256']: raise ValueError('Changed teacher')
        complete[f'imitation-r{r}'] = data['checkpoint_sha256']; fidelity[str(r)] = data
    for method, variant, r, label in labels(protocol):
        directory = root / 'models' / label; done = required(directory / 'complete.json')
        sha = digest(directory / 'final.pt')
        if done['episodes'] != protocol['episodes'] or done['checkpoint_sha256'] != sha: raise ValueError('Incomplete model')
        history = required(directory / 'learning.json')
        if len(history['history']) != protocol['episodes'] or done['update_batches'] != protocol['episodes']:
            raise ValueError('Unequal episode/update budget')
        expected = [(i+1, protocol['training_scenarios'][i % len(protocol['training_scenarios'])],
                     protocol['training_seed_base'] + r * 1000 + i) for i in range(protocol['episodes'])]
        if [(h['episodes'], h['scenario'], h['seed']) for h in history['history']] != expected:
            raise ValueError('Training seed coverage failed')
        expected_validation = {(e, s) for e in protocol['validation_episodes'] for s in protocol['validation_seeds']}
        if len(history['validations']) != len(expected_validation) or {(v['episodes'], v['seed']) for v in history['validations']} != expected_validation:
            raise ValueError('Missing validation curve')
        for h in history['history']:
            for stats in h['update'].values():
                expected_count = h['episodes'] * (179 if protocol['quick'] else 599)
                if stats['normalizer_count'] != expected_count: raise ValueError('Incorrect normalizer sample count')
        hashes = [(h['demand_sha256'], h['network_sha256']) for h in history['history']]
        if r in demands and demands[r] != hashes: raise ValueError('Unmatched training demand')
        demands[r] = hashes; complete[label] = sha
    results = {}
    groups = [f'{m}-{v}' for m in protocol['methods'] for v in protocol['variants']] + ['imitation']
    for scenario in protocol['evaluation_scenarios']:
        bases = {c: [] for c in BASELINES}; networks = {}
        for seed in seeds:
            data = required(root / 'baselines' / f'{scenario}-{seed}.json')
            if sorted(r['controller'] for r in data['records']) != sorted(BASELINES): raise ValueError('Baseline coverage')
            if len({r['network_sha256'] for r in data['records']}) != 1: raise ValueError('Baseline network mismatch')
            networks[seed] = data['records'][0]['network_sha256']
            for row in data['records']:
                validate_result(row['result']); bases[row['controller']].append(row['result'])
        rows = {}
        for group in groups:
            rows[group] = []
            for r in range(nr):
                local = []
                for seed in seeds:
                    label = f'{group}-r{r}'
                    data = required(root / 'evaluation' / f'{label}-{scenario}-{seed}.json')
                    if data['checkpoint_sha256'] != complete[label] or data['network_sha256'] != networks[seed]:
                        raise ValueError('Evaluation identity mismatch')
                    validate_result(data['result']); local.append(data['result'])
                rows[group].append(local)
        result = {'baselines': {}, 'learned': {}, 'paired_ablation': {}}
        for c, data in bases.items():
            result['baselines'][c] = {'absolute': absolute_metrics([data]),
                'versus_actuated': comparison([data], [bases['actuated']], draws=0)}
        for group, data in rows.items():
            result['learned'][group] = {'absolute': absolute_metrics(data),
                'comparisons': {c: comparison(data, [bases[c]] * nr, draws=1000 if c == 'actuated' else 0) for c in BASELINES}}
        for method in protocol['methods']:
            for a, b in [('age', 'control'), ('guided', 'age'), ('rnd', 'age')]:
                result['paired_ablation'][f'{method}:{a}-vs-{b}'] = comparison(rows[f'{method}-{a}'], rows[f'{method}-{b}'])
            result['paired_ablation'][f'{method}:guided-vs-imitation'] = comparison(rows[f'{method}-guided'], rows['imitation'])
        results[scenario] = result
    out = {'fingerprint': fp, 'protocol': protocol, 'completed_models': len(labels(protocol)),
           'imitation_models': nr, 'teacher_fidelity': fidelity, 'scenarios': results,
           'all_training_demands_matched': True, 'demonstration_cost': demos['records']}
    atomic_json(root / 'summary.json', out)
    def fmt(x): return 'NA' if x is None else f'{x:.2f}'
    lines = ['# Guided exploration development study', '',
        f"Engineering smoke: **{protocol['quick']}**. {len(labels(protocol))} trained models; {nr} imitation-only actors. Three junctions, 12 crossings; identical physical phases and clearance.", '',
        'Control = previous repaired PPO. Age = additional bounded quadratic oldest-wait cost at each crossing. Guided = age plus actuated actor pretraining and decaying online teacher cross entropy. RND = age plus local training-only novelty. These are independent adaptations, not reproductions of the cited papers.', '',
        'Imitation-only outcomes and validation fidelity are included even if poor. No replica is discarded and fidelity does not choose checkpoints. Guidance adds demonstration and actor fitting work; RND adds predictor updates. PPO simulation steps, 4 PPO passes per episode and final episode are matched.', '',
        'All intervals are exploratory paired bootstrap intervals across whole training replicas and demand seeds. One action stream per demand. Crossing waits are pooled across seeds within a replica before averaging replica p95. Sparse estimates (<100 entries) are inconclusive. A descriptive pass is not a statistical guarantee or a real-world benefit.', '']
    for scenario, result in results.items():
        lines += [f'## {scenario}', '',
            '| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for label, data in {**result['baselines'], **result['learned']}.items():
            a = data['absolute']
            lines.append('| ' + label + ' | ' + ' | '.join(fmt(a[k]) for k in ('vehicle_mean_time_loss',
                'vehicle_completed_count', 'vehicle_unfinished_count', 'vehicle_uninserted_count',
                'pedestrian_pending_cutoff', 'pedestrian_unresolved')) + ' |')
        lines += ['', '| Model vs actuated | Vehicle delta s | 95% exploratory interval | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |',
                  '|---|---:|---|---:|---:|---:|---:|---|']
        for label, data in result['learned'].items():
            c = data['comparisons']['actuated']; p = c['point']; ci = c['exploratory_intervals']
            interval = str(ci['vehicle_delta_seconds']) if ci else 'not estimated'
            lines.append(f"| {label} | {fmt(p['vehicle_delta_seconds']) if p else 'NA'} | {interval} | {p['crossings_no_worse'] if p else 'NA'}/12 | {fmt(p['worst_crossing_delta_seconds']) if p else 'NA'} | {c['service_failure_pairs']}/{c['pairs']} | {c['minimum_pooled_crossing_entries']} | {c['descriptive_pass']}; sparse={c['sparse']} |")
        lines += ['', 'Full crossing means, p95s, counts, service accounting and component ablations are in summary.json. Per-run waits, speed groups, phases and actor diagnostics remain in evaluation/*.json.', '']
    lines += ['## Limits', '', 'Synthetic demand, exclusive pedestrian phase, four development settings, three initialisations. Short spacing also increases demand. No final test seeds used. Traffic outcomes, not reward or teacher accuracy, determine success. QMIX/COMA have not been tested by this study.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
