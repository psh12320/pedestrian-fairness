"""Audit the received pilot archive and summarise existing results without SUMO."""
from collections import defaultdict
from pathlib import Path
import hashlib
import io
import json
import math
import re
import statistics as st
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.traffic_control.fairness.analysis import quantile, selection_score


def family(label):
    return re.sub(r'-r\d+$', '', label)


def mean(rows, function):
    values = [function(r) for r in rows]
    return st.mean(values) if values and all(v is not None for v in values) else None


def summarise_pilot(rows, scenario, controller, reference):
    selected = [r for r in rows if r['scenario'] == scenario and family(r['controller']) == controller]
    replicas = sorted({r.get('replica') for r in selected}, key=lambda x: -1 if x is None else x)
    crossings = sorted(selected[0]['result']['crossings'])
    baseline = [r for r in reference.values() if r['scenario'] == scenario]
    difference, counts = {}, {}
    for crossing in crossings:
        bw = [w for r in baseline for w in r['result']['crossings'][crossing]['waits']]
        pools = [[w for r in selected if r.get('replica') == replica
                  for w in r['result']['crossings'][crossing]['waits']] for replica in replicas]
        difference[crossing] = st.mean(quantile(p) for p in pools) - quantile(bw) if bw and all(pools) else None
        counts[crossing] = {'baseline': len(bw), 'replicas': [len(p) for p in pools]}
    deltas = [r['result']['vehicles']['completed_mean_time_loss_seconds'] -
              reference[scenario, r['seed']]['result']['vehicles']['completed_mean_time_loss_seconds'] for r in selected]
    scores = [selection_score(r['result'], reference[scenario, r['seed']]['result']) for r in selected]
    means = {}
    for key in ('scheduled_count', 'completed_count', 'unfinished_count', 'uninserted_count', 'completed_mean_time_loss_seconds'):
        means[key] = mean(selected, lambda r: r['result']['vehicles'][key])
    for key in ('pending_at_cutoff', 'unresolved_after_followup', 'unknown_cohort_entries', 'lost'):
        means[key] = mean(selected, lambda r: sum(c[key] for c in r['result']['crossings'].values()))
    means['person_unfinished_at_end'] = mean(selected, lambda r: r['result']['persons']['unfinished'])
    means['person_uninserted_at_end'] = mean(selected, lambda r: r['result']['persons']['uninserted'])
    means['pedestrian_mean_wait'] = st.mean(st.mean([w for r in selected if r.get('replica') == replica
        for c in r['result']['crossings'].values() for w in c['waits']]) for replica in replicas)
    complete = all(v is not None for v in difference.values())
    return {'scenario': scenario, 'controller': controller, 'runs': len(selected), 'training_replicas': len(replicas),
        'means': means, 'vehicle_difference': st.mean(deltas),
        'crossing_p95_differences': difference, 'crossing_counts': counts,
        'crossings_no_worse_point': sum(v is not None and round(v, 9) <= 0 for v in difference.values()),
        'worst_crossing_difference': max(difference.values()) if complete else None,
        'minimum_crossing_entries': min(n for c in counts.values() for n in [c['baseline'], *c['replicas']]),
        'service_failed_runs': sum(s[0] > 0 for s in scores),
        'all_crossing_descriptive_pass': bool(complete and st.mean(deltas) < 0 and
             max(difference.values()) <= 0 and all(s[0] == 0 for s in scores))}


def main():
    source = ROOT / 'runs/service-priority-results.zip'
    destination = ROOT / 'results/service-priority-cluster'
    destination.mkdir(parents=True, exist_ok=True)
    prefix = 'runs/service-priority-pilot/'
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None, 'Archive CRC failure'
        protocol = json.loads(archive.read(prefix + 'protocol.json'))
        assert protocol['quick'] is False and protocol['episodes'] == 32 and protocol['replicas'] == 3
        names = set(archive.namelist())
        expected_models = [f'{mode}-{method}-r{replica}' for mode in ('plain','priority')
                           for method in ('ippo','mappo') for replica in range(3)]
        selected, training = {}, {}
        rows = []
        for model in expected_models:
            base = prefix + model + '/'
            assert json.loads(archive.read(base+'complete.json'))['episodes'] == 32
            assert all(base+f'episode-{i}.json' in names for i in range(1,33))
            selected[model] = json.loads(archive.read(base+'selected.json'))
            training[model] = [json.loads(archive.read(base+f'episode-{i}.json')) for i in range(1,33)]
            outcomes = json.loads(archive.read(base+'evaluation.json'))
            assert {(r['scenario'],r['seed']) for r in outcomes} == {(s,t) for s in protocol['scenarios'] for t in protocol['seeds']}
            rows.extend(dict(r, replica=int(model.rsplit('-r',1)[1])) for r in outcomes)
        for name in sorted(names):
            if re.fullmatch(re.escape(prefix)+r'baseline-[^/]+\.json',name):
                group = json.loads(archive.read(name))
                assert len(group['records']) == 4
                rows.extend(dict(r, scenario=group['scenario'], seed=group['seed'], replica=None) for r in group['records'])
        assert len(rows) == 128
        assert not set(r['seed'] for r in rows) & set(protocol['reserved_test_seeds_not_used'])
        paired = json.loads(archive.read(prefix+'paired_diagnostics.json'))
        paired_lookup = {(r['scenario'],r['seed'],r['controller']):r for r in paired}
        reference = {(r['scenario'],r['seed']):r for r in rows if r['controller']=='actuated'}
        for r in rows:
            b = reference[r['scenario'],r['seed']]
            assert r['result']['demand_sha256'] == b['result']['demand_sha256']
            recorded = paired_lookup[r['scenario'],r['seed'],r['controller']]
            score = selection_score(r['result'],b['result'])
            assert score[0] == recorded['service_gate_score']
            assert math.isclose(score[2]-b['result']['vehicles']['completed_mean_time_loss_seconds'],recorded['vehicle_difference'],abs_tol=1e-9)
            v = r['result']['vehicles']
            assert v['scheduled_count'] == v['completed_count']+v['unfinished_count']+v['uninserted_count']
            for c in r['result']['crossings'].values():
                assert c['entries'] == len(c['waits']) and c['p95'] == quantile(c['waits'])
            for phases in r['diagnostics']['phase_seconds'].values():
                assert math.isclose(sum(phases.values()),2400,abs_tol=1e-6)
        source_matches = {p: (ROOT/p).exists() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
                          for p,h in protocol['source'].items()}
        assert all(source_matches.values()), 'Pilot source/configuration differs from reviewed local code'
        with zipfile.ZipFile(io.BytesIO(archive.read('runs/fairness-diagnosis.zip'))) as diagnosis:
            assert diagnosis.testzip() is None
            original = json.loads(diagnosis.read('diagnosis.json'))
            original_primary = json.loads(diagnosis.read('primary_inference.json'))
            original_protocol = json.loads(diagnosis.read('protocol.json'))
            complete_comparisons = json.loads(diagnosis.read('comparisons.json'))
            assert original['groups']==420 and len(original['records'])==13860 and len(complete_comparisons)==126
            assert original['fingerprint']==original_protocol['fingerprint']
            assert original_primary == json.loads((ROOT/'runs/cluster-final-review/primary_inference.json').read_text())

    order = ['actuated','pressure','priority-pressure','age-pressure','plain-ippo','priority-ippo','plain-mappo','priority-mappo']
    summaries = [summarise_pilot(rows,s,c,reference) for s in protocol['scenarios'] for c in order]
    original_groups=defaultdict(list)
    for r in original['records']:
        original_groups[r['scenario'],r['controller']].append(r)
    original_summary=[]
    for (scenario,controller),rs in sorted(original_groups.items()):
        metrics={k:mean(rs,lambda r:r['vehicles'][k]) for k in ('scheduled_count','completed_count','unfinished_count','uninserted_count','completed_mean_time_loss_seconds')}
        metrics['person_unfinished_at_end']=mean(rs,lambda r:r['persons']['unfinished'])
        metrics['unresolved_crossing_requests']=mean(rs,lambda r:sum(c['unresolved_after_followup'] for c in r['crossings'].values()))
        metrics['unknown_entries']=sum(c['unknown_cohort_entries'] for r in rs for c in r['crossings'].values())
        metrics['lost']=sum(c['lost'] for r in rs for c in r['crossings'].values())
        original_summary.append({'scenario':scenario,'controller':controller,'runs':len(rs),'means':metrics})
    audit={'archive_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'models':12,'episodes_per_model':32,
        'training_episodes':384,'evaluation_records':128,'source_matches':source_matches,
        'matched_demand_verified':True,'counts_reconciled':True,'selected':selected,'original_groups':420,'original_records':13860}
    output={'audit':audit,'protocol':protocol,'pilot':summaries,'original':original_summary,
        'original_components':original['summary'],'original_primary':original_primary}
    (destination/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# Service-priority cluster pilot review','',
        'The received archive passed CRC, source-hash, demand-pairing, completion, vehicle-count and crossing-quantile checks. Twelve models completed 32 episodes each (384 training episodes); 128 evaluation records cover four scenarios and two demand seeds. This is development evidence, not a final held-out comparison. The reserved 58001-58020 seeds were not used.','',
        '## Main finding','',
        'The service-priority intervention did not meet the combined target. Some crossing-tail point estimates improved relative to plain PPO, but the learned controllers still under-served vehicle movements. Low completed-vehicle time loss must be read beside completed and uninserted counts. Do not extend training unchanged merely because the software ran successfully.','',
        '## Pilot outcomes','',
        'Vehicle counts refer to trips scheduled in [300,2100) seconds and their status at the 2100-second cutoff. They are not whole-run completion counts at 2400 seconds. Crossing waits use requests reaching a crossing in [300,2100), including entries during follow-up to 2400. Unresolved requests are still waiting after that follow-up. Fractional counts below are averages across runs.','',
        'Each crossing p95 pools the two demand seeds within each training replica, then averages replica p95s; the reference p95 pools the two actuated runs. Counts of no-worse crossings are descriptive, with only machine-roundoff (1e-9 s) ignored. Several crossing samples are sparse. No confidence intervals or statistical superiority claims are made from this pilot.','']
    for scenario in protocol['scenarios']:
        lines += [f'### {scenario}','',
            '| Controller | Mean vehicle loss s | Delta vs actuated s | Completed | Unfinished | Uninserted | Crossing p95 no worse / 12 | Largest p95 delta s | Failed service runs |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in [x for x in summaries if x['scenario']==scenario]:
            m=r['means']
            lines.append(f"| {r['controller']} | {m['completed_mean_time_loss_seconds']:.2f} | {r['vehicle_difference']:+.2f} | {m['completed_count']:.1f} | {m['unfinished_count']:.1f} | {m['uninserted_count']:.1f} | {r['crossings_no_worse_point']}/12 | {r['worst_crossing_difference']:+.2f} | {r['service_failed_runs']}/{r['runs']} |")
        lines += ['',f"Minimum reported pooled crossing sample across these comparisons: {min(r['minimum_crossing_entries'] for r in summaries if r['scenario']==scenario)} entries.",'']
    lines += ['## A visible failure mechanism','',
        'In plain-mappo-r0, medium demand, seed 47001, C0 received only 12 seconds of EW green across the entire 2400-second simulation, versus 792 seconds NS green. The corresponding actuated C0 received 456 seconds EW and 396 seconds NS. These are actual recorded phase durations, not inferred from a mean performance metric.','',
        'The pilot used pedestrian reward weight 3 in every learned arm. Its medium-demand pedestrian trips all completed, but vehicle service suffered. This is different from the old vehicle-only models starving pedestrians. Protecting only pedestrians does not guarantee adequate service for every vehicle movement.','',
        'Priority-IPPO also has a short-spacing warning: one replica/seed has unresolved pedestrian requests. The exact phase/occupancy sequence is not present in this summary archive, so it is not yet possible to attribute that case to a particular blockage mechanism.','',
        'The training logs show high policy entropy; for example priority-ippo-r0 C0 at episode 32 is 0.688, close to log(2)=0.693. That average applies to sampled training states, not all evaluation states. Evaluation always takes argmax. A slight persistent probability preference can produce repetitive decisions under argmax. This is a testable explanation, not a demonstrated cause or a reason to silently change evaluation after inspecting test results.','',
        'The priority policy receives roughly 20 free actor decisions per junction in a 600-second training episode; forced transitions and required pedestrian service do not train an actor choice. Thirty-two episodes are a pilot, not evidence of convergence.','',
        '## What the original-study diagnosis now establishes','',
        'The nested archive recovered 420 original groups / 13,860 records, the full comparisons JSON and complete vehicle/service summaries. It does not retain individual wait arrays for every original crossing; the saved primary pooled inference is available unchanged.','',
        'At original primary medium demand, vehicle-only IPPO/MAPPO each leave an average 698.65 person trips unfinished at simulation end, versus zero for actuated. The original pedestrian-aware/constrained models have no unfinished person trips or unresolved measured crossing requests in that scenario. Their service flags therefore must not all be described as permanent pedestrian starvation; crossing-cutoff differences and some vehicle differences require separate reporting.','',
        '### Original primary absolute results (20 demand seeds; five replicas for learned methods)','',
        '| Controller | Mean vehicle loss s | Scheduled | Completed | Unfinished | Uninserted | Person trips unfinished at 2400 s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in original_summary:
        if r['scenario']=='vmedium-pmedium':
            m=r['means'];lines.append(f"| {r['controller']} | {m['completed_mean_time_loss_seconds']:.2f} | {m['scheduled_count']:.1f} | {m['completed_count']:.1f} | {m['unfinished_count']:.1f} | {m['uninserted_count']:.2f} | {m['person_unfinished_at_end']:.2f} |")
    lines += ['', '## Recommended next step','',
        'Run a no-training diagnostic of the same saved pilot checkpoints: argmax versus seeded probability sampling, alongside uniform legal-action controls with and without the same priority rule. Use the existing development demand seeds; do not touch final test seeds. Record choice probabilities, movement service gaps, completion counts and critic diagnostics. This distinguishes evaluation collapse from a useful learned policy and from insufficient movement-service protection.','',
        'If sampling restores service but performs no better than uniform choices, the actor has not demonstrated useful learning. If movement neglect remains, introduce a demand-aware maximum-red/service rule covering vehicle movements as well as pedestrians, and compare a conventional version against a learned version. Validate the rule before another matched training pilot. Increasing pedestrian admission duration is a separate sensitivity experiment; it will not directly fix a vehicle direction that is almost never selected.','',
        'Do not launch another 512-episode study or consume the reserved final seeds yet. Preserve this negative pilot as evidence about the intervention.','',
        '## All original absolute scenario summaries','',
        'These descriptive means recover the previously missing fixed-time and actuated values. They are not confidence intervals. Every scenario/controller combination is also in summary.json.','']
    for scenario in original_protocol['scenarios']:
        lines += [f'### Original {scenario}','', '| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |', '|---|---:|---:|---:|---:|---:|']
        for r in original_summary:
            if r['scenario']==scenario:
                m=r['means']; value=m['completed_mean_time_loss_seconds']
                lines.append(f"| {r['controller']} | {value:.2f} | {m['completed_count']:.1f} | {m['unfinished_count']:.1f} | {m['uninserted_count']:.2f} | {m['person_unfinished_at_end']:.1f} |")
        lines.append('')
    (destination/'REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'audit':{k:v for k,v in audit.items() if k not in ('source_matches','selected')},
        'report':str(destination/'REVIEW.md')},indent=2))


if __name__=='__main__':
    main()
