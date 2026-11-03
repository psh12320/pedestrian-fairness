"""Paired development comparisons, including crossing tails and service counts."""
import json
import math
from bisect import bisect_right
from collections import Counter
import random
import statistics as st
from ..fairness.analysis import quantile,selection_score
from ..fairness.pipeline import atomic_json
from .learning import VARIANTS
from .pilot import labels,cached_json,digest,BASELINES,METHODS


def bootstrap_estimator(models,references):
    """Exact nearest-rank tails by weighted CDF search, avoiding repeated sorting."""
    crossings=sorted(models[0][0]['crossings'])
    def prepare(data):
        out=[]
        for row in data:
            local={}
            for c in crossings:
                waits=[sorted(r['crossings'][c]['waits']) for r in row]
                levels=sorted(set(w for ws in waits for w in ws))
                local[c]=(levels,[[bisect_right(ws,v) for v in levels] for ws in waits],[len(ws) for ws in waits])
            out.append(local)
        return out
    left,right=prepare(models),prepare(references)
    def tail(prepared,weights):
        levels,cdf,lengths=prepared;n=sum(lengths[s]*w for s,w in weights)
        if not n:return None
        rank=math.ceil(.95*n);lo,hi=0,len(levels)-1
        while lo<hi:
            mid=(lo+hi)//2
            if sum(cdf[s][mid]*w for s,w in weights)>=rank:hi=mid
            else:lo=mid+1
        return levels[lo]
    def evaluate(replicas,seeds):
        selected=Counter(seeds).items();counts=Counter(replicas);deltas={}
        vehicle=[]
        for r in replicas:
            for s in seeds:
                a=models[r][s]['vehicles']['completed_mean_time_loss_seconds']
                b=references[r][s]['vehicles']['completed_mean_time_loss_seconds']
                if a is None or b is None:return None
                vehicle.append(a-b)
        for c in crossings:
            difference=0.
            for r,count in counts.items():
                a,b=tail(left[r][c],selected),tail(right[r][c],selected)
                if a is None or b is None:
                    difference=None;break
                difference+=(a-b)*count
            deltas[c]=difference/len(replicas) if difference is not None else None
        return {'vehicle_delta_seconds':st.mean(vehicle),'crossing_p95_delta_seconds':deltas,
                'worst_crossing_delta_seconds':max(deltas.values()) if all(d is not None for d in deltas.values()) else None,
                'crossings_no_worse':sum(d is not None and d<=0 for d in deltas.values()),
                'missing_crossings':[c for c,d in deltas.items() if d is None]}
    return evaluate


def estimates(models,references,replicas,seeds):
    vehicle=[];deltas={}
    for r in replicas:
        for s in seeds:
            a=models[r][s]['vehicles']['completed_mean_time_loss_seconds']
            b=references[r][s]['vehicles']['completed_mean_time_loss_seconds']
            if a is None or b is None:return None
            vehicle.append(a-b)
    for crossing in models[0][0]['crossings']:
        left=[quantile([w for s in seeds for w in models[r][s]['crossings'][crossing]['waits']]) for r in replicas]
        right=[quantile([w for s in seeds for w in references[r][s]['crossings'][crossing]['waits']]) for r in replicas]
        deltas[crossing]=st.mean(left)-st.mean(right) if all(v is not None for v in left+right) else None
    return {'vehicle_delta_seconds':st.mean(vehicle),'crossing_p95_delta_seconds':deltas,
            'worst_crossing_delta_seconds':max(deltas.values()) if all(d is not None for d in deltas.values()) else None,
            'crossings_no_worse':sum(d is not None and d<=0 for d in deltas.values()),
            'missing_crossings':[c for c,d in deltas.items() if d is None]}


def comparison(models,references,draws=1000):
    nr,ns=len(models),len(models[0]);replicas=list(range(nr));seeds=list(range(ns))
    if len(references)!=nr or any(len(row)!=ns for row in models+references):raise ValueError('Unpaired results')
    for r in replicas:
        for s in seeds:
            if models[r][s]['demand_sha256']!=references[r][s]['demand_sha256']:raise ValueError('Unmatched demand')
    point=estimates(models,references,replicas,seeds)
    cells=[(models[r][s],references[r][s]) for r in replicas for s in seeds]
    failed=sum(selection_score(a,b)[0]>0 for a,b in cells)
    crossing_order=sorted(models[0][0]['crossings'])
    minimum=min(sum(data[r][s]['crossings'][c]['entries'] for s in seeds)
                for data in (models,references) for r in replicas for c in crossing_order)
    out={'point':point,'service_failure_pairs':failed,'pairs':nr*ns,'minimum_pooled_crossing_entries':minimum,
         'sparse':minimum<100,'descriptive_pass':bool(point and point['vehicle_delta_seconds']<0 and
            point['worst_crossing_delta_seconds'] is not None and point['worst_crossing_delta_seconds']<=0 and failed==0),
         'exploratory_intervals':None,'bootstrap_valid_draws':0,'bootstrap_requested_draws':draws}
    # Smoke tests intentionally do not manufacture uncertainty from one replica/seed.
    if point and nr>=3 and ns>=2 and draws:
        rng=random.Random(424672);samples=[];estimate=bootstrap_estimator(models,references)
        for _ in range(draws):
            value=estimate(rng.choices(replicas,k=nr),rng.choices(seeds,k=ns))
            if value is not None:samples.append(value)
        out['bootstrap_valid_draws']=len(samples)
        if len(samples)==draws:
            def interval(values):return [quantile(values,.025),quantile(values,.975)]
            out['exploratory_intervals']={key:(interval([x[key] for x in samples]) if all(x[key] is not None for x in samples) else None)
                for key in ('vehicle_delta_seconds','worst_crossing_delta_seconds')}
            out['bootstrap_valid_crossing_draws']=sum(x['worst_crossing_delta_seconds'] is not None for x in samples)
    out['service_component_mean_deltas']={field:st.mean(a['vehicles'][field]-b['vehicles'][field] for a,b in cells)
        for field in ('completed_count','unfinished_count','uninserted_count')}
    return out


def absolute_metrics(models):
    rows=[a for replica in models for a in replica]
    out={'vehicle_'+key:st.mean(a['vehicles'][key] for a in rows) for key in
         ('scheduled_count','completed_count','unfinished_count','uninserted_count')}
    losses=[a['vehicles']['completed_mean_time_loss_seconds'] for a in rows]
    out['vehicle_mean_time_loss']=st.mean(losses) if all(v is not None for v in losses) else None
    out['pedestrian_unresolved']=st.mean(sum(v['unresolved_after_followup'] for v in a['crossings'].values()) for a in rows)
    out['pedestrian_pending_cutoff']=st.mean(sum(v['pending_at_cutoff'] for v in a['crossings'].values()) for a in rows)
    out['crossings']={}
    for c in rows[0]['crossings']:
        waits=[[w for a in replica for w in a['crossings'][c]['waits']] for replica in models]
        tails=[quantile(w) for w in waits]
        out['crossings'][c]={'mean_wait':st.mean(st.mean(w) for w in waits) if all(waits) else None,
            'mean_replica_pooled_p95':st.mean(tails) if all(v is not None for v in tails) else None,
            'entries_per_replica':[len(w) for w in waits],
            'pending_cutoff_mean_per_run':st.mean(a['crossings'][c]['pending_at_cutoff'] for a in rows),
            'unresolved_mean_per_run':st.mean(a['crossings'][c]['unresolved_after_followup'] for a in rows)}
    return out


def validate_result(result):
    v=result['vehicles']
    if v['scheduled_count']!=v['completed_count']+v['unfinished_count']+v['uninserted_count']:
        raise ValueError('Vehicle accounting failed')
    if len(result['crossings'])!=12:raise ValueError('Expected 12 crossings')
    for crossing in result['crossings'].values():
        waits=crossing['waits']
        if len(waits)!=crossing['entries'] or quantile(waits)!=crossing['p95'] or any(w<0 for w in waits):
            raise ValueError('Invalid crossing measurement')


def report(root,protocol):
    fp=protocol['fingerprint'];seeds=protocol['evaluation_seeds'];nr=protocol['replicas']
    completed={};demand_by_replica={}
    for method,variant,r,label in labels(protocol):
        folder=root/'models'/label;done=cached_json(folder/'complete.json',fp)
        if not done or done['episodes']!=protocol['episodes'] or done['checkpoint_sha256']!=digest(folder/'final.pt'):
            raise ValueError('Training incomplete: '+label)
        curve=cached_json(folder/'learning.json',fp)
        rollouts=[e for h in curve['history'] for e in h['rollouts']]
        expected=[(i+1,protocol['training_scenarios'][i%len(protocol['training_scenarios'])],
                   protocol['training_seed_base']+r*1000+i) for i in range(protocol['episodes'])]
        if [(e['episode'],e['scenario'],e['seed']) for e in rollouts]!=expected:raise ValueError('Training coverage mismatch')
        matched=[(e['demand_sha256'],e['network_sha256']) for e in rollouts]
        if r in demand_by_replica and matched!=demand_by_replica[r]:raise ValueError('Unmatched training demands')
        demand_by_replica[r]=matched;completed[label]=done
    scenarios={}
    for scenario in protocol['evaluation_scenarios']:
        base={c:[] for c in BASELINES};records={}
        for seed in seeds:
            data=cached_json(root/'baselines'/f'{scenario}-{seed}.json',fp)
            for c in BASELINES:
                value=next(v for v in data['records'] if v['controller']==c)['result']
                validate_result(value);base[c].append(value)
        for method,variant,r,label in labels(protocol):
            record=[]
            for seed in seeds:
                data=cached_json(root/'evaluation'/f'{label}-{scenario}-{seed}.json',fp)
                if data['checkpoint_sha256']!=completed[label]['checkpoint_sha256']:raise ValueError('Evaluation/model mismatch')
                validate_result(data['result']);record.append(data['result'])
            records[label]=record
        summary={'baselines':{},'learned':{},'paired_ablation':{}}
        for c in BASELINES:
            summary['baselines'][c]={'absolute':absolute_metrics([base[c]]),
                                    'versus_actuated':comparison([base[c]],[base['actuated']])}
        for method in METHODS:
            for variant in VARIANTS:
                key=f'{method}-{variant}';models=[records[f'{key}-r{r}'] for r in range(nr)]
                summary['learned'][key]={'absolute':absolute_metrics(models),
                    'comparisons':{c:comparison(models,[base[c]]*nr,draws=1000 if c=='actuated' else 0) for c in BASELINES}}
            for a,b in [('normalized-clock','original'),('normalized-clock-batch4','normalized-clock')]:
                summary['paired_ablation'][f'{method}:{a}-vs-{b}']=comparison(
                    [records[f'{method}-{a}-r{r}'] for r in range(nr)],
                    [records[f'{method}-{b}-r{r}'] for r in range(nr)])
        scenarios[scenario]=summary
    output={'fingerprint':fp,'protocol':protocol,'completed_models':len(completed),
            'all_training_demands_matched':True,'scenarios':scenarios,
            'interpretation':'Exploratory pilot; three initialisations and one seeded action stream per demand. No confirmatory claim.'}
    atomic_json(root/'summary.json',output)
    def fmt(x):return 'NA' if x is None else f'{x:.2f}'
    lines=['# On-policy PPO repair pilot','',
        f"Engineering smoke only: **{protocol['quick']}**. Models completed: {len(completed)}. Final episode checkpoints; no best-run selection. All training demand hashes match across methods/variants within each replica. Evaluation uses categorical sampling, three junctions, 12 crossings and unchanged safety.",'',
        'A = original PPO; B = normalised critic with remaining time; C = B with four independent rollouts per update. All use pedestrian weight 3. This follow-up supplements the original four-controller reward comparison.','',
        'Every confidence interval below is exploratory: resample whole training replicas and demand seeds together, preserving paired references. A single action-sampling stream is used per demand; action randomness is not separately replicated. Crossing p95 is pooled across demand seeds within each replica, then averaged across replicas. These are not averages of individual run p95s. Sparse crossing estimates (<100 entries in any replica or baseline) are inconclusive. No confirmatory test or final seeds are used.','']
    for scenario,s in scenarios.items():
        lines.extend([f'## {scenario}','',
            '| Controller | Vehicle loss s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved after followup |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for label,row in {**s['baselines'],**s['learned']}.items():
            a=row['absolute'];lines.append('| '+label+' | '+' | '.join(fmt(a[k]) for k in
                ('vehicle_mean_time_loss','vehicle_scheduled_count','vehicle_completed_count','vehicle_unfinished_count',
                 'vehicle_uninserted_count','pedestrian_pending_cutoff','pedestrian_unresolved'))+' |')
        lines.extend(['','| Learned controller vs actuated | Vehicle delta s (95% interval) | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |',
                      '|---|---|---:|---:|---:|---:|---|'])
        for label,row in s['learned'].items():
            c=row['comparisons']['actuated'];p=c['point'];ci=c['exploratory_intervals']
            interval='['+', '.join(fmt(v) for v in ci['vehicle_delta_seconds'])+']' if ci else 'not estimated'
            lines.append(f"| {label} | {fmt(p['vehicle_delta_seconds'] if p else None)} ({interval}) | {p['crossings_no_worse'] if p else 'NA'}/12 | {fmt(p['worst_crossing_delta_seconds'] if p else None)} | {c['service_failure_pairs']}/{c['pairs']} | {c['minimum_pooled_crossing_entries']} | {c['descriptive_pass']} {'(sparse)' if c['sparse'] else ''} |")
        lines.extend(['','| Paired training change | Vehicle delta s | Worst crossing p95 delta s | Crossings no worse |','|---|---:|---:|---:|'])
        for label,c in s['paired_ablation'].items():
            p=c['point'];lines.append(f"| {label} | {fmt(p['vehicle_delta_seconds'] if p else None)} | {fmt(p['worst_crossing_delta_seconds'] if p else None)} | {p['crossings_no_worse'] if p else 'NA'}/12 |")
        lines.extend(['','Full crossing means, p95s, sample counts, pending requests, paired intervals and comparisons against fixed/pressure/uniform controls are in summary.json. Per-run waits, slower-walker groups and diagnostics remain in evaluation JSON.',''])
    lines.extend(['## Limits and decision','',
        'These synthetic settings are hypotheses about traffic demand, not calibrated Singapore traffic. The original objective remains lower vehicle loss with no worse p95 at every crossing and no service deterioration. Better critic fits alone do not meet that objective. Descriptive passage is not statistical confirmation. Baseline service at a finite cutoff is strict and must be interpreted alongside followup counts.','',
        'A and B perform one update batch per episode; C performs one per four episodes. At 64 episodes, A/B make 64 batches and C 16; each uses four full-batch PPO passes. Equal simulated transitions do not imply equal update counts or equal policy trajectories. B combines output normalisation and clock inputs; it does not separate their individual causal contributions. C varies batch composition and policy-update frequency together.','',
        'Inspect repeated-seed traffic outcomes and action/critic diagnostics before extending training. Preserve negative results, and freeze any subsequent larger evaluation before using untouched test seeds.',''])
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report:',root/'REPORT.md',flush=True)
