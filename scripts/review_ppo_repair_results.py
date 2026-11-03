"""Audit downloaded PPO pilot statistics without extracting weights or retraining."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.traffic_control.ppo_repair.reporting import absolute_metrics,comparison,validate_result
from src.traffic_control.fairness.analysis import quantile

PREFIX='runs/ppo-repair-pilot/'


def same(actual,expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
        for key in actual:same(actual[key],expected[key])
    elif isinstance(actual,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):same(a,b)
    elif isinstance(actual,float):
        assert math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9),(actual,expected)
    else:assert actual==expected,(actual,expected)


def result_check(row,horizon=2400):
    result=row['result'];validate_result(result)
    v=result['vehicles'];p=result['persons']
    assert v['inserted_count']==v['completed_count']+v['unfinished_count']
    assert p['scheduled']==p['completed']+p['unfinished']+p['uninserted']
    assert p['seen']==p['completed']+p['unfinished']
    for crossing,c in result['crossings'].items():
        assert c['mean']==(st.mean(c['waits']) if c['waits'] else None)
        assert len(c['wait_lower'])==len(c['wait_upper'])==c['entries']
        assert all(lo<=w+1e-8 and w<=hi+1e-8 for lo,w,hi in zip(c['wait_lower'],c['waits'],c['wait_upper']))
        assert 0<=c['unresolved_after_followup']<=c['pending_at_cutoff']
    for d in row['diagnostics'].values():
        assert math.isclose(sum(d['phase_seconds'].values()),horizon,abs_tol=1e-6)
        assert sum(d['free_action_counts'].values())==d['free_decisions']
        if 'critic_probe' in d:
            c=d['critic_probe'];assert c['horizon_seconds']==600
            assert c['rmse']>=abs(c['bias'])-1e-5
            assert math.isclose(c['bias'],c['prediction_mean']-c['target_mean'],rel_tol=1e-4,abs_tol=1e-4)


def diagnostic_summary(rows):
    ds=[d for row in rows for d in row['diagnostics'].values()]
    return {'vehicle_loss':st.mean(r['result']['vehicles']['completed_mean_time_loss_seconds'] for r in rows),
            'critic_probe_rmse':st.mean(d['critic_probe']['rmse']**2 for d in ds)**.5,
            'critic_probe_mean_bias':st.mean(d['critic_probe']['bias'] for d in ds),
            'critic_probe_mean_ev':st.mean(d['critic_probe']['explained_variance'] for d in ds),
            'mean_action_entropy':st.mean(d['mean_entropy'] for d in ds),
            'free_action_fractions':{str(i):sum(d['free_action_counts'][str(i)] for d in ds)/sum(d['free_decisions'] for d in ds) for i in range(3)},
            'phase_fractions':{stage:st.mean(d['phase_seconds'].get(stage,0) for d in ds)/2400
                               for stage in ('EW','NS','PED','PED_CLEARANCE','ALL_RED')}}


def main():
    source=ROOT/'runs/ppo-repair-results.zip';destination=ROOT/'results/ppo-repair-cluster'
    checks=Counter();curves=[];diagnostics={};training_signatures={};completed={};baseline={};evaluations={}
    with zipfile.ZipFile(source) as z:
        names=z.namelist();assert len(names)==len(set(names))==498
        assert z.testzip() is None
        manifest=json.loads(z.read('manifest.json'));assert set(manifest)==set(names)-{'manifest.json'}
        for name,sha in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
        p=json.loads(z.read(PREFIX+'protocol.json'));summary=json.loads(z.read(PREFIX+'summary.json'))
        same(summary['protocol'],p)
        unhashed=dict(p);unhashed.pop('fingerprint')
        assert hashlib.sha256(json.dumps(unhashed,sort_keys=True).encode()).hexdigest()==p['fingerprint']
        assert not p['quick'] and p['episodes']==64 and p['replicas']==3 and p['reward_weight']==3
        assert p['initial_seeds']==[72001,72002,72003] and p['evaluation_seeds']==list(range(830001,830007))
        assert p['validation_seeds']==[820001,820002] and p['validation_episodes']==[16,32,64]
        assert p['methods']==['ippo','mappo']
        assert p['variants']==['original','normalized-clock','normalized-clock-batch4']
        assert p['evaluation_scenarios']==['vmedium-pmedium','pedestrian-burst','short-spacing','slow-walkers']
        trainseeds={p['training_seed_base']+r*1000+i for r in range(3) for i in range(64)}
        splits=[trainseeds,set(p['evaluation_seeds']),set(p['validation_seeds']),set(range(58001,58021))]
        assert all(not a&b for i,a in enumerate(splits) for b in splits[i+1:])
        for name,sha in p['source'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
        def read(relative):
            data=json.loads(z.read(PREFIX+relative));assert data['fingerprint']==p['fingerprint'];return data
        expected={'manifest.json',PREFIX+'protocol.json',PREFIX+'summary.json',PREFIX+'REPORT.md'}
        settings={(s,d) for s in p['evaluation_scenarios'] for d in p['evaluation_seeds']}|{('vmedium-pmedium',d) for d in p['validation_seeds']}
        for scenario,seed in sorted(settings):
            filename=f'baselines/{scenario}-{seed}.json';expected.add(PREFIX+filename);group=read(filename)
            assert group['scenario']==scenario and group['seed']==seed
            assert [r['controller'] for r in group['records']]==p['baselines']
            assert len({r['result']['demand_sha256'] for r in group['records']})==1
            assert len({r['network_sha256'] for r in group['records']})==1
            for row in group['records']:result_check(row);checks['baseline_runs']+=1
            baseline[scenario,seed]={r['controller']:r for r in group['records']}
        for method in p['methods']:
            for variant in p['variants']:
                for replica in range(3):
                    label=f'{method}-{variant}-r{replica}';step=4 if variant.endswith('batch4') else 1
                    for name in ('learning.json','complete.json'):expected.add(PREFIX+f'models/{label}/{name}')
                    done=read(f'models/{label}/complete.json');learning=read(f'models/{label}/learning.json')
                    assert done['label']==learning['label']==label and done['episodes']==64 and done['update_batches']==64//step
                    assert [h['episodes'] for h in learning['history']]==list(range(step,65,step))
                    rollouts=[r for h in learning['history'] for r in h['rollouts']]
                    expected_train=[(i+1,p['training_scenarios'][i%4],p['training_seed_base']+replica*1000+i) for i in range(64)]
                    assert [(r['episode'],r['scenario'],r['seed']) for r in rollouts]==expected_train
                    signature=[(r['demand_sha256'],r['network_sha256']) for r in rollouts]
                    if replica in training_signatures:assert signature==training_signatures[replica]
                    training_signatures[replica]=signature
                    for h in learning['history']:
                        assert len(h['rollouts'])==step
                        for stats in h['update'].values():
                            assert stats['states']==step*599 and stats['episodes_in_update']==step
                            assert stats['normalizer_count']==(0 if variant=='original' else h['episodes']*599)
                            assert stats['normalizer_scale']>=1
                    validations=learning['validations']
                    assert {(r['episodes'],r['seed']) for r in validations}=={(e,d) for e in (16,32,64) for d in p['validation_seeds']}
                    assert len(validations)==6
                    for row in validations:
                        result_check(row)
                        assert row['result']['demand_sha256']==baseline['vmedium-pmedium',row['seed']]['actuated']['result']['demand_sha256']
                        checks['validation_runs']+=1
                    for e in (16,32,64):
                        curves.append({'label':label,'method':method,'variant':variant,'replica':replica,'episodes':e,
                                       **diagnostic_summary([v for v in validations if v['episodes']==e])})
                    completed[label]=done;checks['training_episodes']+=len(rollouts)
                    for scenario in p['evaluation_scenarios']:
                        for seed in p['evaluation_seeds']:
                            filename=f'evaluation/{label}-{scenario}-{seed}.json';expected.add(PREFIX+filename)
                            row=read(filename);assert row['label']==label and row['seed']==seed and row['scenario']==scenario
                            assert row['checkpoint_sha256']==done['checkpoint_sha256'];result_check(row)
                            ref=baseline[scenario,seed]['actuated']
                            assert row['result']['demand_sha256']==ref['result']['demand_sha256'] and row['network_sha256']==ref['network_sha256']
                            evaluations[label,scenario,seed]=row;checks['evaluation_runs']+=1
        assert set(names)==expected and len(completed)==18
        for scenario in p['evaluation_scenarios']:
            print('Recomputing paired summaries:',scenario,flush=True)
            s=summary['scenarios'][scenario];ds=p['evaluation_seeds']
            for controller in p['baselines']:
                values=[baseline[scenario,d][controller]['result'] for d in ds]
                references=[baseline[scenario,d]['actuated']['result'] for d in ds]
                same(absolute_metrics([values]),s['baselines'][controller]['absolute'])
                same(comparison([values],[references]),s['baselines'][controller]['versus_actuated'])
            def models(method,variant):
                return [[evaluations[f'{method}-{variant}-r{r}',scenario,d]['result'] for d in ds] for r in range(3)]
            for method in p['methods']:
                for variant in p['variants']:
                    key=f'{method}-{variant}';values=models(method,variant)
                    same(absolute_metrics(values),s['learned'][key]['absolute'])
                    for controller in p['baselines']:
                        refs=[[baseline[scenario,d][controller]['result'] for d in ds]]*3
                        same(comparison(values,refs,draws=1000 if controller=='actuated' else 0),s['learned'][key]['comparisons'][controller])
                    if scenario=='vmedium-pmedium':
                        rows=[evaluations[f'{key}-r{r}',scenario,d] for r in range(3) for d in ds]
                        diagnostics[key]=diagnostic_summary(rows)
                for a,b in [('normalized-clock','original'),('normalized-clock-batch4','normalized-clock')]:
                    same(comparison(models(method,a),models(method,b)),s['paired_ablation'][f'{method}:{a}-vs-{b}'])
        archive_report=z.read(PREFIX+'REPORT.md').decode()
    all_results=[r['result'] for r in evaluations.values()]+[r['result'] for b in baseline.values() for r in b.values()]
    checks['unresolved_crossing_requests']=sum(c['unresolved_after_followup'] for r in all_results for c in r['crossings'].values())
    checks['unknown_crossing_entries']=sum(c['unknown_cohort_entries'] for r in all_results for c in r['crossings'].values())
    checks['lost_crossing_requests']=sum(c['lost'] for r in all_results for c in r['crossings'].values())
    checks['unfinished_person_trips']=sum(r['persons']['unfinished'] for r in all_results)
    # Count separate cutoff conditions; never add overlapping components as people.
    service={}
    for method in p['methods']:
        for variant in p['variants']:
            key=f'{method}-{variant}';counts=Counter()
            for r in range(3):
                for seed in p['evaluation_seeds']:
                    a=evaluations[f'{key}-r{r}','vmedium-pmedium',seed]['result'];b=baseline['vmedium-pmedium',seed]['actuated']['result']
                    counts['fewer_completed_vehicles']+=a['vehicles']['completed_count']<b['vehicles']['completed_count']
                    counts['more_unfinished_vehicles']+=a['vehicles']['unfinished_count']>b['vehicles']['unfinished_count']
                    counts['more_uninserted_vehicles']+=a['vehicles']['uninserted_count']>b['vehicles']['uninserted_count']
                    counts['any_crossing_more_pending_at_cutoff']+=any(a['crossings'][c]['pending_at_cutoff']>b['crossings'][c]['pending_at_cutoff'] for c in a['crossings'])
            service[key]=dict(counts)
    audit={'archive_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'archive_members':498,
           'completed_models':18,'counts':dict(checks),'all_source_protocol_manifest_hashes_match':True,
           'all_paired_summaries_and_intervals_recomputed':True,
           'limitation':'No model weights, raw TraCI traces or scheduler log in bundle. Traffic statistics are recomputed from per-run waits/counts; critic predictions and phase conflicts cannot be independently replayed here.'}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'summary.json').write_text(json.dumps({'audit':audit,'primary_diagnostics':diagnostics,
        'validation_curves':curves,'primary_service_components':service,'pilot_summary':summary},indent=2)+'\n')
    (destination/'CLUSTER_REPORT.md').write_text(archive_report,encoding='utf-8')
    lines=['# PPO repair pilot review','',
        'All 18 models reached 64 episodes. The archive contains all 1,152 training-episode records, 108 validation runs, 432 development evaluation runs and 104 baseline runs. File coverage, source/protocol/manifest hashes, model-hash references, matched demand/network hashes, training normaliser counts, accounting, crossing statistics, phase-duration sums, and every aggregate/paired interval passed audit. Raw actor weights and simulator traces are absent, so the critic probes and physical conflicts cannot be replayed independently from this ZIP.','',
        '## Decision','',
        'No learned configuration meets the descriptive joint target in any of the four evaluated settings. The normalisation/clock repair improves critic calibration, but does not produce a reliable joint vehicle/pedestrian gain. Four-rollout batching is worse for pedestrian tails under this fixed-episode budget. Do not promote a winner or launch another unchanged large study.','',
        'A = original PPO. B = repaired critic (normalisation + clock), one episode/update. C = B with four episodes/update. These are all pedestrian-weight-3 follow-ups, not the original vehicle-versus-pedestrian reward comparison.','',
        '## Primary medium-demand traffic results','',
        '| Controller | Vehicle loss s | Delta vs actuated, 95% exploratory interval | Crossings no worse | Worst crossing p95 increase s |',
        '|---|---:|---|---:|---:|']
    primary=summary['scenarios']['vmedium-pmedium']
    for key,row in primary['baselines'].items():
        c=row['versus_actuated'];point=c['point']
        lines.append(f"| {key} | {row['absolute']['vehicle_mean_time_loss']:.2f} | {point['vehicle_delta_seconds']:+.2f} | {point['crossings_no_worse']}/12 | {point['worst_crossing_delta_seconds']:+.2f} |")
    for key,row in primary['learned'].items():
        c=row['comparisons']['actuated'];point=c['point'];lo,hi=c['exploratory_intervals']['vehicle_delta_seconds']
        lines.append(f"| {key} | {row['absolute']['vehicle_mean_time_loss']:.2f} | {point['vehicle_delta_seconds']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {point['crossings_no_worse']}/12 | {point['worst_crossing_delta_seconds']:+.2f} |")
    lines+=['','Every learned variant has higher p95 at all 12 crossings in the medium-demand comparison. Minimum pooled crossing counts are 272-274 across the compared replicas/reference, above the prespecified 100-entry diagnostic threshold. This failure cannot be dismissed as the earlier five-entry crossing problem. Six demand seeds and three training seeds still give limited uncertainty estimates.','',
        'The 15-18/18 failed primary service checks include pending requests at the measurement cutoff. All medium-demand person trips complete by the simulation end, with no unresolved measured-cohort crossing requests after followup. Across all evaluated settings, there are 78 unfinished person-trip records (75 in slow-walker runs and 3 under short-spacing pressure control), despite zero unresolved measured-cohort crossing requests and zero unknown/lost measured events. A person finishing their entire trip is distinct from a request that reached a crossing in the measurement window; later requests or walking after crossing require raw traces to classify. Individual vehicle and crossing cutoff differences remain in summary.json.','',
        '## What changed in the critic','',
        '| Controller | 600-second probe RMSE | Mean bias | Mean explained variance | Free-decision entropy |',
        '|---|---:|---:|---:|---:|']
    for key,d in diagnostics.items():
        lines.append(f"| {key} | {d['critic_probe_rmse']:.2f} | {d['critic_probe_mean_bias']:.2f} | {d['critic_probe_mean_ev']:.3f} | {d['mean_action_entropy']:.3f} |")
    lines+=['','B removes much of the positive value bias: roughly +105 to +7 reward units for IPPO, and +104 to +5 for MAPPO. However, explained variance remains low (about 0.067 and 0.032 on medium evaluation), so much of the gain is calibration rather than strong state-dependent prediction. Each policy generates different states/targets: these probe errors are diagnostics, not a matched-target regression comparison. They are not waiting-time improvements.','',
        'C remains biased (+95 IPPO, +85 MAPPO) and has higher action entropy. It receives only 16 update batches versus 64 for A/B, with four PPO passes per batch. This is consistent with slower learning under fewer updates, not proof that collecting independent trajectories is harmful. The offline 32-run check used 512 critic fitting steps on fixed data; this online training design does not replicate that update budget.','',
        '## Actual evidence of partial improvement','',
        'On short spacing, IPPO B lowers vehicle time loss against IPPO A by 7.63 seconds (exploratory paired interval -13.43 to -3.45 seconds), and improves 8/12 crossing p95 point estimates against A. The remaining crossings and service components prevent a joint pass. In slow walking, IPPO B improves all 12 crossing point estimates against A, but its vehicle interval includes no improvement, and it still loses every crossing against actuated. These are secondary exploratory findings, not a replacement primary target.','',
        '## Why service accounting matters','',
        'On short spacing, actuated control completes about 788/2,431 scheduled measurement-cohort vehicles, with about 1,569 uninserted. Its completed-vehicle time loss of 46.86 seconds therefore describes a selected served subset. IPPO B completes about 1,474 vehicles with 860 uninserted, at 76.56 seconds completed-vehicle time loss. Fixed time completes about 2,317 with 28 uninserted, at 39.28 seconds, but delays pedestrians. It would be misleading to rank this setting by completed-vehicle delay alone.','',
        'Pressure-inspired control is a useful strong benchmark at medium demand: 40.16 seconds vehicle loss versus 50.78 actuated, but its worst crossing p95 is 14.0 seconds higher and none of the crossings meets non-inferiority. Uniform choices have slightly lower vehicle loss than actuated but much worse pedestrian tails (worst +63.0 seconds). Randomness is not the overall winner. All four A/B configurations improve every crossing p95 point estimate against uniform control, while increasing vehicle loss. For example IPPO B costs 2.86 seconds more vehicle loss than uniform, but improves every crossing p95 by at least 4.8 seconds. This is a learned trade-off, not overall dominance or passage against actuated. C improves only 10/12 crossing point estimates against uniform.','',
        '## Recommended next decision','',
        'Preserve this as a completed controlled negative result and move the report forward. Do not rerun the same 64/512-episode configuration or claim that the earlier diagnostic solved the controller. The next technical step should be a bounded decision-learning audit: examine action advantages, actor versus critic gradient magnitudes and the cost of switching/PED service before another training campaign. Shared gradient clipping is a possible mechanism to inspect, not a proven cause from the total-gradient logs alone.','',
        'If testing another training intervention, isolate its change and keep a matched control; use the strong pressure/actuated references and retain all crossings and service counts. A baseline-guided actor is a possible future hypothesis, not a demonstrated improvement. Preserve the original IPPO/MAPPO proposal question and untouched final seeds.','',
        'Results are from synthetic three-junction settings, not calibrated Singapore operations. The burst and medium scenarios share much traffic and are not independent replications; short spacing also increases demand, combining two stressors. The four scenarios, three initialisations and single action-sampling stream per demand limit generalisation.','']
    (destination/'REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(audit,indent=2));print(destination/'REVIEW.md')


if __name__=='__main__':main()
