"""Independently audit the fine-tuning archive without importing its code/weights."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics as st
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'runs/pedestrian-finetune-cluster-results.zip'
PREFIX = 'runs/pedestrian-finetune-cluster/'
DEST = ROOT / 'results/pedestrian-finetune-cluster-review'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def quantile(values, p=.95):
    return sorted(values)[max(0, math.ceil(p*len(values))-1)] if values else None


def close(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-7), (a,b)


def main():
    checks = Counter()
    training_stats = {}
    anomalies, admission_anomalies, incomplete_people, arithmetic_ties = [], [], [], []
    with zipfile.ZipFile(ARCHIVE) as z:
        def read(name):
            return json.loads(z.read(name if name in z.namelist() else PREFIX+name))
        manifest = read('manifest.json')
        assert set(z.namelist()) == set(manifest) | {'manifest.json'}
        for name, expected in manifest.items():
            assert digest(z.read(name)) == expected, name
            checks['manifest_files'] += 1
        p, s, status = (read(n) for n in ['protocol.json','summary.json','status.json'])
        assert p == s['protocol'] and status['fingerprint'] == s['fingerprint'] == p['fingerprint']
        assert status['state'] == 'complete' and status['completed_simulations'] == 876
        assert status['completed'] == p['simulation_totals'] == dict(training=576,validation=54,baselines=54,evaluation=192)
        with zipfile.ZipFile(io.BytesIO(z.read(PREFIX+'source.zip'))) as source:
            assert set(source.namelist()) == set(p['source'])
            for name,h in p['source'].items():
                assert digest(source.read(name)) == digest(z.read(name)) == h
                checks['source_files'] += 1
        groups = [set(p['validation_seeds']),set(p['evaluation_seeds']),set(p['reserved_final_seeds_not_used']),
                  {p['training_seed_base']+r*p['training_seed_stride']+e for r in range(3) for e in range(64)}]
        assert all(not a & b for i,a in enumerate(groups) for b in groups[i+1:])
        parent = ROOT/'runs/guided-exploration-results.zip'
        assert digest(parent.read_bytes()) == p['starting_models']['archive_sha256']
        identities = {}
        with zipfile.ZipFile(parent) as previous:
            for label,info in p['starting_models']['checkpoints'].items():
                raw = z.read(PREFIX+'models/'+label+'/final.pt')
                assert digest(raw) == info['sha256'] == digest(previous.read(info['member']))
                identities[label] = info['sha256'];checks['parent_checkpoints'] += 1
        demand_identity, records, validation_rows = {}, {}, []
        for arm in p['arms']:
            for r in range(3):
                label = f'mappo-{arm}-r{r}'
                curve,done = (read(f'models/{label}/{name}.json') for name in ['learning','complete'])
                assert curve['fingerprint'] == done['fingerprint'] == p['fingerprint']
                assert done['episodes'] == 64 and done['parent_sha256'] == identities[f'mappo-original-r{r}']
                assert digest(z.read(PREFIX+f'models/{label}/final.pt')) == done['checkpoint_sha256']
                identities[label] = done['checkpoint_sha256'];checks['trained_checkpoints'] += 1
                assert [h['episodes'] for h in curve['history']] == list(range(1,65))
                assert [(v['episodes'],v['seed']) for v in curve['validations']] == [(e,seed) for e in p['validation_episodes'] for seed in p['validation_seeds']]
                stats = Counter(); early,late = Counter(),Counter()
                for h in curve['history']:
                    e = h['episodes'];seed = p['training_seed_base']+r*p['training_seed_stride']+e-1
                    assert h['seed'] == seed and h['scenario'] == p['training_scenarios'][(e-1)%len(p['training_scenarios'])]
                    pair = h['demand_sha256'],h['network_sha256']
                    assert demand_identity.setdefault((r,e),pair) == pair
                    raw = z.read(PREFIX+f'teacher-data/{label}/{e:03d}.json.gz')
                    assert digest(raw) == h['dataset_sha256']
                    data = json.loads(gzip.decompress(raw))
                    assert data['label'] == label and data['episode'] == e and data['seed'] == seed
                    assert data['parent_sha256'] == curve['parent_sha256'] == identities[f'mappo-original-r{r}']
                    assert (data['demand_sha256'],data['network_sha256']) == pair
                    base_keys = ['schema','horizon_seconds','urgency_seconds','features','data','counts','outcome_filtering','target']
                    assert digest(json.dumps({k:data[k] for k in base_keys},sort_keys=True,separators=(',',':')).encode()) == data['data_sha256']
                    counts = Counter()
                    for tls,local in data['data'].items():
                        n = len(local['labels']);assert all(len(v)==n for v in local.values())
                        assert local['times'] == sorted(set(local['times'])) and all(t<600 for t in local['times'])
                        u = h['update'][tls]
                        assert u['decisions'] == u['teacher_labels'] == n
                        assert u['teacher_selected'] == sum(local['guidance_weights'])
                        assert u['guidance_weight'] == (.5 if arm=='guided' else 0)
                        assert u['tail_weight'] == (0 if arm=='continue' else 1)
                        for i,(teacher,student,mask,weight,meta) in enumerate(zip(local['labels'],local['student_actions'],local['masks'],local['guidance_weights'],local['metadata'])):
                            assert sum(mask)>1 and mask[teacher] and mask[student]
                            assert teacher == meta['action'] and weight == float(meta['guidance_selected'])
                            values = dict(zip(data['features'][tls],local['observations'][i]))
                            ages = [values[k]*60 for k in values if k.endswith('/oldest') and values[k.removesuffix('/oldest')+'/count']>0]
                            selected = bool(mask[2] and ages and max(ages)>=p['learner']['urgency_seconds'])
                            assert bool(weight) == selected
                            if weight:assert teacher == 2
                            counts['free_decisions'] += 1
                            counts['selected_guidance_decisions'] += int(weight)
                            counts['student_teacher_disagreements'] += int(student!=teacher)
                            if weight:
                                counts['urgent_actions_following_teacher'] += int(student==teacher)
                        close(u['normalizer_count'],76672+e*599)
                    for key in ['free_decisions','selected_guidance_decisions','student_teacher_disagreements']:
                        assert counts[key] == data['counts'][key]
                    stats.update(counts)
                    if e<=16:early.update(counts)
                    if e>48:late.update(counts)
                    checks['training_episodes'] += 1
                training_stats[label] = dict(totals=dict(stats),first16=dict(early),last16=dict(late),
                    validation_vehicle_loss={str(e):st.mean(v['result']['vehicles']['completed_mean_time_loss_seconds'] for v in curve['validations'] if v['episodes']==e) for e in p['validation_episodes']})
                validation_rows.extend((v,f'validation-{label}-{v["episodes"]}-{v["seed"]}') for v in curve['validations'])
        assert len(validation_rows)==54

        def verify(row, filename, scenario, seed, label):
            result = row['result'];v=result['vehicles'];person=result['persons']
            for a in [v,row['vehicles_end']]:
                assert a['scheduled_count'] == sum(a[k] for k in ['completed_count','unfinished_count','uninserted_count'])
                close(a['completed_mean_time_loss_seconds']+a['completed_mean_depart_delay_seconds'],a['completed_mean_time_loss_plus_depart_delay_seconds'])
            assert person['scheduled'] == sum(person[k] for k in ['completed','unfinished','uninserted'])
            if person['unfinished']:incomplete_people.append(dict(label=label,scenario=scenario,seed=seed,persons=person))
            for diag in row['diagnostics'].values():close(sum(diag['phase_seconds'].values()),2400)
            raw = z.read(PREFIX+'events/'+filename+'.json.gz');assert digest(raw) == row['events_sha256']
            events = json.loads(gzip.decompress(raw));checks['event_files'] += 1
            for c,data in result['crossings'].items():
                all_visits = [e for e in events if e['crossing_id']==c]
                visits = [e for e in all_visits if e['reached_at'] is not None and 300<=e['reached_at']<2100]
                waits = [e['entered_at']-e['reached_at'] for e in visits if e['entered_at'] is not None]
                assert sorted(waits)==sorted(data['waits']) and len(waits)==data['entries']
                close(quantile(waits),data['p95']);close(st.mean(waits) if waits else None,data['mean'])
                assert sum(e['entered_at'] is None for e in visits)==data['unresolved_after_followup']
                assert sum(e['entered_at'] is None or e['entered_at']>=2100 for e in visits)==data['pending_at_cutoff']
                completion = row['crossing_completion'][c]
                assert sum(e['entered_at'] is not None and e['exited_at'] is None for e in all_visits)==completion['entered_without_observed_exit']
                assert sum(e['exited_at'] is not None for e in all_visits)==completion['completed_visits']
                if data['unknown_cohort_entries'] or data['lost'] or data['unresolved_after_followup']:
                    anomalies.append([label,scenario,seed,c,data['unknown_cohort_entries'],data['lost'],data['unresolved_after_followup']])
                tls=c.split('_')[0].lstrip(':');history=row['phase_history'][tls]
                times=[x[0] for x in history]
                for event in all_visits:
                    if event['entered_at'] is not None:
                        # The environment observes completed SUMO movement first,
                        # then applies a transition at the same timestamp for the
                        # next step. At an exact boundary, use the preceding state.
                        at = event['entered_at']
                        stage=history[max(0,bisect_left(times,at)-1)][1]
                        if bisect_left(times,at)!=bisect_right(times,at):
                            checks['entries_observed_at_phase_change']+=1
                        if stage!='PED':admission_anomalies.append([label,scenario,seed,c,event['entered_at'],stage])
                        checks['observed_crossing_entries']+=1
                checks['crossing_rows']+=1
            checks['traffic_runs']+=1

        for name in z.namelist():
            if name.startswith(PREFIX+'baselines/') and name.endswith('.json'):
                group=read(name);assert group['fingerprint']==p['fingerprint']
                assert [r['controller'] for r in group['records']]==p['baselines']
                for row in group['records']:
                    c,sc,seed=row['controller'],group['scenario'],group['seed']
                    verify(row,f'baseline-{c}-{sc}-{seed}',sc,seed,c);records[c,sc,seed]=row
        for name in z.namelist():
            if name.startswith(PREFIX+'evaluation/') and name.endswith('.json'):
                row=read(name);label,sc,seed=row['label'],row['scenario'],row['seed']
                assert row['fingerprint']==p['fingerprint'] and row['checkpoint_sha256']==identities[label]
                ref=records['actuated',sc,seed]
                assert (row['network_sha256'],row['result']['demand_sha256'])==(ref['network_sha256'],ref['result']['demand_sha256'])
                verify(row,f'{label}-{sc}-{seed}',sc,seed,label);records[label,sc,seed]=row
        for row,name in validation_rows:
            ref=records['actuated',row['scenario'],row['seed']]
            assert (row['network_sha256'],row['result']['demand_sha256'])==(ref['network_sha256'],ref['result']['demand_sha256'])
            verify(row,name,row['scenario'],row['seed'],name)

        def compare(left,right,saved,tag):
            nr,ns=len(left),len(left[0]);deltas={}
            for c in left[0][0]['crossings']:
                delta=st.mean(quantile([w for item in a for w in item['crossings'][c]['waits']])-quantile([w for item in b for w in item['crossings'][c]['waits']]) for a,b in zip(left,right))
                close(delta,saved['point']['crossing_p95_delta_seconds'][c]);deltas[c]=delta
            diffs=[[a['vehicles']['completed_mean_time_loss_seconds']-b['vehicles']['completed_mean_time_loss_seconds'] for a,b in zip(x,y)] for x,y in zip(left,right)]
            close(st.mean(x for row in diffs for x in row),saved['point']['vehicle_delta_seconds'])
            close(max(deltas.values()),saved['point']['worst_crossing_delta_seconds'])
            no_worse = sum(v<=1e-9 for v in deltas.values())
            assert no_worse == sum(v<=1e-9 for v in saved['point']['crossing_p95_delta_seconds'].values())
            if no_worse != saved['point']['crossings_no_worse']:
                arithmetic_ties.append(dict(comparison=tag,archived=saved['point']['crossings_no_worse'],ties_corrected=no_worse))
            failures,cutoff_only=0,0
            components=Counter()
            for ar,br in zip(left,right):
                for a,b in zip(ar,br):
                    av,bv=a['vehicles'],b['vehicles'];ap,bp=a['persons'],b['persons']
                    flags={
                        'fewer_completed_vehicles':av['completed_count']<bv['completed_count'],
                        'more_unfinished_vehicles':av['unfinished_count']>bv['unfinished_count'],
                        'more_uninserted_vehicles':av['uninserted_count']>bv['uninserted_count'],
                        'more_crossing_requests_pending_at_cutoff':any(a['crossings'][c]['pending_at_cutoff']>b['crossings'][c]['pending_at_cutoff'] for c in deltas),
                        'more_crossing_requests_unresolved_after_followup':any(a['crossings'][c]['unresolved_after_followup']>b['crossings'][c]['unresolved_after_followup'] for c in deltas),
                        'unknown_or_lost_measurements':any(v['lost'] or v['unknown_cohort_entries'] for row in [a,b] for v in row['crossings'].values()),
                        'worse_person_trip_service':ap['completed']<bp['completed'] or ap['unfinished']>bp['unfinished'] or ap['uninserted']>bp['uninserted'],
                    }
                    failures+=any(flags.values());components.update({k:int(v) for k,v in flags.items()})
                    cutoff_only+=flags['more_crossing_requests_pending_at_cutoff'] and not any(v for k,v in flags.items() if k!='more_crossing_requests_pending_at_cutoff')
            assert failures==saved['service_failure_pairs'] and cutoff_only==saved['cutoff_only_failure_pairs']
            assert dict(components)==saved['service_failure_components']
            if saved.get('exploratory_intervals'):
                rng=random.Random(424693);means=[]
                for _ in range(saved['bootstrap_requested_draws']):
                    rr=rng.choices(range(nr),k=nr);ss=rng.choices(range(ns),k=ns)
                    means.append(st.mean(diffs[r][seed] for r in rr for seed in ss))
                ci=saved['exploratory_intervals']['vehicle_delta_seconds']
                close(quantile(means,.025),ci[0]);close(quantile(means,.975),ci[1]);checks['vehicle_intervals']+=1
            checks['paired_comparisons']+=1

        extra={}
        for sc,d in s['scenarios'].items():
            matrices={name:[[records[name,sc,seed]['result'] for seed in p['evaluation_seeds']]] for name in p['baselines']}
            for arm in ['original']+p['arms']:
                matrices['mappo-'+arm]=[[records[f'mappo-{arm}-r{r}',sc,seed]['result'] for seed in p['evaluation_seeds']] for r in range(3)]
            extra[sc]={}
            for group in ['baselines','learned']:
                for name,row in d[group].items():
                    matrix=matrices[name];a=row['absolute'];flat=[x for m in matrix for x in m]
                    for out,field in [('vehicle_mean_time_loss','completed_mean_time_loss_seconds'),('vehicle_mean_insertion_delay','completed_mean_depart_delay_seconds'),('vehicle_mean_time_loss_plus_insertion_delay','completed_mean_time_loss_plus_depart_delay_seconds'),('vehicle_scheduled_count','scheduled_count'),('vehicle_completed_count','completed_count'),('vehicle_unfinished_count','unfinished_count'),('vehicle_uninserted_count','uninserted_count')]:
                        close(st.mean(x['vehicles'][field] for x in flat),a[out])
                    for c,v in a['crossings'].items():
                        waits=[[w for item in m for w in item['crossings'][c]['waits']] for m in matrix]
                        close(st.mean(quantile(w) for w in waits),v['mean_replica_pooled_p95'])
                        close(st.mean(st.mean(w) for w in waits),v['mean_wait']);assert [len(w) for w in waits]==v['entries_per_replica']
                    comps={'actuated':row['versus_actuated']} if group=='baselines' else row['comparisons']
                    for reference,comp in comps.items():
                        ref=matrices[reference]
                        compare(matrix,ref*len(matrix) if len(ref)==1 else ref,comp,f'{sc}/{name}-vs-{reference}')
                    rr=[records[(f'{name}-r{r}' if group=='learned' else name),sc,seed] for r in range(3 if group=='learned' else 1) for seed in p['evaluation_seeds']]
                    extra[sc][name]=dict(phase_fractions={stage:st.mean(diag['phase_seconds'].get(stage,0)/2400 for x in rr for diag in x['diagnostics'].values()) for stage in ['EW','NS','PED','PED_CLEARANCE','ALL_RED','EW_YELLOW','NS_YELLOW']},
                        max_queued_without_green=max(max(diag['max_queued_no_green_seconds'].values()) for x in rr for diag in x['diagnostics'].values()))
                    checks['aggregate_rows']+=1
            for name,c in d['paired_contrasts'].items():
                left,right=name.split('-vs-');compare(matrices[left],matrices[right],c,f'{sc}/{name}')
        DEST.mkdir(parents=True,exist_ok=True)
        for name in ['REPORT.md','summary.json','protocol.json','status.json']:
            (DEST/('archived_'+name)).write_bytes(z.read(PREFIX+name))
        result=dict(archive_sha256=digest(ARCHIVE.read_bytes()),checks=dict(checks),fingerprint=p['fingerprint'],
            measurement_anomalies=anomalies,admission_anomalies=admission_anomalies,incomplete_person_runs=incomplete_people,
            arithmetic_ties=arithmetic_ties,
            training=training_stats,diagnostics=extra,
            limitations=['No fresh SUMO replay or model deserialization.','Crossing bootstrap intervals were inspected but not independently regenerated.'])
        (DEST/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(checks,indent=2));print('Measurement/admission anomalies:',len(anomalies),len(admission_anomalies))
        print('Incomplete person runs (including validation):',len(incomplete_people))


if __name__=='__main__':
    main()
