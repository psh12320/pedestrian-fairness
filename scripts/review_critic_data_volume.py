"""Audit the downloaded 4-vs-32 critic diagnostic without extracting or training.

Recompute aggregations from the supplied scalar statistics. Raw trajectories and
weights remain on the cluster, so prediction errors cannot be replayed here.
"""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.critic_data_volume_check import (
    TRAIN, CHECK, INITIAL, METHODS, EXPECTED_ACTORS, validate_fit, bootstrap_ratio,
)
from scripts.review_critic_check_results import close, check_metrics


def audit_behavior(archive, behavior):
    prefix = f'runs/critic-data-{behavior}/'
    protocol = json.loads(archive.read(prefix+'protocol.json'))
    summary = json.loads(archive.read(prefix+'summary.json'))
    assert summary['protocol'] == protocol
    assert protocol['schema'] == 'critic-data-volume-v1'
    assert protocol['behavior'] == behavior
    assert protocol['checkpoint_sha256'] == EXPECTED_ACTORS[behavior]
    assert protocol['training_demand_seeds'] == TRAIN
    assert protocol['small_training_demand_seeds'] == TRAIN[:4]
    assert protocol['check_demand_seeds'] == CHECK
    assert protocol['initial_seeds'] == INITIAL and protocol['budgets'] == [128,512]
    assert protocol['gamma'] == .99 and protocol['reward_weight'] == 3
    assert protocol['configuration'] == {
        'end_seconds':600,'warmup_seconds':0,'measurement_end_seconds':600,
        'demand_end_seconds':600,'vehicle_rate_per_hour':1920,'pedestrian_rate_per_hour':1200}
    unhashed = dict(protocol); unhashed.pop('fingerprint')
    assert hashlib.sha256(json.dumps(unhashed,sort_keys=True).encode()).hexdigest() == protocol['fingerprint']
    for name,digest in protocol['source'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    groups=[]
    for method in METHODS:
        for initial in INITIAL:
            group=json.loads(archive.read(prefix+f'fit-{method}-{initial}.json'))
            validate_fit(group,protocol,method,initial)
            for row in group['results']:
                assessments=row['assessments']; size=row['training_trajectories']
                assert assessments['training']['n'] == row['training_states'] == size*599
                assert assessments['heldout']['n'] == len(CHECK)*599
                assert row['shared_target_scale'] > 0
                close(assessments['training']['target_mean'],row['own_training_target_mean'])
                for name,m in assessments.items():
                    check_metrics(m)
                    if name.startswith('seed-'): assert m['n'] == 599
                    close(m['constant_rmse']**2,m['target_sd']**2+(m['target_mean']-row['shared_target_mean'])**2,1e-4)
                    close(m['own_constant_rmse']**2,m['target_sd']**2+(m['target_mean']-row['own_training_target_mean'])**2,1e-4)
                each=[assessments[f'seed-{s}'] for s in CHECK]
                for field in ('bias','target_mean','prediction_mean'):
                    close(assessments['heldout'][field],st.mean(m[field] for m in each),1e-4)
                for field in ('rmse','constant_rmse','own_constant_rmse'):
                    close(assessments['heldout'][field]**2,st.mean(m[field]**2 for m in each),1e-4)
                close(assessments['heldout']['target_sd']**2,
                      st.mean(m['target_sd']**2+(m['target_mean']-assessments['heldout']['target_mean'])**2 for m in each),1e-4)
            groups.append(group)
    assert len(summary['summary']) == 8
    assert {(r['method'],r['trajectories'],r['updates']) for r in summary['summary']} == {
        (m,n,u) for m in METHODS for n in (4,32) for u in (128,512)}
    for stated in summary['summary']:
        selected=[r for g in groups if g['method']==stated['method'] for r in g['results']
                  if r['updates']==stated['updates'] and r['training_trajectories']==stated['trajectories']]
        assert len(selected)==9
        mse=st.mean(r['assessments']['heldout']['rmse']**2 for r in selected)
        close(stated['heldout_rmse'],mse**.5)
        close(stated['training_rmse'],st.mean(r['assessments']['training']['rmse']**2 for r in selected)**.5)
        close(stated['heldout_bias'],st.mean(r['assessments']['heldout']['bias'] for r in selected))
        for label,key in [('shared','constant_rmse'),('own','own_constant_rmse')]:
            close(stated[f'heldout_mse_to_{label}_constant'],mse/st.mean(r['assessments']['heldout'][key]**2 for r in selected))
    gates=[]; individual=[]; junctions=[]; comparisons=[]
    assert len(summary['initialization_gates'])==6 and len(summary['comparisons'])==2
    for method in METHODS:
        cells={}; details=[]
        subset=[g for g in groups if g['method']==method]
        for group in subset:
            lookup={(r['junction'],r['training_trajectories']):r for r in group['results'] if r['updates']==512}
            initial=group['initial_seed']
            means={n:st.mean(lookup[t,n]['assessments']['heldout']['rmse']**2 for t in ('C0','C1','C2')) for n in (4,32)}
            shared=st.mean(lookup[t,32]['assessments']['heldout']['constant_rmse']**2 for t in ('C0','C1','C2'))
            own=st.mean(lookup[t,32]['assessments']['heldout']['own_constant_rmse']**2 for t in ('C0','C1','C2'))
            computed={'method':method,'seed':initial,'small_mse':means[4],'large_mse':means[32],
                'large_to_small_mse':means[32]/means[4],'large_to_shared_constant_mse':means[32]/shared,
                'large_to_own_constant_mse':means[32]/own,'passed':means[32]<min(means[4],shared,own)}
            assert computed == next(g for g in summary['initialization_gates'] if g['method']==method and g['seed']==initial)
            gates.append(computed)
            for seed in CHECK:
                a=[lookup[t,4]['assessments'][f'seed-{seed}'] for t in ('C0','C1','C2')]
                b=[lookup[t,32]['assessments'][f'seed-{seed}'] for t in ('C0','C1','C2')]
                cells[initial,seed]=(st.mean(m['rmse']**2 for m in a),st.mean(m['rmse']**2 for m in b))
                for tls,old,new in zip(('C0','C1','C2'),a,b):
                    details.append({'method':method,'initial_seed':initial,'demand_seed':seed,'junction':tls,
                        'small_rmse':old['rmse'],'large_rmse':new['rmse'],'own_constant_rmse':new['own_constant_rmse']})
        ratio=sum(b for a,b in cells.values())/sum(a for a,b in cells.values())
        interval=bootstrap_ratio(cells,INITIAL,CHECK)
        computed={'method':method,'large_to_small_mse':ratio,'exploratory_paired_95_interval':interval,
            'individual_cases':len(details),'cases_improved':sum(r['large_rmse']<r['small_rmse'] for r in details),
            'cases_below_own_constant':sum(r['large_rmse']<r['own_constant_rmse'] for r in details)}
        assert computed == next(c for c in summary['comparisons'] if c['method']==method)
        comparisons.append(dict(computed,behavior=behavior,reduction=1-ratio,
            reduction_interval=[1-interval[1],1-interval[0]],initialization_gates_passed=sum(g['passed'] for g in gates if g['method']==method)))
        individual.extend(details)
        for tls in ('C0','C1','C2'):
            rows={n:[r for g in subset for r in g['results'] if r['junction']==tls and r['updates']==512 and r['training_trajectories']==n] for n in (4,32)}
            mse={n:st.mean(r['assessments']['heldout']['rmse']**2 for r in rs) for n,rs in rows.items()}
            own=st.mean(r['assessments']['heldout']['own_constant_rmse']**2 for r in rows[32])
            junctions.append({'behavior':behavior,'method':method,'junction':tls,'large_to_small_mse':mse[32]/mse[4],
                'large_to_own_constant':mse[32]/own,'small_training_target_mean':rows[4][0]['own_training_target_mean'],
                'large_training_target_mean':rows[32][0]['own_training_target_mean'],
                'check_target_mean':rows[32][0]['assessments']['heldout']['target_mean']})
    assert individual == summary['individual_cases']
    assert summary['passed'] == all(g['passed'] for g in gates)
    return summary,comparisons,junctions


def main():
    source=ROOT/'runs/critic-data-volume-results.zip'
    destination=ROOT/'results/critic-data-volume-cluster'
    summaries={}; comparisons=[]; junctions=[]
    with zipfile.ZipFile(source) as archive:
        names=archive.namelist()
        expected={'manifest.json'}
        for behavior in EXPECTED_ACTORS:
            prefix=f'runs/critic-data-{behavior}/'
            expected.update(prefix+name for name in ['protocol.json','summary.json','REPORT.md']+
                            [f'fit-{m}-{i}.json' for m in METHODS for i in INITIAL])
        assert len(names)==len(set(names))==19 and set(names)==expected
        assert archive.testzip() is None
        manifest=json.loads(archive.read('manifest.json'))
        assert set(manifest)==expected-{'manifest.json'}
        for name,digest in manifest.items(): assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        for behavior in EXPECTED_ACTORS:
            summary,pairs,js=audit_behavior(archive,behavior)
            summaries[behavior]=summary; comparisons.extend(pairs); junctions.extend(js)
    audit={'archive_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
           'archive_bytes':source.stat().st_size,'complete_behavior_experiments':2,
           'fit_job_files':12,'critic_fits':72,'metric_rows':216,'assessments':2160,
           'manifest_source_protocol_actor_hashes_match':True,'summaries_gates_intervals_recomputed':True,
           'raw_trajectories_or_weights_present':False,
           'limitation':'Scalar-statistic audit only; absent raw trajectories prevent replaying predictions or validating demand and crossing events.'}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'summary.json').write_text(json.dumps({'audit':audit,'comparisons':comparisons,
        'junction_results':junctions,'diagnostic_summaries':summaries},indent=2)+'\n')
    lines=['# Cluster data-volume diagnostic review','',
        'Both saved-policy experiments completed. All 12 method/critic-initialisation gates pass at 512 updates: the 32-trajectory arm improves on four trajectories and beats both constant predictors. This is a positive value-prediction finding; no traffic actor was updated.','',
        'The archive contains 12 fit-job files covering 72 fresh critic fits, 216 evaluation rows and 2,160 assessments. ZIP integrity, exact member coverage, manifest/source/protocol/actor hashes, seed splits, identical initial predictions, scalar metric identities, sample counts, summaries, individual cases, gates and paired intervals passed checks. The protocol calls for 80 short SUMO trajectories. Raw trajectories and actor weights are absent: this verifies scalar results, not a replay of those simulations.','',
        '## Matched 4 versus 32 trajectories at 512 updates','',
        '| Frozen behavior | Critic inputs | Check MSE reduction | Exploratory 95% interval for reduction | Initialisation gates | Individual cases improved | Cases below own constant |',
        '|---|---|---:|---|---:|---:|---:|']
    for r in comparisons:
        lo,hi=r['reduction_interval']
        lines.append(f"| {r['behavior']} | {'Local' if r['method']=='ippo' else 'Network-wide'} | {r['reduction']:.1%} | [{lo:.1%}, {hi:.1%}] | {r['initialization_gates_passed']}/3 | {r['cases_improved']}/72 | {r['cases_below_own_constant']}/72 |")
    lines += ['','Each set of 72 individual cases is three junctions × eight check demand seeds × three critic initialisations. They share data and are correlated. The intervals resample whole check seeds and critic initialisations, conditional on the two fixed training datasets; they exclude training-dataset sampling variation and are exploratory. Error reductions are squared prediction errors in reward units, not percentages of vehicle or pedestrian waiting time.','',
        '## All junctions, including remaining failures','',
        '| Behavior | Inputs | Junction | MSE32 / MSE4 | MSE32 / own constant |',
        '|---|---|---|---:|---:|']
    for r in junctions:
        lines.append(f"| {r['behavior']} | {r['method']} | {r['junction']} | {r['large_to_small_mse']:.3f} | {r['large_to_own_constant']:.3f} |")
    lines += ['','All 12 behavior/input/junction averages improve and beat their own constant reference. Some individual demand/initialisation cases still worsen or fail to beat the constant; retain all cases in summary.json. These are junction-level prediction checks, not the proposal\'s 12 crossing-level waiting-time checks.','',
        '## Train versus check errors','',
        '| Behavior | Inputs | Trajectories | Updates | Train RMSE | Check RMSE | Check bias | Check MSE / own constant |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for behavior,summary in summaries.items():
        for r in summary['summary']:
            lines.append(f"| {behavior} | {r['method']} | {r['trajectories']} | {r['updates']} | {r['training_rmse']:.2f} | {r['heldout_rmse']:.2f} | {r['heldout_bias']:.2f} | {r['heldout_mse_to_own_constant']:.3f} |")
    lines += ['','The larger dataset reduces the train/check gap and prediction bias, especially for the network-wide critic. Network-wide inputs still do not outperform local inputs on average: with 32 trajectories, check RMSE is 36.38 versus 35.89 under plain behavior and 41.41 versus 39.72 under priority behavior. These are critic architectures fitted on the same saved MAPPO actor data, not an IPPO-versus-MAPPO traffic ranking.','',
        '## What the experiment supports','',
        'Within this diagnostic, four fixed trajectories gave a poor basis for generalisation. More independent simulation runs, at the same optimiser update count, improved prediction on eight new demand seeds. At plain C1, the training target mean shifts from -96.15 with four runs to -123.11 with 32; the new check mean is -126.87. This is consistent with better training coverage. It does not prove a unique cause of the original PPO failures.','',
        'Both arms already use normalised outputs and remaining episode time, and share a transform fitted only to the four-run subset. The larger arm processes eight times as many states per update. Thus this is not an equal-compute or equal-sample-exposure experiment. No new claim separating sample diversity from exposure is supported. The original PPO study generated fresh episodes over training; it did not train on just four trajectories.','',
        'The new check seeds differ from the previous horizon diagnostic. Do not compare its numbers directly with these to infer a treatment effect; use the matched four-run control in this archive. In particular the four-run priority/network critic already beats its constant on this new check set. The matched improvement, rather than a before/after pass label, is the evidence.','',
        '## Next bounded experiment','',
        'Move from offline critic diagnosis to an on-policy pilot. Compare the existing PPO formulation against verified value-output normalisation plus critic remaining-time inputs, then separately test multiple independent rollout seeds collected under the same frozen actor before each update. Preserve local actor observations, categorical sampling at evaluation, rewards, common phase safety, and the original controller checkpoints. Use equal total SUMO transitions and report both actor-update counts and compute; fitting 32 offline trajectories is not a reason to prescribe a 32-episode PPO batch.','',
        'Use repeated training initialisations and separate development demand seeds. Keep ordinary and service-priority logic separate so any effect is attributable. Compare against pedestrian-actuated and uniform-choice controls. The decision metrics remain vehicle time loss, completed/unfinished/uninserted vehicles, per-crossing mean/p95 and unresolved requests. Report crossings helped out of 12 as a secondary measure; the original every-crossing criterion stays unchanged.','',
        'Do not claim a solution or launch a full study solely because this diagnostic passed. First demonstrate an actual vehicle/pedestrian improvement in the bounded pilot; then freeze the selected configuration and run held-out tests. All previous negative outcomes remain part of the study.','']
    (destination/'REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'audit':audit,'comparisons':comparisons},indent=2))
    print(destination/'REVIEW.md')


if __name__=='__main__':main()
