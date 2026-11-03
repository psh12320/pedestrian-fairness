"""No-training, development-only evaluation of saved service-priority policies.

Compare argmax with seeded sampling and uniform legal choices. Do not use final
test seeds to select an evaluation mode. No source/model/old result is modified.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import statistics as st
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.traffic_control.fairness.analysis import summarise, selection_score
from src.traffic_control.fairness.environment import run
from src.traffic_control.fairness.pipeline import atomic_json
from src.traffic_control.connected import incoming_groups
from src.traffic_control.service_priority.control import service_mask
from src.traffic_control.service_priority.pilot import cfg_for


def explained_variance(targets, predictions):
    if len(targets) != len(predictions) or not targets:
        raise ValueError('Expected equal nonempty target and prediction arrays')
    variance = st.pvariance(targets)
    return 1 - st.pvariance(a-b for a,b in zip(targets,predictions))/variance if variance > 1e-12 else None


def choose_action(probabilities, mask, mode, rng):
    legal = [i for i, yes in enumerate(mask) if yes]
    if not legal:
        raise ValueError('No legal choice')
    if mode == 'argmax':
        return max(legal, key=lambda i: probabilities[i])
    if mode == 'uniform':
        return rng.choice(legal)
    if mode == 'sample':
        return rng.choices(legal, weights=[probabilities[i] for i in legal])[0]
    raise ValueError(mode)


def job_key(job):
    return f"{job['controller']}-{job['mode']}-d{job['seed']}-p{job['policy_seed']}"


def evaluate(job):
    import torch
    from src.traffic_control.fairness.networks import JunctionNetworks
    from src.traffic_control.fairness.learning import advantages
    torch.set_num_threads(1)
    pilot, output = Path(job['pilot']), Path(job['output'])
    path = output / (job_key(job)+'.json')
    if path.exists():
        cached = json.loads(path.read_text())
        if cached['job'] != job:
            raise ValueError('Cached job differs')
        return str(path)
    saved_protocol = json.loads((pilot/'protocol.json').read_text())
    baseline = json.loads((pilot/f"baseline-vmedium-pmedium-{job['seed']}.json").read_text())
    features = baseline['features']
    model = (JunctionNetworks.load(pilot/job['controller']/'best.pt', features)
             if job['mode'] in ('argmax','sample') else None)
    if model is not None:
        model.eval()
    # Separate deterministic streams per junction avoid cross-junction call-count coupling.
    rngs = {tls: random.Random(job['policy_seed']+i*1000003) for i,tls in enumerate(sorted(features))}
    choices = defaultdict(list)
    def policy(tls, obs, global_state, physical):
        effective = service_mask(obs, features[tls], physical) if job['priority'] else physical
        with torch.no_grad():
            probabilities = model.distribution(tls,obs,effective).probs.tolist() if model is not None else [float(x) for x in effective]
        if sum(effective)>1 and model is not None:
            import math
            ps = [p for p,legal in zip(probabilities,effective) if legal]
            choices[tls].append((max(ps),-sum(p*math.log(p) for p in ps if p>0),sorted(ps)[-1]-sorted(ps)[-2]))
        return choose_action(probabilities,effective,job['mode'],rngs[tls])
    trace=[]
    cfg=cfg_for(saved_protocol,'vmedium-pmedium')
    with tempfile.TemporaryDirectory(prefix='audit-',dir=output) as temp:
        metrics=run('actuated' if job['mode']=='baseline' else 'policy',job['seed'],temp,
                    overrides=cfg,policy=None if job['mode']=='baseline' else policy,trace_sink=trace)
        events=json.loads((Path(temp)/'events.json').read_text())
    result=summarise(metrics,events)
    reference=next(r['result'] for r in baseline['records'] if r['controller']=='actuated')
    if result['demand_sha256'] != reference['demand_sha256']:
        raise ValueError('Unmatched demand')
    diagnostics={}
    incoming = incoming_groups(3)
    for tls in features:
        local=[r for r in trace if r['junction']==tls]
        gaps={axis:0 for axis in ('EW','NS')}; longest=dict(gaps)
        for row in local:
            obs=dict(zip(features[tls],row['observation']))
            for axis,edges in incoming[tls].items():
                queued=any(obs[e+'/queue']>0 for e in edges)
                gaps[axis]=gaps[axis]+1 if queued and not obs[axis] else 0
                longest[axis]=max(longest[axis],gaps[axis])
        entry={'max_consecutive_queued_no_green_samples_1s':longest,
               'policy_free_decisions':len(choices[tls])}
        if choices[tls]:
            entry.update(mean_max_probability=st.mean(x[0] for x in choices[tls]),
                         mean_entropy=st.mean(x[1] for x in choices[tls]),
                         mean_top_two_probability_gap=st.mean(x[2] for x in choices[tls]))
        if model is not None:
            with torch.no_grad():
                values,_=model.values(tls,[r['observation'] for r in local[:-1]],
                                     [r['global_state'] for r in local[:-1]])
            values=values.tolist()
            rewards=[-(r['vehicle_queue']+r['uninserted_due']+3*r['pending_count'])/10 for r in local[1:]]
            _,gae=advantages(rewards,values)
            monte_carlo=[];future=0.
            for reward in reversed(rewards):
                future=reward+.99*future;monte_carlo.append(future)
            monte_carlo.reverse()
            entry.update(value_mean=st.mean(values),value_sd=st.pstdev(values),
                         discounted_return_mean=st.mean(monte_carlo),discounted_return_sd=st.pstdev(monte_carlo),
                         monte_carlo_explained_variance=explained_variance(monte_carlo,values),
                         gae_explained_variance=explained_variance(gae,values))
        diagnostics[tls]=entry
    original=None
    if job['mode']=='baseline': original=reference
    elif job['mode']=='argmax':
        original=next(r['result'] for r in json.loads((pilot/job['controller']/'evaluation.json').read_text())
                      if r['scenario']=='vmedium-pmedium' and r['seed']==job['seed'])
    replay={k:result[k]==original[k] for k in ('vehicles','persons','crossings')} if original is not None else None
    loss = result['vehicles']['completed_mean_time_loss_seconds']
    baseline_loss = reference['vehicles']['completed_mean_time_loss_seconds']
    data={'job':job,'result':result,'diagnostics':diagnostics,'replay_matches':replay,
          'service_score_vs_actuated':selection_score(result,reference)[0],
          'vehicle_difference_vs_actuated':loss-baseline_loss if loss is not None and baseline_loss is not None else None,
          'sumo_version':metrics['sumo_version']}
    atomic_json(path,data)
    return str(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,default=Path('runs/service-priority-pilot'))
    parser.add_argument('--output',type=Path,default=Path('runs/service-policy-audit'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    args.pilot=args.pilot.resolve();args.output=args.output.resolve()
    if args.workers<1 or args.output==args.pilot or args.pilot in args.output.parents:
        parser.error('Use positive workers and a separate output directory')
    protocol=json.loads((args.pilot/'protocol.json').read_text())
    for path,sha in protocol['source'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise ValueError('Source changed since pilot: '+path)
    if protocol['quick'] and not args.smoke:
        parser.error('Quick checkpoints require --smoke')
    models=sorted(p.parent.name for p in args.pilot.glob('*/complete.json'))
    expected=4*protocol['replicas']
    if len(models)!=expected:
        raise ValueError(f'Expected {expected} completed models, found {len(models)}')
    seeds=protocol['seeds'][:1] if args.smoke else protocol['seeds']
    if args.smoke: models=[m for m in models if m in ('plain-ippo-r0','priority-ippo-r0')]
    policy_seeds=[48001] if args.smoke else [48001,48002,48003]
    jobs=[]
    for seed in seeds:
        jobs.append(dict(controller='actuated',mode='baseline',seed=seed,policy_seed=0,priority=False))
        for priority in (False,True):
            for policy_seed in policy_seeds:
                jobs.append(dict(controller='uniform-priority' if priority else 'uniform-plain',mode='uniform',
                                 seed=seed,policy_seed=policy_seed,priority=priority))
        for model in models:
            jobs.append(dict(controller=model,mode='argmax',seed=seed,policy_seed=0,priority=model.startswith('priority-')))
            for policy_seed in policy_seeds:
                jobs.append(dict(controller=model,mode='sample',seed=seed,policy_seed=policy_seed,priority=model.startswith('priority-')))
    fingerprint=hashlib.sha256(json.dumps({'pilot':protocol,'models':{m:hashlib.sha256((args.pilot/m/'best.pt').read_bytes()).hexdigest() for m in models},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()
    for job in jobs:job.update(pilot=str(args.pilot),output=str(args.output),fingerprint=fingerprint)
    manifest={'status':'development diagnostic; no training and no untouched test seeds','smoke':args.smoke,
              'fingerprint':fingerprint,'jobs':jobs,
              'sampling':'Python categorical sampling; separate reproducible stream per junction; policy seeds 48001-48003',
              'critic_diagnostic':'Discounted Monte Carlo and GAE targets on evaluation trajectories, gamma .99; not a training-convergence test',
              'service_gap':'Consecutive one-second observations with an approach queue and no green; not individual vehicle waiting time'}
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Audit changed; use a new output directory')
    atomic_json(path,manifest)
    with ProcessPoolExecutor(args.workers) as pool:
        for i,path in enumerate(pool.map(evaluate,jobs),1):print(f'{i}/{len(jobs)} {Path(path).name}',flush=True)
    results=[json.loads((args.output/(job_key(j)+'.json')).read_text()) for j in jobs]
    mismatches=[job_key(r['job']) for r in results if r['replay_matches'] is not None and not all(r['replay_matches'].values())]
    lines=['# Policy evaluation diagnostic','',f'{len(results)}/{len(jobs)} evaluations complete. Replay mismatches: {len(mismatches)}.',
           'Development data only. Sampling is a diagnostic alternative, not a post-hoc final-test victory. Uniform controls distinguish learning from useful random phase allocation.','',
           '| Controller | Mode | Runs | Mean vehicle loss s | Completed | Uninserted | Service failures |',
           '|---|---|---:|---:|---:|---:|---:|']
    grouped=defaultdict(list)
    for r in results:grouped[r['job']['controller'],r['job']['mode']].append(r)
    for (label,mode),rs in sorted(grouped.items()):
        metric=lambda k:st.mean(r['result']['vehicles'][k] for r in rs)
        losses = [r['result']['vehicles']['completed_mean_time_loss_seconds'] for r in rs]
        loss_text = f'{st.mean(losses):.2f}' if all(v is not None for v in losses) else 'incomplete'
        lines.append(f"| {label} | {mode} | {len(rs)} | {loss_text} | {metric('completed_count'):.1f} | {metric('uninserted_count'):.1f} | {sum(r['service_score_vs_actuated']>0 for r in rs)} |")
    lines += ['', 'Counts describe the measurement cohort at its cutoff, not vehicle status after the follow-up. The JSON files retain unfinished vehicles, individual crossing waits and unresolved requests. A service-failure flag combines checks; it is not a count of unique unserved people.', '',
              'Probability and critic diagnostics are in each JSON file. Consecutive queued/no-green samples are a movement-service diagnostic, not an individual waiting-time measurement. These runs do not establish training convergence.']
    if mismatches:lines+=['','Replayed deterministic outcomes differ; investigate before attributing changes to action sampling.',*mismatches]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    archive_path=args.output.parent/(args.output.name+'-results.zip')
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted([*args.output.glob('*.json'),args.output/'REPORT.md']):bundle.write(path,path.name)
    print(f'Results: {archive_path} ({archive_path.stat().st_size/1024:.1f} KiB)')


if __name__=='__main__':
    main()
