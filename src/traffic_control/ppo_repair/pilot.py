"""Reproducible 18-run on-policy repair pilot, isolated from previous studies."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time
import torch
from ..scenario import ROOT,executable
from ..fairness.demand import scenarios
from ..fairness.environment import run
from ..fairness.analysis import summarise
from ..fairness.pipeline import atomic_json
from ..connected import incoming_groups
from . import SCHEMA
from .learning import VARIANTS,RepairNetworks,RepairLearner,raw_return,prediction_stats

METHODS=('ippo','mappo')
BASELINES=('fixed','actuated','pressure','uniform')
TRAIN_SCENARIOS=('vmedium-pmedium','vmedium-plow','pedestrian-burst','short-spacing')
EVAL_SCENARIOS=('vmedium-pmedium','pedestrian-burst','short-spacing','slow-walkers')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(protocol):
    return [(m,v,r,f'{m}-{v}-r{r}') for m in METHODS for v in VARIANTS for r in range(protocol['replicas'])]


def cfg_for(protocol,scenario,training=False):
    cfg=dict(scenarios()[scenario])
    if protocol['quick']:
        cfg.update(end_seconds=180,warmup_seconds=0,measurement_end_seconds=120,
                   demand_end_seconds=90,burst_start_seconds=40)
    elif training:
        cfg.update(end_seconds=600,warmup_seconds=0,measurement_end_seconds=600,demand_end_seconds=600)
    return cfg


def episode(root,protocol,scenario,seed,controller,model=None,training=False):
    cfg=cfg_for(protocol,scenario,training); horizon=cfg.get('end_seconds',2400)
    streams={tls:random.Random(900000+seed+i*1000003) for i,tls in enumerate(('C0','C1','C2'))}
    logged={tls:[] for tls in streams};trace=[]
    def policy(tls,obs,state,mask):
        legal=[i for i,yes in enumerate(mask) if yes]
        with torch.no_grad():
            probs=model.distribution(tls,obs,mask).probs.tolist() if model else [float(x) for x in mask]
        action=(legal[0] if len(legal)==1 else streams[tls].choices(legal,weights=[probs[i] for i in legal])[0])
        # Recompute through the distribution's log_prob, rather than rounded probabilities.
        if model:
            with torch.no_grad():lp=float(model.distribution(tls,obs,mask).log_prob(torch.tensor(action)))
            logged[tls].append(lp)
        return action
    with tempfile.TemporaryDirectory(prefix='rollout-',dir=root) as folder:
        metrics=run('policy' if model is not None or controller=='uniform' else controller,seed,folder,
                    overrides=cfg,policy=policy if model is not None or controller=='uniform' else None,trace_sink=trace)
        events=json.loads((Path(folder)/'events.json').read_text())
    features=metrics['observation_features'];diagnostics={}
    for tls in features:
        rows=[r for r in trace if r['junction']==tls]; index=0
        if model:
            for row in rows:
                if row['action'] is not None:
                    row['behavior_logp']=logged[tls][index];index+=1
            if index!=len(logged[tls]):raise AssertionError('Action/trace alignment mismatch')
        free=[r for r in rows if r['action'] is not None and sum(r['mask'])>1]
        counts=Counter(r['action'] for r in free)
        gap={axis:0 for axis in ('EW','NS')};longest=dict(gap)
        for row in rows:
            obs=dict(zip(features[tls],row['observation']))
            for axis,edges in incoming_groups(3)[tls].items():
                queued=any(obs[e+'/queue']>0 for e in edges)
                gap[axis]=gap[axis]+1 if queued and not obs[axis] else 0
                longest[axis]=max(longest[axis],gap[axis])
        history=metrics['phase_history'][tls]
        duration={}
        for (start,stage),(end,_) in zip(history,history[1:]+[(horizon,'')]):
            duration[stage]=duration.get(stage,0)+end-start
        d={'free_decisions':len(free),'free_action_counts':{str(i):counts[i] for i in range(3)},
           'max_queued_no_green_seconds':longest,'phase_seconds':duration}
        if model:
            # This 600-second return probe shares the training finite horizon.
            # It is evaluated before fitting; validation/evaluation never update moments.
            probe_horizon=min(horizon,600);probe=[r for r in rows if r['time']<=probe_horizon]
            with torch.no_grad():
                x=model.critic_inputs([r['observation'] for r in probe[:-1]],
                    [r['global_state'] for r in probe[:-1]],[r['time'] for r in probe[:-1]],probe_horizon)
                predicted=model.prediction(tls,x)
                rewards=[-(r['vehicle_queue']+r['uninserted_due']+3*r['pending_count'])/10 for r in probe[1:]]
                d['critic_probe']=prediction_stats(torch.tensor(raw_return(rewards)),predicted)
                d['critic_probe']['horizon_seconds']=probe_horizon
                if free:
                    distributions=model.distribution(tls,[r['observation'] for r in free],[r['mask'] for r in free])
                    d['mean_entropy']=float(distributions.entropy().mean())
        diagnostics[tls]=d
    return {'result':summarise(metrics,events),'features':features,'diagnostics':diagnostics,
            'sumo_version':metrics['sumo_version'],'network_sha256':metrics['network_sha256'],
            'rollout':{'trace':trace,'horizon':horizon}}


def cached_json(path,fp):
    if not path.exists():return None
    data=json.loads(path.read_text())
    if data.get('fingerprint')!=fp:raise ValueError('Cached result has different protocol: '+str(path))
    return data


def baseline_job(args):
    root,protocol,scenario,seed=args;root=Path(root)
    path=root/'baselines'/f'{scenario}-{seed}.json'
    cached=cached_json(path,protocol['fingerprint'])
    if cached:return str(path)
    records=[]
    for controller in BASELINES:
        outcome=episode(root,protocol,scenario,seed,controller);outcome.pop('rollout')
        records.append({'controller':controller,**outcome})
    if len({r['result']['demand_sha256'] for r in records})!=1 or len({r['network_sha256'] for r in records})!=1:
        raise AssertionError('Baseline demand/network mismatch')
    atomic_json(path,{'fingerprint':protocol['fingerprint'],'scenario':scenario,'seed':seed,'records':records})
    return str(path)


def atomic_torch(path,data):
    temporary=path.with_suffix('.tmp');torch.save(data,temporary);temporary.replace(path)


def train_job(args):
    root,protocol,method,variant,replica,label=args;root=Path(root)
    directory=root/'models'/label;directory.mkdir(parents=True,exist_ok=True)
    fp=protocol['fingerprint'];done=cached_json(directory/'complete.json',fp)
    if done:
        if digest(directory/'final.pt')!=done['checkpoint_sha256']:raise ValueError('Changed final checkpoint')
        return label
    torch.set_num_threads(1);torch.manual_seed(protocol['initial_seeds'][replica])
    first=json.loads(next((root/'baselines').glob('*.json')).read_text())['records'][0]
    model=RepairNetworks(method,first['features'],variant,protocol['initial_seeds'][replica])
    learner=RepairLearner(model);latest=directory/'latest.pt';completed=0;history=[];validations=[]
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['fingerprint']!=fp or saved['label']!=label:raise ValueError('Checkpoint protocol mismatch')
        model.load_state_dict(saved['model']['weights']);learner.optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']);completed=saved['episodes'];history=saved['history'];validations=saved['validations']
    batch_size=4 if variant.endswith('batch4') else 1
    while completed<protocol['episodes']:
        started=time.perf_counter();rollouts=[];demands=[]
        for index in range(completed,completed+batch_size):
            scenario=protocol['training_scenarios'][index%len(protocol['training_scenarios'])]
            seed=protocol['training_seed_base']+replica*1000+index
            outcome=episode(root,protocol,scenario,seed,'policy',model,True)
            rollouts.append(outcome['rollout'])
            demands.append({'episode':index+1,'scenario':scenario,'seed':seed,
                'demand_sha256':outcome['result']['demand_sha256'],'network_sha256':outcome['network_sha256'],
                'vehicles':outcome['result']['vehicles'],'diagnostics':outcome['diagnostics']})
        update=learner.update(rollouts);completed+=batch_size
        history.append({'episodes':completed,'optimizer_update_batches':len(history)+1,
                        'rollouts':demands,'update':update,'seconds':time.perf_counter()-started})
        if completed in protocol['validation_episodes']:
            for seed in protocol['validation_seeds']:
                outcome=episode(root,protocol,'vmedium-pmedium',seed,'policy',model);outcome.pop('rollout')
                ref=json.loads((root/'baselines'/f'vmedium-pmedium-{seed}.json').read_text())['records'][0]
                if outcome['result']['demand_sha256']!=ref['result']['demand_sha256']:raise AssertionError('Validation demand mismatch')
                validations.append({'episodes':completed,'scenario':'vmedium-pmedium','seed':seed,**outcome})
        # One atomic source of truth includes completed validation and all history.
        atomic_torch(latest,{'fingerprint':fp,'label':label,'episodes':completed,'model':model.snapshot(),
            'optimizer':learner.optimizer.state_dict(),'rng':torch.get_rng_state(),'history':history,'validations':validations})
        atomic_json(directory/'learning.json',{'fingerprint':fp,'label':label,'history':history,'validations':validations})
        print(label,'episodes',completed,flush=True)
    # Recover an interrupted JSON export even if latest.pt already reached the budget.
    atomic_json(directory/'learning.json',{'fingerprint':fp,'label':label,'history':history,'validations':validations})
    atomic_torch(directory/'final.pt',{'fingerprint':fp,'label':label,'episodes':completed,'model':model.snapshot()})
    atomic_json(directory/'complete.json',{'fingerprint':fp,'label':label,'episodes':completed,
        'update_batches':len(history),'checkpoint_sha256':digest(directory/'final.pt')})
    return label


def evaluation_job(args):
    root,protocol,label,scenario,seed=args;root=Path(root)
    path=root/'evaluation'/f'{label}-{scenario}-{seed}.json'
    cached=cached_json(path,protocol['fingerprint'])
    checkpoint=root/'models'/label/'final.pt';sha=digest(checkpoint)
    if cached:
        if cached['checkpoint_sha256']!=sha:raise ValueError('Evaluation checkpoint changed')
        return str(path)
    torch.set_num_threads(1)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['fingerprint']!=protocol['fingerprint'] or saved['episodes']!=protocol['episodes']:
        raise ValueError('Incomplete/wrong checkpoint')
    model=RepairNetworks.from_snapshot(saved['model']);model.eval()
    outcome=episode(root,protocol,scenario,seed,'policy',model);outcome.pop('rollout')
    ref=json.loads((root/'baselines'/f'{scenario}-{seed}.json').read_text())['records'][0]
    if outcome['result']['demand_sha256']!=ref['result']['demand_sha256'] or outcome['network_sha256']!=ref['network_sha256']:
        raise AssertionError('Evaluation demand/network mismatch')
    atomic_json(path,{'fingerprint':protocol['fingerprint'],'label':label,'scenario':scenario,'seed':seed,
                     'checkpoint_sha256':sha,**outcome})
    return str(path)


def make_protocol(quick=False):
    source=[*(ROOT/'src/traffic_control').rglob('*.py'),*(ROOT/'config').glob('*.json')]
    protocol={'schema':SCHEMA,'quick':quick,'status':'development pilot; no confirmatory claims',
        'episodes':4 if quick else 64,'replicas':1 if quick else 3,
        'initial_seeds':[72001] if quick else [72001,72002,72003],
        'training_seed_base':810001,'training_scenarios':list(TRAIN_SCENARIOS),
        'validation_episodes':[4] if quick else [16,32,64],'validation_seeds':[820001] if quick else [820001,820002],
        'evaluation_seeds':[830001] if quick else list(range(830001,830007)),
        'evaluation_scenarios':['vmedium-pmedium'] if quick else list(EVAL_SCENARIOS),
        'variants':list(VARIANTS),'methods':list(METHODS),'baselines':list(BASELINES),
        'reward_weight':3,'gamma':.99,'gae_lambda':.95,'ppo_epochs':4,
        'selection':'Final episode checkpoint only; validation does not select a model',
        'actor_evaluation':'categorical; fixed per-demand/per-junction streams shared across arms',
        'normalization':'cumulative training GAE-target moments; PopArt-style raw-output preservation; no evaluation updates',
        'budget':'Equal 64x600 simulation seconds/model; A/B 64 update batches; C 16. Four PPO passes per batch.',
        'reserved_final_seeds_not_used':list(range(58001,58021)),
        'python':platform.python_version(),'torch':str(torch.__version__),
        'sumo':subprocess.run([executable('sumo'),'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],
        'source':{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(source)}}
    train={protocol['training_seed_base']+r*1000+i for r in range(protocol['replicas']) for i in range(protocol['episodes'])}
    sets=[train,set(protocol['validation_seeds']),set(protocol['evaluation_seeds']),set(protocol['reserved_final_seeds_not_used'])]
    if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):raise AssertionError('Seed overlap')
    protocol['fingerprint']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/ppo-repair-pilot'))
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--stage',choices=('all','baselines','train','evaluate','report'),default='all')
    args=parser.parse_args()
    if args.workers<1:parser.error('Positive workers required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    protocol=make_protocol(args.quick);previous=cached_json(root/'protocol.json',protocol['fingerprint'])
    if previous and previous!=protocol:raise ValueError('Changed protocol')
    atomic_json(root/'protocol.json',protocol)
    for name in ('baselines','models','evaluation'):(root/name).mkdir(exist_ok=True)
    jobs=labels(protocol)
    if args.stage in ('all','baselines'):
        settings=sorted({(s,d) for s in protocol['evaluation_scenarios'] for d in protocol['evaluation_seeds']}|
                        {('vmedium-pmedium',d) for d in protocol['validation_seeds']})
        with ProcessPoolExecutor(args.workers) as pool:
            for path in pool.map(baseline_job,[(str(root),protocol,s,d) for s,d in settings]):print('Baseline',Path(path).name,flush=True)
    if args.stage in ('all','train'):
        with ProcessPoolExecutor(args.workers) as pool:
            for label in pool.map(train_job,[(str(root),protocol,*job) for job in jobs]):print('Trained',label,flush=True)
    if args.stage in ('all','evaluate'):
        with ProcessPoolExecutor(args.workers) as pool:
            for path in pool.map(evaluation_job,[(str(root),protocol,label,s,d) for _,_,_,label in jobs
                    for s in protocol['evaluation_scenarios'] for d in protocol['evaluation_seeds']]):print('Evaluated',Path(path).name,flush=True)
    if args.stage in ('all','report'):
        from .reporting import report
        report(root,protocol)


if __name__=='__main__':main()
