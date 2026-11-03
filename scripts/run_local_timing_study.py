"""Single-worker, memory-light pedestrian admission timing experiment.

This is a conventional-control sensitivity study, not evidence of an RL gain.
No Torch dependency, training, or changes to the frozen cluster experiment.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SUMO = ROOT / '.local/sumo-1.27.1'
if (SUMO / 'tools/traci').is_dir():
    sys.path.insert(0, str(SUMO / 'tools'))
    os.environ.setdefault('SUMO_HOME', str(SUMO))
from src.traffic_control.fairness.environment import run
from src.traffic_control.fairness.demand import scenarios
from src.traffic_control.fairness.analysis import summarise, quantile, selection_score
from src.traffic_control.fairness.pipeline import atomic_json
from src.traffic_control.scenario import executable

CONTROLLERS = ('fixed', 'actuated', 'pressure')
DURATIONS = (12, 18, 24)
SETTINGS = ('vmedium-pmedium', 'pedestrian-burst', 'slow-walkers', 'short-spacing')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()


def protocol(quick=False):
    paths = [*sorted((ROOT/'src/traffic_control').rglob('*.py')), ROOT/'config/fairness_v2.json', Path(__file__)]
    p = {'schema': 'local-admission-timing-v1', 'quick': quick,
         'controllers': list(CONTROLLERS), 'admission_seconds': list(DURATIONS),
         'settings': ['vmedium-pmedium'] if quick else list(SETTINGS),
         'seeds': [910001] if quick else list(range(910001, 910005)),
         'horizon_seconds': 180 if quick else 2400,
         'measurement_window': [0,120] if quick else [300,2100],
         'reserved_final_seeds_not_used': list(range(58001,58021)),
         'purpose': 'Timing sensitivity and feasibility; no RL performance claim',
         'primary_comparison': 'Each controller at 18/24 versus itself at 12; identical demand/geometry',
         'secondary_comparison': 'Controllers versus actuated with the SAME admission setting',
         'safety': 'Original common SafeSignal; clearance, yellow, all-red and vehicle minimums unchanged',
         'resources': {'workers': 1, 'minimum_free_disk_mib': 512, 'output_limit_mib': 256},
         'python': platform.python_version(),
         'sumo': subprocess.run([executable('sumo'),'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],
         'source': {f.relative_to(ROOT).as_posix(): digest(f) for f in paths}}
    p['fingerprint'] = hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    return p


def configurations(p):
    return [(s, seed, duration, controller) for s in p['settings'] for seed in p['seeds']
            for duration in p['admission_seconds'] for controller in p['controllers']]


def record_path(root, job):
    scenario, seed, duration, controller = job
    return root/'records'/f'{scenario}-{seed}-{duration}s-{controller}.json.gz'


def read_record(path, p):
    if not path.exists(): return None
    with gzip.open(path,'rt',encoding='utf-8') as f: data=json.load(f)
    if data['fingerprint'] != p['fingerprint']: raise ValueError('Cached record has a different protocol')
    validate(data['result'])
    return data


def validate(result):
    v=result['vehicles']; people=result['persons']
    if v['scheduled_count'] != sum(v[k] for k in ('completed_count','unfinished_count','uninserted_count')):
        raise ValueError('Vehicle accounting failed')
    if people['scheduled'] != sum(people[k] for k in ('completed','unfinished','uninserted')):
        raise ValueError('Person accounting failed')
    if len(result['crossings']) != 12: raise ValueError('Expected twelve crossings')
    for c in result['crossings'].values():
        if len(c['waits']) != c['entries'] or quantile(c['waits']) != c['p95'] or any(w<0 for w in c['waits']):
            raise ValueError('Invalid crossing events')


def phase_checks(metrics):
    cfg=metrics['config']; checks=Counter()
    for history in metrics['phase_history'].values():
        for (start, stage),(end,_) in zip(history,history[1:]):
            elapsed=end-start
            if stage=='PED' and abs(elapsed-cfg['pedestrian_green_seconds'])>1e-6:
                raise AssertionError('Pedestrian admission duration differs from requested setting')
            minimum=(cfg['vehicle_green_min_seconds'] if stage in ('EW','NS') else
                     cfg['yellow_seconds'] if stage.endswith('_YELLOW') else
                     cfg['all_red_seconds'] if stage=='ALL_RED' else 0)
            if elapsed+1e-6<minimum: raise AssertionError('Minimum phase duration violated')
            checks[stage]+=1
    return dict(checks)


def execute(root,p,job):
    path=record_path(root,job)
    old=read_record(path,p)
    if old: return old
    if shutil.disk_usage(root).free < p['resources']['minimum_free_disk_mib']*1024**2:
        raise OSError('Less than the reserved 512 MiB free; stopping safely')
    if sum(f.stat().st_size for f in root.rglob('*') if f.is_file()) > p['resources']['output_limit_mib']*1024**2:
        raise OSError('Study exceeded its 256 MiB output budget')
    for relative,sha in p['source'].items():
        if digest(ROOT/relative)!=sha: raise ValueError('Source changed while experiment was running: '+relative)
    scenario,seed,duration,controller=job
    cfg={**scenarios()[scenario],'pedestrian_green_seconds':duration}
    if p['quick']:
        cfg.update(end_seconds=180,warmup_seconds=0,measurement_end_seconds=120,demand_end_seconds=90,burst_start_seconds=40)
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='simulation-',dir=root) as folder:
        try:
            metrics=run(controller,seed,folder,overrides=cfg)
            events=json.loads((Path(folder)/'events.json').read_text())
        except Exception:
            failure=Path(folder)/'failure.json'
            if failure.exists():
                shutil.copyfile(failure,root/'last_simulation_failure.json')
            raise
    result=summarise(metrics,events); validate(result)
    record={'fingerprint':p['fingerprint'],'scenario':scenario,'seed':seed,'admission_seconds':duration,
            'controller':controller,'result':result,'config':metrics['config'],'events':events,
            'network_sha256':metrics['network_sha256'],'phase_history':metrics['phase_history'],
            'phase_checks':phase_checks(metrics),'sumo_version':metrics['sumo_version'],
            'vehicles_end':metrics['vehicles_end'],'wall_seconds':time.perf_counter()-start}
    # Compare to any existing controller/duration on this exact scheduled demand.
    for existing in (root/'records').glob(f'{scenario}-{seed}-*.json.gz'):
        ref=read_record(existing,p)
        if (ref['result']['demand_sha256'],ref['network_sha256']) != (result['demand_sha256'],record['network_sha256']):
            raise AssertionError('Timing experiment changed demand or geometry')
    temporary=path.with_suffix('.tmp')
    with gzip.open(temporary,'wt',encoding='utf-8') as f: json.dump(record,f,sort_keys=True)
    temporary.replace(path)
    return record


def aggregate(rows):
    if not rows: raise ValueError('Empty aggregation')
    loss=[r['vehicles']['completed_mean_time_loss_seconds'] for r in rows]
    return {'vehicle_loss':st.mean(loss) if all(v is not None for v in loss) else None,
        'vehicles':{k:st.mean(r['vehicles'][k] for r in rows) for k in
                    ('scheduled_count','completed_count','unfinished_count','uninserted_count')},
        'persons':{k:st.mean(r['persons'][k] for r in rows) for k in ('scheduled','completed','unfinished','uninserted')},
        'crossings':{c:{'mean':st.mean(waits) if waits else None,'p95':quantile(waits),'entries':len(waits),
            'pending_cutoff':st.mean(r['crossings'][c]['pending_at_cutoff'] for r in rows),
            'unresolved':st.mean(r['crossings'][c]['unresolved_after_followup'] for r in rows)}
            for c in rows[0]['crossings']
            for waits in [[w for r in rows for w in r['crossings'][c]['waits']]]}}


def compare(rows,refs,draws=1000):
    if len(rows)!=len(refs) or not rows: raise ValueError('Unpaired comparison')
    if any(a['demand_sha256']!=b['demand_sha256'] for a,b in zip(rows,refs)):
        raise ValueError('Unmatched demands')
    @lru_cache(None)
    def point(indices):
        losses=[rows[i]['vehicles']['completed_mean_time_loss_seconds'] for i in indices]
        base=[refs[i]['vehicles']['completed_mean_time_loss_seconds'] for i in indices]
        vehicle=st.mean(a-b for a,b in zip(losses,base)) if all(v is not None for v in losses+base) else None
        deltas={}
        for c in rows[0]['crossings']:
            a=quantile([w for i in indices for w in rows[i]['crossings'][c]['waits']])
            b=quantile([w for i in indices for w in refs[i]['crossings'][c]['waits']])
            deltas[c]=a-b if a is not None and b is not None else None
        return {'vehicle_delta':vehicle,'crossing_p95_deltas':deltas,
            'worst_crossing_delta':max(deltas.values()) if all(v is not None for v in deltas.values()) else None,
            'crossings_no_worse':sum(v is not None and v<=0 for v in deltas.values())}
    estimate=point(tuple(range(len(rows))))
    minimum=min(sum(r['crossings'][c]['entries'] for r in group) for group in (rows,refs) for c in rows[0]['crossings'])
    failed=sum(selection_score(a,b)[0]>0 for a,b in zip(rows,refs))
    interval=None
    if len(rows)>=4 and draws:
        rng=random.Random(424691); samples=[point(tuple(sorted(rng.choices(range(len(rows)),k=len(rows))))) for _ in range(draws)]
        interval={k:([quantile([s[k] for s in samples],.025),quantile([s[k] for s in samples],.975)]
            if all(s[k] is not None for s in samples) else None) for k in ('vehicle_delta','worst_crossing_delta')}
    return {'point':estimate,'paired_seed_intervals':interval,'service_failure_pairs':failed,'pairs':len(rows),
        'minimum_pooled_entries':minimum,'sparse':minimum<100,
        'descriptive_joint_pass':bool(estimate['vehicle_delta'] is not None and estimate['vehicle_delta']<0 and
            estimate['worst_crossing_delta'] is not None and estimate['worst_crossing_delta']<=0 and failed==0)}


def report(root,p,draws=1000):
    data={job:read_record(record_path(root,job),p) for job in configurations(p)}
    if any(r is None for r in data.values()): raise ValueError('Cannot produce final report from incomplete study')
    results={}
    for scenario in p['settings']:
        groups={(duration,c):[data[scenario,seed,duration,c]['result'] for seed in p['seeds']]
                for duration in p['admission_seconds'] for c in p['controllers']}
        results[scenario]={f'{c}-{duration}s':{'absolute':aggregate(rows),
            'versus_same_controller_12s':compare(rows,groups[12,c],draws),
            'versus_same_duration_actuated':compare(rows,groups[duration,'actuated'],draws)}
            for (duration,c),rows in groups.items()}
    output={'fingerprint':p['fingerprint'],'protocol':p,'completed_runs':len(data),'scenarios':results,
            'note':'Exploratory timing sensitivity only. Four demand seeds, no learned models, no final test seeds.'}
    atomic_json(root/'summary.json',output)
    def fmt(value): return 'NA' if value is None else f'{value:.2f}'
    lines=['# Local pedestrian admission timing study','',
        f"Engineering smoke: **{p['quick']}**. Completed: {len(data)} simulations. Single CPU worker; no neural training.",'',
        '12, 18 and 24 seconds are admission periods. Separate protected clearance is unchanged. Every controller receives the same timing at each setting. Effects against each controller at 12 seconds are timing changes, not RL gains. Controllers are also compared against actuated at the same duration. No comparison with a differently timed baseline may be used as the original algorithm claim.','',
        'Metrics include scheduled/completed/unfinished/uninserted vehicles, whole person trips and crossing-cohort requests. P95 pools crossing events across matched demand seeds. Intervals resample whole paired seeds; four seeds provide limited uncertainty estimates. Sparse crossing samples (<100) are inconclusive. No final test seeds are used.','']
    for scenario,groups in results.items():
        lines += [f'## {scenario}','','| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Worst p95 change vs itself at 12s | Crossings no worse | Service failures |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for label,d in groups.items():
            a=d['absolute'];c=d['versus_same_controller_12s'];q=c['point'];v=a['vehicles']
            lines.append(f"| {label} | {fmt(a['vehicle_loss'])} | {fmt(v['completed_count'])} | {fmt(v['unfinished_count'])} | {fmt(v['uninserted_count'])} | {fmt(q['worst_crossing_delta'])} | {q['crossings_no_worse']}/12 | {c['service_failure_pairs']}/{c['pairs']} |")
        lines += ['','All crossing means/p95/counts, service outcomes, same-duration actuated comparisons and exploratory intervals are retained in summary.json. Raw event timestamps and phase histories are in compressed per-run records.','']
    lines += ['Synthetic geometry/demand; four development scenarios. Short spacing also increases vehicle demand. This study does not evaluate the Gini-based learning method or demonstrate deployed benefits.','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return output


@contextmanager
def exclusive_lock(root):
    lock=(root/'writer.lock').open('a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        lock.close(); raise RuntimeError('Another process already owns this experiment')
    try: yield
    finally:
        lock.seek(0)
        if os.name=='nt': msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        else: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/local-timing-study')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--max-new-runs',type=int)
    args=parser.parse_args();root=args.output.resolve()
    if args.status:
        print((root/'status.json').read_text() if (root/'status.json').exists() else 'Not started'); return
    root.mkdir(parents=True,exist_ok=True);(root/'records').mkdir(exist_ok=True)
    with exclusive_lock(root):
        p=protocol(args.quick);path=root/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=p: raise ValueError('Changed protocol/source; choose a new output directory')
        atomic_json(path,p)
        if args.report_only: report(root,p);return
        jobs=configurations(p); complete=sum(record_path(root,j).exists() for j in jobs); new=0
        def status(state,current=None,error=None):
            atomic_json(root/'status.json',{'fingerprint':p['fingerprint'],'state':state,'pid':os.getpid(),
                'completed':complete,'total':len(jobs),'current':current,'updated_utc':utc(),'error':error,
                'stop_instruction':'Create STOP in this output folder to stop after the current simulation.'})
        try:
            status('running')
            for job in jobs:
                if read_record(record_path(root,job),p): continue
                if (root/'STOP').exists(): status('stopped_after_current_simulation');return
                if args.max_new_runs is not None and new>=args.max_new_runs: status('batch_limit_reached');return
                status('running',job);value=execute(root,p,job); complete+=1;new+=1
                print(f'{complete}/{len(jobs)} {job} {value["wall_seconds"]:.1f}s',flush=True)
                status('running')
            report(root,p);status('complete')
            print('Complete:',root/'REPORT.md',flush=True)
        except BaseException as error:
            status('failed',error=repr(error));raise


if __name__=='__main__': main()
