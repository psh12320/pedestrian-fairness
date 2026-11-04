"""Run the frozen fine-tuning experiment with an allocated CPU worker pool.

The laptop runner and its frozen source are unchanged. This orchestrator reuses
the same jobs, checkpoints, safety rules, seeds and reporting functions. Its own
hash is added to the cluster protocol; cluster and laptop outputs stay separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def stages(p):
    """Describe work explicitly so stage coverage can be verified without SUMO."""
    models = [(arm, r, f'mappo-{arm}-r{r}') for r in range(p['replicas']) for arm in p['arms']]
    all_models = [(arm, r, f'mappo-{arm}-r{r}') for r in range(p['replicas'])
                  for arm in ['original'] + p['arms']]
    yield 'prepare', 'baseline_job', [('vmedium-pmedium', seed) for seed in p['validation_seeds']]
    for target in p['checkpoint_stages']:
        yield f'train-{target}', 'train_job', [(*model, target) for model in models]
    yield 'evaluation-baselines', 'baseline_job', [
        (scenario, seed) for scenario in p['evaluation_scenarios'] for seed in p['evaluation_seeds']]
    yield 'evaluate', 'evaluation_job', [(*model, scenario, seed)
        for scenario in p['evaluation_scenarios'] for seed in p['evaluation_seeds'] for model in all_models]


def run(args):
    # Imports are deferred so --status never loads Torch or starts work on a login node.
    from scripts.pedestrian_finetune import pilot
    root = args.output.resolve()
    if args.status:
        print((root / 'status.json').read_text() if (root / 'status.json').exists() else 'Not started')
        return
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated and args.workers > int(allocated):
        raise ValueError('Requested more workers than allocated CPUs')
    root.mkdir(parents=True, exist_ok=True)
    pilot.torch.set_num_threads(1)
    with pilot.exclusive_lock(root):
        p = pilot.make_protocol(args.quick, args.parent_archive)
        entry = Path(__file__).resolve()
        p['source'][entry.relative_to(ROOT).as_posix()] = pilot.digest(entry)
        p['execution_entrypoint'] = 'scripts.pedestrian_finetune_cluster'
        p.pop('fingerprint')
        p['fingerprint'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
        old = pilot.cached_json(root / 'protocol.json', p['fingerprint'])
        if old and old != p:
            raise ValueError('Protocol changed; use the original frozen checkout to resume')
        pilot.atomic_json(root / 'protocol.json', p)
        for name in ('baselines', 'models', 'evaluation', 'events', 'teacher-data'):
            (root / name).mkdir(exist_ok=True)
        if not (root / 'source.zip').exists():
            with zipfile.ZipFile(root / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                for name in p['source']:
                    z.write(ROOT / name, name)
        pilot.verify_source_archive(root, p)
        pilot.prepare_parents(root, p, args.parent_archive)
        try:
            pilot.resource_check(root, p)
            for stage, function_name, jobs in stages(p):
                pilot.pool_stage(root, p, args.workers, stage, getattr(pilot, function_name),
                                 [(str(root), p, *job) for job in jobs])
            pilot.progress(root, p, stage='report')
            pilot.report(root, p)
            pilot.progress(root, p, state='complete', stage='complete')
            print('Complete:', root / 'REPORT.md', flush=True)
        except BaseException as exc:
            pilot.progress(root, p, state='stopped' if isinstance(exc, InterruptedError) else 'failed', error=repr(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 33), default=8)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pedestrian-finetune-cluster')
    parser.add_argument('--parent-archive', type=Path, default=ROOT / 'runs/guided-exploration-results.zip')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        path = args.output / 'status.json'
        print(path.read_text() if path.exists() else 'Not started')
        return
    run(args)


if __name__ == '__main__':
    main()
