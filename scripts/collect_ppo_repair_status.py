"""Read progress without loading weights or requiring Torch/SUMO imports."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('runs/ppo-repair-pilot'))
    args=parser.parse_args();root=args.root
    if not (root/'protocol.json').is_file():
        print('Not prepared yet. Check the scheduler and job log.');return
    p=json.loads((root/'protocol.json').read_text());fp=p['fingerprint']
    def read(path):
        if not path.exists():return None
        row=json.loads(path.read_text())
        if row.get('fingerprint')!=fp:raise ValueError('Protocol mismatch: '+str(path))
        return row
    nb=len(p['evaluation_scenarios'])*len(p['evaluation_seeds'])+len(p['validation_seeds'])
    expected=len(p['methods'])*len(p['variants'])*p['replicas']
    ne=expected*len(p['evaluation_scenarios'])*len(p['evaluation_seeds'])
    print('Baseline demand groups:',len(list((root/'baselines').glob('*.json'))),'/',nb)
    completed=0
    for method in p['methods']:
        for variant in p['variants']:
            for r in range(p['replicas']):
                label=f'{method}-{variant}-r{r}';folder=root/'models'/label
                curve=read(folder/'learning.json');done=read(folder/'complete.json')
                episode=curve['history'][-1]['episodes'] if curve and curve['history'] else 0
                completed+=done is not None
                print(f'{label:40} {episode:3}/{p["episodes"]} episodes; complete={done is not None}')
    print('Completed models:',completed,'/',expected)
    print('Evaluation records:',len(list((root/'evaluation').glob('*.json'))),'/',ne)
    print('Report present:',(root/'REPORT.md').exists())
    bundle='ppo-repair-smoke-results.zip' if p['quick'] else 'ppo-repair-results.zip'
    print('Results ZIP present:',(root.parent/bundle).exists())
    print('Progress files may lag the atomic checkpoint after interruption; resume uses the checkpoint.')
    print('A job disappearing from squeue does not prove success: inspect sacct and its exit code.')


if __name__=='__main__':main()
