"""Read-only progress; absence from squeue is not proof of completion."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/guided-exploration'))
    args = parser.parse_args(); root = args.root
    if not (root/'protocol.json').exists():
        print('Study has not started in', root); return
    p = json.loads((root/'protocol.json').read_text())
    total = len(p['methods'])*len(p['variants'])*p['replicas']
    for method in p['methods']:
        for variant in p['variants']:
            for r in range(p['replicas']):
                label = f'{method}-{variant}-r{r}'; directory = root/'models'/label
                curve = directory/'learning.json'; done = directory/'complete.json'
                episode = json.loads(curve.read_text())['history'][-1]['episodes'] if curve.exists() else 0
                print(f'{label}: {episode}/{p["episodes"]}; completion marker={done.exists()}')
    expected = (total+p['replicas'])*len(p['evaluation_scenarios'])*len(p['evaluation_seeds'])
    print('Imitation checkpoints:', len(list((root/'teacher').glob('final-r*.json'))), '/', p['replicas'])
    print('Evaluation records:', len(list((root/'evaluation').glob('*.json'))), '/', expected)
    print('Report present:', (root/'REPORT.md').exists())
    print('Check sacct for COMPLETED and exit code 0:0; packaging audits completeness.')


if __name__ == '__main__': main()
