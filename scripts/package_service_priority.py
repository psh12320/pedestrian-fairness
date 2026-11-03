"""Small source/summary bundles; no credentials, models or raw SUMO traces."""
import argparse
from pathlib import Path
import tarfile
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.results:
        path = root / 'runs/service-priority-results.zip'
        files = list((root/'runs/service-priority-pilot').rglob('*.json'))
        files += list((root/'runs/service-priority-pilot').glob('*.md'))
        files += [root/'runs/fairness-diagnosis.zip']
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in files:
                if f.is_file():
                    z.write(f, f.relative_to(root).as_posix())
    else:
        path = root / 'runs/service-priority-source.tar.gz'
        path.parent.mkdir(exist_ok=True)
        files = []
        for folder, pattern in [('src','*.py'),('config','*.json'),('tests','*.py'),('scenario','single.*.xml')]:
            files.extend((root/folder).rglob(pattern))
        files += [root/'scripts'/n for n in ('diagnose_fairness_results.py','run_service_priority_cluster.sh', 'package_service_priority.py')]
        files += [root/'docs/service_priority_followup.md', root/'requirements.txt']
        with tarfile.open(path, 'w:gz') as tar:
            for f in sorted(files):
                if f.is_file():
                    tar.add(f, arcname='trafficlightcontrol/'+f.relative_to(root).as_posix())
    print(path)
    print(f'{path.stat().st_size/1024:.1f} KiB')


if __name__ == '__main__':
    main()
