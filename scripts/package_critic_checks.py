"""Small patch/summary bundle only; do not copy training data or old checkpoints."""
import argparse
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', action='store_true')
    args = parser.parse_args()
    if args.results:
        path = ROOT/'runs/critic-check-results.zip'
        files = []
        for model in ('plain-mappo-r0', 'priority-mappo-r0'):
            for kind in ('fit', 'horizon'):
                folder = ROOT/f'runs/critic-{kind}-{model}'
                if not (folder/'summary.json').is_file() or not (folder/'REPORT.md').is_file():
                    raise ValueError('Diagnostic incomplete: '+str(folder))
                files.extend([*folder.glob('*.json'), folder/'REPORT.md'])
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in sorted(files): z.write(file, file.relative_to(ROOT).as_posix())
    else:
        path = ROOT/'runs/critic-check-source.tar.gz'
        files = [ROOT/'scripts'/n for n in ('critic_fit_check.py', 'critic_horizon_check.py',
                                           'run_critic_checks.sh', 'package_critic_checks.py')]
        files += [ROOT/'tests/test_critic_fit.py', ROOT/'docs/critic_checks.md']
        with tarfile.open(path, 'w:gz') as z:
            for file in files: z.add(file, arcname=file.relative_to(ROOT).as_posix())
        with tarfile.open(path, 'r:gz') as z:
            for file in files:
                if z.extractfile(file.relative_to(ROOT).as_posix()).read() != file.read_bytes():
                    raise AssertionError('Source bundle mismatch')
    print(path)
    print(f'{path.stat().st_size/1024:.1f} KiB')


if __name__ == '__main__':
    main()
