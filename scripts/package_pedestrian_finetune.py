"""Package a small self-contained cluster input or verified completed results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_files():
    files = []
    for folder, pattern in [('src', '*.py'), ('config', '*.json'),
                            ('scenario', 'single.*.xml'), ('tests', '*.py')]:
        files.extend((ROOT / folder).rglob(pattern))
    files.extend((ROOT / 'scripts/pedestrian_finetune').glob('*.py'))
    files += [ROOT / name for name in [
        'scripts/pedestrian_finetune_cluster.py', 'scripts/package_pedestrian_finetune.py',
        'scripts/run_pedestrian_finetune_cluster.sh', 'docs/pedestrian_finetune.md',
        'docs/pedestrian_finetune_cluster.md', 'requirements.txt',
        'runs/guided-exploration-results.zip']]
    return sorted(set(files))


def make_source():
    files = source_files()
    output = ROOT / 'runs/pedestrian-finetune-source.tar.gz'
    temp = output.with_suffix('.tmp')
    manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    with tarfile.open(temp, 'w:gz') as z:
        for p in files:
            z.add(p, arcname='trafficlightcontrol/' + p.relative_to(ROOT).as_posix())
    with tarfile.open(temp, 'r:gz') as z:
        if len(z.getnames()) != len(set(z.getnames())):
            raise ValueError('Duplicate input paths')
        for name, sha in manifest.items():
            if hashlib.sha256(z.extractfile('trafficlightcontrol/' + name).read()).hexdigest() != sha:
                raise ValueError('Input archive verification failed: ' + name)
    temp.replace(output)
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return output


def make_results(root):
    from scripts.pedestrian_finetune import pilot
    root = root.resolve()
    if ROOT not in root.parents:
        raise ValueError('Results must be inside this checkout')
    p = json.loads((root / 'protocol.json').read_text())
    with pilot.exclusive_lock(root):
        status = json.loads((root / 'status.json').read_text())
        if status.get('state') != 'complete' or status.get('fingerprint') != p['fingerprint']:
            raise ValueError('Experiment has not completed successfully')
        pilot.report(root, p)  # Revalidate all pairing, sources, weights, events and demonstrations.
        files = sorted(set([*root.rglob('*.json'), *root.rglob('*.json.gz'),
            *root.glob('models/*/final.pt'), root / 'REPORT.md', root / 'source.zip'] +
            [ROOT / name for name in p['source']]))
        output = ROOT / 'runs' / ('pedestrian-finetune-cluster-smoke-results.zip' if p['quick']
                                  else 'pedestrian-finetune-cluster-results.zip')
        temp = output.with_suffix('.tmp')
        manifest = {f.relative_to(ROOT).as_posix(): digest(f) for f in files}
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f.relative_to(ROOT).as_posix())
            z.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise ValueError('Results archive integrity check failed')
        temp.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/pedestrian-finetune-cluster')
    args = parser.parse_args()
    (ROOT / 'runs').mkdir(exist_ok=True)
    output = make_results(args.root) if args.results else make_source()
    print(output)
    print(f'{output.stat().st_size / 1024**2:.2f} MiB')
    print('SHA256:', digest(output))


if __name__ == '__main__':
    main()
