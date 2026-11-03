"""Small verified source bundle or complete results with final actor weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT/'runs/guided-exploration')
    args = parser.parse_args(); (ROOT/'runs').mkdir(exist_ok=True)
    if args.results:
        from src.traffic_control.guided_exploration.reporting import report
        root = args.root.resolve(); protocol = json.loads((root/'protocol.json').read_text())
        # Re-audit coverage, pairing and actual checkpoint hashes before exporting.
        report(root, protocol)
        output = ROOT/'runs'/('guided-exploration-smoke-results.zip' if protocol['quick'] else 'guided-exploration-results.zip')
        files = [*root.rglob('*.json'), root/'REPORT.md', *root.glob('models/*/final.pt'),
                 *root.glob('teacher/final-r*.pt')]
        for path in files:
            if path.suffix == '.json' and json.loads(path.read_text()).get('fingerprint') != protocol['fingerprint']:
                raise ValueError('Mixed result protocols: '+str(path))
        # Include matching source for independent review; never package old runs or archives.
        for relative, sha in protocol['source'].items():
            path = ROOT/relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError('Source changed during study: '+relative)
            files.append(path)
        files = sorted(set(files)); temporary = output.with_suffix('.tmp')
        manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in files: archive.write(path, path.relative_to(ROOT).as_posix())
            archive.writestr('manifest.json', json.dumps(manifest, indent=2)+'\n')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None: raise ValueError('Corrupt result archive')
        temporary.replace(output)
    else:
        output = ROOT/'runs/guided-exploration-source.tar.gz'
        files = []
        for folder, pattern in [('src','*.py'), ('config','*.json'), ('scenario','single.*.xml'), ('tests','*.py')]:
            files.extend((ROOT/folder).rglob(pattern))
        files += list((ROOT/'scripts').glob('*guided_exploration*'))
        files += [ROOT/'docs/guided_exploration.md', ROOT/'requirements.txt']
        files = sorted(set(files)); temporary = output.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as archive:
            for path in files: archive.add(path, arcname='trafficlightcontrol/'+path.relative_to(ROOT).as_posix())
        with tarfile.open(temporary, 'r:gz') as archive:
            if len(archive.getnames()) != len(set(archive.getnames())): raise ValueError('Duplicate paths')
            for path in files:
                name = 'trafficlightcontrol/'+path.relative_to(ROOT).as_posix()
                if archive.extractfile(name).read() != path.read_bytes(): raise ValueError('Source verification failed')
        temporary.replace(output)
    print(output)
    print(f'{output.stat().st_size/1024:.1f} KiB')
    print('SHA256:', hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__': main()
