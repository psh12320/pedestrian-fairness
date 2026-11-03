"""Package a small patch or verified completed diagnostic summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.critic_data_volume_check import EXPECTED_ACTORS, METHODS, INITIAL, validate_fit


def preflight():
    pilot = ROOT/'runs/service-priority-pilot'
    prior = json.loads((pilot/'protocol.json').read_text())
    if prior['quick']:
        raise ValueError('Pilot contains engineering smoke models, not the reviewed actors')
    if not list(pilot.glob('baseline-*.json')):
        raise ValueError('Missing pilot baseline feature definitions')
    for name, expected in prior['source'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Pilot source changed: '+name)
    for model, expected in EXPECTED_ACTORS.items():
        actual = hashlib.sha256((pilot/model/'best.pt').read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Actor differs from the reviewed diagnostic: '+model)
    print('Preflight passed: both original actor hashes and pilot source match.', flush=True)


def package_results():
    files = []
    for model in EXPECTED_ACTORS:
        folder = ROOT/f'runs/critic-data-{model}'
        protocol = json.loads((folder/'protocol.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        if summary['protocol'] != protocol or protocol['checkpoint_sha256'] != EXPECTED_ACTORS[model]:
            raise ValueError('Results/protocol mismatch: '+model)
        for method in METHODS:
            for initial in INITIAL:
                row = json.loads((folder/f'fit-{method}-{initial}.json').read_text())
                validate_fit(row, protocol, method, initial)
        if not (folder/'REPORT.md').is_file():
            raise ValueError('Missing report: '+model)
        files.extend([*folder.glob('*.json'), folder/'REPORT.md'])
    path = ROOT/'runs/critic-data-volume-results.zip'
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(files): archive.write(file,file.relative_to(ROOT).as_posix())
        archive.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
    return path


def package_source():
    names = ['scripts/critic_data_volume_check.py','scripts/run_critic_data_volume.sh',
             'scripts/package_critic_data_volume.py','scripts/critic_fit_check.py',
             'scripts/critic_horizon_check.py','tests/test_critic_data_volume.py',
             'tests/test_critic_fit.py','docs/critic_data_volume.md']
    path = ROOT/'runs/critic-data-volume-source.tar.gz'
    path.parent.mkdir(exist_ok=True)
    with tarfile.open(path,'w:gz') as archive:
        for name in names: archive.add(ROOT/name,arcname=name)
    with tarfile.open(path,'r:gz') as archive:
        if set(archive.getnames()) != set(names): raise ValueError('Unexpected patch members')
        for name in names:
            if archive.extractfile(name).read() != (ROOT/name).read_bytes():
                raise ValueError('Source bundle mismatch: '+name)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--results',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(); return
    path = package_results() if args.results else package_source()
    print(path)
    print(f'{path.stat().st_size/1024:.1f} KiB')


if __name__ == '__main__': main()
