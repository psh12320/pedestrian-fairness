"""Verify the shared E13 checkpoint ZIP without Torch, SUMO or extraction.

Optional --repair-line-endings restores only newline-equivalent frozen sources.
Actual code changes, missing files and unexpected archive bytes are rejected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = '255b82536b3a341c8964ef2338b5bd0a5e54fbe73dfd8e89fd9315b18c46354b'
PREFIX = 'runs/pedestrian-finetune-cluster/'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_repairs(root, frozen):
    """Plan all changes first. Reject non-newline differences before any write."""
    root = root.resolve()
    repairs = []
    for name, original in frozen.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or unsafe frozen source path: {name}')
        current = path.read_bytes()
        if current == original:
            continue
        if current.replace(b'\r\n', b'\n') != original.replace(b'\r\n', b'\n'):
            raise ValueError(f'Actual source difference: {name}. Use the frozen checkout; '
                             'this helper will not overwrite code changes.')
        repairs.append((path, original))
    return repairs


def verify(root, archive, repair=False):
    if not archive.is_file():
        raise ValueError(f'Missing archive: {archive}\n'
                         'Obtain the original 21,368,736-byte ZIP from the project owner. '
                         'Keep it zipped; the 116 MiB evaluation folder is a different artifact.')
    if sha(archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError('Archive SHA-256 mismatch. Obtain the exact original ZIP; do not repack it.')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('manifest.json'))

        def checked(name):
            raw = z.read(name)
            if sha(raw) != manifest.get(name):
                raise ValueError(f'Archive member hash mismatch: {name}')
            return raw

        protocol = json.loads(checked(PREFIX + 'protocol.json'))
        frozen = {}
        for name, expected in protocol['source'].items():
            raw = checked(name)
            if sha(raw) != expected:
                raise ValueError(f'Frozen protocol/source mismatch: {name}')
            frozen[name] = raw
        for arm in ('original', 'continue', 'tail', 'guided'):
            for replica in range(3):
                checked(PREFIX + f'models/mappo-{arm}-r{replica}/final.pt')
                if arm != 'original':
                    checked(PREFIX + f'models/mappo-{arm}-r{replica}/complete.json')

    repairs = source_repairs(root, frozen)
    if repairs and not repair:
        names = ', '.join(p.relative_to(root).as_posix() for p, _ in repairs)
        raise ValueError('Only line endings differ in: ' + names + '\n'
                         'Run again with --repair-line-endings to restore exact frozen bytes.')
    for path, raw in repairs:
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if source_repairs(root, frozen):
        raise ValueError('Source verification failed after line-ending repair.')
    return {'archive_bytes': archive.stat().st_size, 'sha256': ARCHIVE_SHA,
            'checkpoints': 12, 'frozen_source_files': len(frozen),
            'line_endings_repaired': len(repairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-line-endings', action='store_true')
    args = parser.parse_args()
    try:
        result = verify(ROOT, ROOT / 'runs/pedestrian-finetune-cluster-results.zip',
                        args.repair_line_endings)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, str(error) + '\n')
    print('PASS: ' + json.dumps(result))
    print('No simulation or training was started; checkpoint tensors were not deserialized.')


if __name__ == '__main__':
    main()
