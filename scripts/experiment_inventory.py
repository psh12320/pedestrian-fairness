"""Export compact evidence and verify the shareable catalogue; no simulation.

--check works on a clean checkout with the Python standard library.
--export-local is for the artifact owner's workspace, before committing updates.
It reads only named metadata from ZIPs and never executes or extracts their code.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'results/experiment-index'
REGISTRY = ROOT / 'docs/experiments/registry.json'
MANIFEST = INDEX / 'manifest.json'
AS_OF = '2026-11-05'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_bytes(path):
    # Historical tracked text may be checked out as LF or CRLF. Its public
    # content identity is newline-normalized; exports also retain a separate
    # exact original payload hash below. Binary/gzip records are byte-exact.
    raw = path.read_bytes()
    if path.suffix in {'.json', '.md', '.csv', '.tsv', '.svg', '.py', '.txt'}:
        return raw.replace(b'\r\n', b'\n'), 'lf-normalized'
    return raw, 'bytes'


def json_read(path):
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def inside(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path outside repository: {relative}')
    return path


def write_export(relative, raw, source, exports, compress=False):
    # Check before writing; never publish a silently truncated JSON document.
    if relative.endswith(('.json', '.json.gz')):
        json.loads(raw)
    path = inside(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0) if compress else raw)
    exports.append({'path': relative, 'source': source,
                    'payload_sha256': sha(raw), 'payload_bytes': len(raw),
                    'encoding': 'gzip' if compress else 'identity'})


def export_local():
    exports = []
    for source_dir, dest_dir in [
        ('runs/demand-service-test', 'results/demand-service-review'),
        ('runs/audit-repair-trial', 'results/audit-repair-review'),
    ]:
        for name in ('REPORT.md', 'summary.json', 'protocol.json', 'status.json'):
            source = f'{source_dir}/{name}'
            compress = source_dir.endswith('audit-repair-trial') and name == 'summary.json'
            dest = f'{dest_dir}/archived_{name}' + ('.gz' if compress else '')
            write_export(dest, inside(source).read_bytes(), source, exports, compress)

    for name in ('REPORT.md', 'protocol.json', 'primary_inference.json'):
        source = f'runs/cluster-final-review/{name}'
        write_export(f'results/fairness-v2-cluster-review/archived_{name}',
                     inside(source).read_bytes(), source, exports)
    for source, dest in [
        ('runs/pedestrian-finetune/status.json', 'results/experiment-index/partial_local_finetune_status.json'),
        ('runs/guided-failure-replay/protocol.json', 'results/failure-mechanism-review/archived_protocol.json'),
        ('runs/audit-repair-trial/recovery-20261103.json', 'results/audit-repair-review/archived_recovery-20261103.json'),
    ]:
        write_export(dest, inside(source).read_bytes(), source, exports)

    # Export all protocol members (small JSON only), preserving behavior/run labels.
    for study, name in [
        ('E05', 'service-priority-results.zip'),
        ('E06', 'service-policy-audit-results.zip'),
        ('E07', 'critic-fit-actuated-results.zip'),
        ('E07', 'critic-horizon-actuated-results.zip'),
        ('E07', 'critic-check-results.zip'),
        ('E08', 'critic-data-volume-results.zip'),
        ('E09', 'ppo-repair-results.zip'),
    ]:
        source = 'runs/' + name
        with zipfile.ZipFile(inside(source)) as archive:
            for member in archive.infolist():
                if Path(member.filename).name != 'protocol.json':
                    continue
                if member.file_size > 1024 * 1024:
                    raise ValueError(f'Unexpected protocol size: {source}:{member.filename}')
                label = Path(member.filename).parent.name or Path(name).stem
                dest = f'results/experiment-index/protocols/{study}/{label}.json'
                if any(e['path'] == dest for e in exports):
                    raise ValueError(f'Duplicate export: {dest}')
                write_export(dest, archive.read(member), f'{source}!{member.filename}', exports)

    registry = json_read(REGISTRY)
    archives, directories = [], []
    prefixes = [(prefix, s['id']) for s in registry['studies'] for prefix in s['local_run_prefixes']]
    for path in sorted((ROOT / 'runs').iterdir()):
        if path.is_file() and (path.name.endswith('.zip') or path.name.endswith('.tar.gz')):
            item = {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                    'sha256': sha(path.read_bytes()), 'shared_in_git': False}
            if path.suffix == '.zip':
                with zipfile.ZipFile(path) as archive:
                    item.update(members=len(archive.infolist()),
                                uncompressed_bytes=sum(i.file_size for i in archive.infolist()))
            archives.append(item)
        elif path.is_dir():
            files = [p for p in path.rglob('*') if p.is_file()]
            matches = sorted({study for prefix, study in prefixes if path.name.startswith(prefix)})
            directories.append({'path': path.relative_to(ROOT).as_posix(),
                                'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
                                'related_studies': matches,
                                'classification': 'related-output-or-engineering-check' if matches
                                else 'unclassified-local-artifact', 'shared_in_git': False})
    write_json(INDEX / 'local_artifacts.json', {
        'as_of': AS_OF, 'scope': 'One local workspace; no remote availability guarantee. Prefix association is not a completion claim.',
        'archives': archives, 'directories': directories,
        'known_incomplete_artifact': {
            'path': 'runs/cluster-final-review/comparisons.json',
            'reason': 'Status collector capped the file at 2097152 bytes; invalid truncated JSON. Not exported as results.'},
    })

    # Cover complete registered result folders, but not this manifest recursively.
    public = set()
    for study in registry['studies']:
        for name in study['evidence']:
            path = inside(name)
            if path.parent == INDEX:
                public.add(path)
            else:
                public.update(p for p in path.parent.rglob('*')
                              if p.is_file() and '__pycache__' not in p.parts)
    public.update(p for p in (INDEX / 'protocols').rglob('*.json'))
    public.add(INDEX / 'local_artifacts.json')
    public.discard(MANIFEST)
    records = []
    for p in sorted(public):
        raw, encoding = public_bytes(p)
        records.append({'path': p.relative_to(ROOT).as_posix(), 'hashed_bytes': len(raw),
                        'hash_encoding': encoding, 'sha256': sha(raw)})
    write_json(MANIFEST, {'schema_version': 1, 'as_of': AS_OF,
                         'scope': 'Public result files, not entire source history or raw runs.',
                         'files': records, 'exports': exports})
    print(f'Exported {len(exports)} compact records; indexed {len(records)} public files, '
          f'{len(archives)} local archives and {len(directories)} local run directories.')


def check():
    registry, manifest = json_read(REGISTRY), json_read(MANIFEST)
    ids = [s['id'] for s in registry['studies']]
    assert len(ids) == len(set(ids)), 'Duplicate study IDs'
    for study in registry['studies']:
        for relative in study['code'] + study['evidence'] + study['guides']:
            assert inside(relative).is_file(), f'Missing {study["id"]}: {relative}'
    names = [r['path'] for r in manifest['files']]
    assert len(names) == len(set(names)), 'Duplicate manifest paths'
    for record in manifest['files']:
        path = inside(record['path'])
        raw, encoding = public_bytes(path)
        assert (len(raw) == record['hashed_bytes'] and sha(raw) == record['sha256']
                and encoding == record['hash_encoding']), f'Changed: {path}'
        if path.name.endswith(('.json', '.json.gz')):
            json_read(path)
    for record in manifest['exports']:
        raw = inside(record['path']).read_bytes()
        if record['encoding'] == 'gzip':
            raw = gzip.decompress(raw)
        assert sha(raw) == record['payload_sha256'] and len(raw) == record['payload_bytes'], record['path']
        # On the owner's machine, additionally compare to the source. Clean
        # checkouts legitimately lack these sources and still validate exports.
        source, *member = record['source'].split('!', 1)
        if inside(source).exists():
            if member:
                with zipfile.ZipFile(inside(source)) as archive:
                    original = archive.read(member[0])
            else:
                original = inside(source).read_bytes()
            assert raw == original, f'Export differs from source: {source}'
    for doc in (ROOT / 'docs/experiments').glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            target = unquote(target.split('#', 1)[0])
            assert (doc.parent / target).exists(), f'Broken catalogue link: {doc.name} -> {target}'
    expected = {
        'results/demand-service-review/archived_status.json': ('completed', 720),
        'results/audit-repair-review/archived_status.json': ('completed_simulations', 1396),
        'results/pedestrian-finetune-cluster-review/archived_status.json': ('completed_simulations', 876),
    }
    for name, (key, count) in expected.items():
        status = json_read(inside(name))
        assert status['state'] == 'complete' and status[key] == count, name
    partial = json_read(INDEX / 'partial_local_finetune_status.json')
    assert partial['state'] == 'stopped' and partial['completed_simulations'] == 42
    print(f'PASS: {len(ids)} catalogue entries, {len(names)} public artifact hashes, '
          f'{len(manifest["exports"])} lossless exports, catalogue links and completion snapshots.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--export-local', action='store_true')
    args = parser.parse_args()
    if args.export_local:
        export_local()
    else:
        check()


if __name__ == '__main__':
    main()
