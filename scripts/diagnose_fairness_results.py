"""Read-only audit of complete cluster groups; never truncate structured JSON.

Run on the cluster after evaluation. Output contains no model weights/raw traces.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic_control.service_priority.control import service_components
from src.traffic_control.fairness.analysis import selection_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=Path('runs/fairness-v2-study'))
    p.add_argument('--output', type=Path, default=Path('runs/fairness-diagnosis.zip'))
    args = p.parse_args()
    protocol = json.loads((args.study / 'protocol.json').read_text())
    rows, hashes, labels = [], {}, defaultdict(list)
    for path in sorted((args.study / 'evaluation').glob('*.json.gz')):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, 'rt') as stream:
            group = json.load(stream)
        if group['fingerprint'] != protocol['fingerprint']:
            raise ValueError('Mismatched protocol in ' + str(path))
        baseline = next(r['result'] for r in group['records'] if r['controller'] == 'actuated')
        if len({r['result']['demand_sha256'] for r in group['records']}) != 1:
            raise ValueError('Unmatched demand')
        for record in group['records']:
            result = record['result']
            row = {k: group[k] for k in ('scenario', 'seed')}
            row.update(controller=record['controller'], replica=record['replica'],
                       components=service_components(result, baseline),
                       vehicles=result['vehicles'], persons=result.get('persons'),
                       score=selection_score(result, baseline),
                       crossings={c: {k: v for k, v in data.items() if k not in
                           ('waits', 'wait_lower', 'wait_upper', 'speed_groups')} for c, data in result['crossings'].items()})
            rows.append(row)
            labels[(row['scenario'], row['controller'])].append(row)
    summary = []
    for (scenario, controller), records in sorted(labels.items()):
        keys = [k for k, v in records[0]['components'].items() if isinstance(v, (int, float))]
        summary.append({'scenario': scenario, 'controller': controller, 'runs': len(records),
            'component_means': {k: statistics.mean(r['components'][k] for r in records) for k in keys},
            'component_nonzero_runs': {k: sum(r['components'][k] != 0 for r in records) for k in keys},
            'service_gate_failed_runs': sum(r['score'][0] > 0 for r in records)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('diagnosis.json', json.dumps({'groups': len(hashes), 'fingerprint': protocol['fingerprint'],
            'group_hashes': hashes, 'summary': summary, 'records': rows}))
        for name in ('REPORT.md', 'primary_inference.json', 'comparisons.json', 'protocol.json'):
            path = args.study / name
            if path.exists():
                archive.write(path, name)
    print(f'{len(hashes)} groups; {len(rows)} records; {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)')


if __name__ == '__main__':
    main()
