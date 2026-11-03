"""Read local repair-study progress without importing Torch or starting SUMO."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/audit-repair-trial')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    path = args.root / 'status.json'
    if not path.exists():
        print('No saved status yet.')
        return
    status = json.loads(path.read_text())
    if args.json:
        print(json.dumps(status, indent=2))
        return
    print('Last recorded state:', status['state'], '| stage:', status.get('stage'))
    print('Updated:', status['updated_utc'], '| process:', status['pid'])
    print('Completed simulations:', status['completed_simulations'], '/', status['total_simulations'])
    for kind in ('training', 'validation', 'baseline', 'evaluation'):
        print(kind.capitalize() + ':', status[kind + '_completed'], '/', status[kind + '_total'])
    print('Report present:', (args.root / 'REPORT.md').exists())
    if status.get('error'):
        print('Error:', status['error'])
    if (args.root / 'STOP').exists():
        print('STOP marker present; workers stop before their next simulation.')
    print('A stale running status is not proof the process is still alive.')


if __name__ == '__main__':
    main()
