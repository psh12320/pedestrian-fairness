"""Post-hoc saved-trace diagnostic of vehicle choices after clearance."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/demand-service-test'
DEST = ROOT / 'results/demand-service-review'


def main():
    totals = defaultdict(Counter)
    for path in (RUN / 'records').glob('*.json'):
        record = json.loads(path.read_text())
        raw = (RUN / 'decisions' / (path.stem + '.json.gz')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['decisions_sha256']
        counts = totals[record['scenario'], record['group']]
        counts['runs'] += 1
        for decision in json.loads(gzip.decompress(raw)):
            if not 300 <= decision['time'] < 2100 or decision['action'] not in (0, 1):
                continue
            values = dict(zip(record['features'][decision['junction']], decision['observation']))
            if not values['READY']:
                continue
            i = int(decision['junction'][1:])
            roads = [[f'h{i}e', f'h{i+1}w'], [f'ni{i}', f'si{i}']]
            queues = [sum(round(values[e + '/queue'] * 10) for e in axis) for axis in roads]
            selected, other = decision['action'], 1 - decision['action']
            counts['after_clearance_vehicle_choices'] += 1
            counts['selected_smaller_stopped_queue'] += queues[selected] < queues[other]
            counts['selected_zero_stopped_queue_while_other_queued'] += queues[selected] == 0 and queues[other] > 0
            counts['selected_no_observed_vehicles_while_other_queued'] += (
                all(values[e + '/count'] == 0 for e in roads[selected]) and queues[other] > 0)
    output = dict(scope='Post-hoc descriptive analysis of saved free vehicle choices after clearance in [300,2100). Counts are on each controller\'s own states; a smaller queue choice is not automatically an erroneous or causally harmful action.',
        groups={s: {g: dict(totals[s, g]) for sc, g in totals if sc == s} for s in sorted({s for s, _ in totals})})
    (DEST / 'vehicle_choice_diagnostic.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    for s, groups in output['groups'].items():
        print(s)
        for g in ['actuated', 'mappo-continue', 'mappo-early']:
            r = groups[g]
            print(g, {k: round(v / r['runs'], 2) for k, v in r.items() if k != 'runs'})


if __name__ == '__main__':
    main()
