"""Read-only audit of saved timing runs; optional fresh SUMO safety diagnostic.

No production source, checkpoint or saved study is modified. Fresh diagnostics
use development seeds and explicit junction collision detection.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.local/sumo-1.27.1/tools'))


def records_audit():
    report = {'runs': 0, 'entries_by_phase': Counter(), 'unknown_approaches': 0,
              'lost_events': 0, 'phase_timing_violations': [], 'examples': {}}
    for path in sorted((ROOT / 'runs/local-timing-study/records').glob('*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            row = json.load(stream)
        report['runs'] += 1
        histories = row['phase_history']
        durations = defaultdict(list)
        for junction, history in histories.items():
            for first, second in zip(history, history[1:]):
                seconds, stage = second[0] - first[0], first[1]
                durations[stage].append(seconds)
                required = {'EW': 12, 'NS': 12, 'EW_YELLOW': 3, 'NS_YELLOW': 3,
                            'ALL_RED': 2, 'PED': row['admission_seconds'], 'PED_CLEARANCE': 11}.get(stage, 0)
                if seconds + 1e-7 < required:
                    report['phase_timing_violations'].append([path.name, junction, first, seconds])
        for event in row['events']:
            report['unknown_approaches'] += event['reached_at'] is None
            report['lost_events'] += event['lost']
            if event['entered_at'] is None:
                continue
            history = histories[event['crossing_id'].split('_')[0][1:]]
            # Sampling precedes a signal change at the same timestamp.
            index = max(0, bisect.bisect_left([p[0] for p in history], event['entered_at']) - 1)
            report['entries_by_phase'][history[index][1]] += 1
        if row['seed'] == 910001 and row['admission_seconds'] == 12:
            report['examples'][f"{row['scenario']}/{row['controller']}"] = {
                'phase_durations': {phase: {'count': len(values), 'mean': sum(values)/len(values),
                                           'max': max(values)} for phase, values in durations.items()},
                'vehicles_at_cutoff': row['result']['vehicles'], 'vehicles_at_end': row['vehicles_end']}
    return report


def live_diagnostic(kind, output, horizon=600, scenario='short-spacing'):
    import traci
    from src.traffic_control.fairness.environment import run
    from src.traffic_control.fairness.demand import scenarios
    original_start = traci.start
    record = {'controller': kind, 'scenario': scenario, 'junction_collision_detection': True, 'collisions': [],
              'pedestrian_green_internal_vehicle_seconds': Counter(), 'examples': []}

    def start(command, *args, **kwargs):
        # The repaired runner already supplies the junction option. SUMO rejects
        # duplicate options, so replace existing values instead of appending twice.
        command = list(command)
        for option, value in (('--collision.check-junctions', 'true'),
                              ('--collision.action', 'warn'), ('--intermodal-collision.action', 'warn')):
            if option in command:
                command[command.index(option) + 1] = value
            else:
                command.extend((option, value))
        return original_start(command, *args, **kwargs)

    def sample(now, people, endpoints):
        for collision in traci.simulation.getCollisions():
            record['collisions'].append({'time': now, 'collision': str(collision)})
        if abs(now - round(now)) > 1e-8:
            return
        pedestrians_green = set()
        for junction in ('C0', 'C1', 'C2'):
            state = traci.trafficlight.getRedYellowGreenState(junction)
            links = traci.trafficlight.getControlledLinks(junction)
            if any(state[i] in 'Gg' and any(any(lane.startswith(c + '_') for c in endpoints)
                       for lane in connection) for i, group in enumerate(links) for connection in group):
                pedestrians_green.add(junction)
        for vehicle in traci.vehicle.getIDList():
            edge = traci.vehicle.getRoadID(vehicle)
            if not edge.startswith(':'):
                continue
            junction = edge[1:].split('_')[0]
            if junction in pedestrians_green:
                record['pedestrian_green_internal_vehicle_seconds'][junction] += 1
                if len(record['examples']) < 30:
                    record['examples'].append({'time': now, 'junction': junction, 'vehicle': vehicle,
                        'road': edge, 'speed': traci.vehicle.getSpeed(vehicle),
                        'position': traci.vehicle.getPosition(vehicle)})

    traci.start = start
    try:
        metrics = run(kind, 920101, output / scenario / kind, overrides={**scenarios()[scenario],
            'end_seconds': horizon, 'warmup_seconds': 0, 'measurement_end_seconds': horizon - 60,
            'demand_end_seconds': horizon - 60}, sample_sink=sample)
        record['status'] = 'complete'
        record['vehicles'] = metrics['vehicles_end']
    except Exception as error:
        record.update(status='failed', error=repr(error))
    finally:
        traci.start = original_start
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/deep-audit-simulation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = records_audit()
    if args.live:
        report['live'] = [live_diagnostic(controller, args.output) for controller in ('fixed', 'actuated', 'pressure')]
    # Preserve the original consolidated pre-repair evidence in audit.json.
    destination = args.output / ('live-audit.json' if args.live else 'records-audit.json')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'examples'}, indent=2))
    if args.live and any(r['status'] != 'complete' or r['collisions'] for r in report['live']):
        raise SystemExit('Live safety diagnostic failed; inspect ' + str(destination))


if __name__ == '__main__':
    main()
