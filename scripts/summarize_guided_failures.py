"""Summarise completed, verified replays without treating unfinished runs as data."""
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.replay_guided_failures import CASES


def main():
    source = ROOT / 'runs/guided-failure-replay'
    output = ROOT / 'results/failure-mechanism-review'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, seed in CASES:
        folder = source / f'{label}-{seed}'
        if not (folder / 'complete.json').exists():
            continue
        check = json.loads((folder / 'complete.json').read_text())
        if not check['exact_measured_replay']:
            raise ValueError('Unverified replay')
        d = json.loads((folder / 'diagnosis.json').read_text())
        with gzip.open(folder / 'events.json.gz', 'rt') as f:
            events = json.load(f)
        intervals = [(v['seconds'], tls, v) for tls, vv in d['clearance_intervals'].items() for v in vv]
        longest = max(intervals, key=lambda x: x[0])
        stationary = d['stationary_crossing_people'][0] if d['stationary_crossing_people'] else None
        row = {'label': label, 'seed': seed, 'exact_outcomes': True,
               'diagnostic_roundoff': d.get('diagnostic_roundoff_only', {}),
               'checkpoint_sha256': d['checkpoint_sha256'],
               'trace_sha256': d['trace_sha256'],
               'longest_clearance': {'junction': longest[1], **longest[2]},
               'longest_stationary_crossing_person': stationary,
               'unresolved_measured_crossing_requests': sum(c['unresolved_after_followup'] for c in d['result']['crossings'].values()),
               'crossing_visits_entered_without_observed_exit': [e for e in events if e['entered_at'] is not None and e['exited_at'] is None],
               'vehicles': d['result']['vehicles'], 'persons': d['result']['persons'],
               'historical_collision_counter_events': d['detected_collision_counts']}
        if stationary:
            person = stationary[0]
            with gzip.open(folder / 'positions.jsonl.gz', 'rt') as f:
                first = last = None
                for line in f:
                    r = json.loads(line)
                    if 'internal_vehicles' in r and person in r['people']:
                        example = {'time': r['time'], 'person': r['people'][person],
                                   'internal_vehicles': r['internal_vehicles']}
                        if first is None:
                            first = example
                        last = example
            row['first_and_last_stationary_vehicle_context'] = [first, last]
        rows.append(row)
    report = {'planned_replays': len(CASES), 'completed_verified_replays': len(rows),
              'complete': len(rows) == len(CASES), 'cases': rows,
              'scope': 'Targeted post-hoc replays; no retraining, replacement of scores or estimated population failure rate.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    lines = ['# Replayed congested failures', '',
             f'Verified replays: **{len(rows)}/{len(CASES)}**. This snapshot includes completed runs only.', '',
             '| Controller | Demand seed | Longest continuous clearance | Longest stationary crossing occupancy | Unresolved measured requests |',
             '|---|---:|---|---|---:|']
    for r in rows:
        c, s = r['longest_clearance'], r['longest_stationary_crossing_person']
        stop = f"{s[0]}: {s[1]['seconds']:.1f} s" if s else 'None observed'
        lines.append(f"| {r['label']} | {r['seed']} | {c['junction']}: {c['seconds']:.1f} s | {stop} | {r['unresolved_measured_crossing_requests']} |")
    lines += ['',
        'Demand (normalised line endings), network, vehicle/person/crossing outcomes and discrete diagnostics reproduce the archived study. Floating diagnostic reductions may differ by CPU roundoff; every observed difference is retained in summary.json. Traffic metrics have no roundoff exemption.', '',
        'The initial verified case, guided replica 2 / seed 890005, has pedestrian p000076 stopped on :C0_c2 from 315.8 until 2400 seconds. C0 remains continuously in pedestrian clearance from 320 to 2400 seconds. A stationary northbound vehicle v000252 occupies internal edge :C0_5 in the recorded prolonged-stall snapshots. This establishes occupied-crossing lockout of the actor; the coincident vehicle is a candidate obstruction. It does not by itself establish why that vehicle stopped or isolate the causal effect of removing it.', '',
        'This person entered the blocked crossing at 315.6 seconds after a measured wait of only 1.4 seconds, but never exited it within the run. Arrival-to-entry wait and safe completion of crossing are distinct outcomes. Keep entered-but-not-exited visits and full person-trip completion alongside queued requests; a short recorded entry wait does not establish successful pedestrian service.', '',
        'The safety machine correctly refuses to release conflicting vehicles while the crossing remains occupied. Once clearance is active, the actor has no legal choice. Earlier policy choices may contribute to the state that produces the obstruction; this replay does not absolve the policy or prove that the simulator alone is responsible.', '',
        'Historical settings are preserved for exact reproduction, including collision settings and disabling obstacle-ignoring pedestrian jam recovery. No detected collision in this replay is not a comprehensive safety certificate. Do not remove travellers, shorten clearance, enable unsafe movements, or replace the archived score to manufacture an improvement.', '',
        'The remaining matched replays determine whether the same demand jams under actuated and imitation-only control. These cases were selected because they failed, so they cannot estimate the frequency of rare failures. Separate any later all-controller geometry/simulator sensitivity study from this replay.', '',
        'Rebuild this snapshot with `python scripts/summarize_guided_failures.py`. Raw positions, event timestamps, demand and phase histories are retained under `runs/guided-failure-replay/`. See [the staged research protocol](../../docs/failure_explanation_protocol.md).', '']
    (output / 'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Verified replays: {len(rows)}/{len(CASES)}; report: {output / "REVIEW.md"}')


if __name__ == '__main__':
    main()
