"""Replay the three archived guided-study failures without changing their policy.

Reads only hash-verified source/checkpoints from the team's downloaded archive.
Source is isolated under runs/, so an active experiment's source is untouched.
Records observed positions and signal occupancy; does not bypass clearance,
enable SUMO's obstacle-ignoring jam recovery, or replace historical results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'runs/guided-exploration/'
CASES = [('mappo-age-r2', 890004), ('mappo-guided-r1', 890004),
         ('mappo-guided-r2', 890005), ('actuated', 890004),
         ('actuated', 890005), ('imitation-r2', 890004),
         ('imitation-r1', 890004), ('imitation-r2', 890005)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out[prefix + '/' + key] = [a.get(key), b.get(key)]
            else:
                out.update(differences(a[key], b[key], prefix + '/' + key))
        return out
    return {} if a == b else {prefix: [a, b]}


def split_diagnostic_roundoff(diffs):
    """Permit only floating diagnostic roundoff, never a traffic metric change."""
    material, roundoff = {}, {}
    for path, pair in diffs.items():
        allowed = (path.startswith('/diagnostics/') and
                   all(isinstance(x, float) for x in pair) and
                   math.isclose(*pair, rel_tol=1e-6, abs_tol=1e-8))
        (roundoff if allowed else material)[path] = pair
    return material, roundoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/guided-failure-replay')
    parser.add_argument('--case', type=int, choices=range(len(CASES)), action='append')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('Diagnostic output must stay inside this project runs directory')
    output.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'runs/guided-exploration-results.zip'
    snapshot = output / 'archive-source'
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('manifest.json'))

        def read(name):
            data = z.read(name)
            if sha(data) != manifest[name]:
                raise ValueError('Archive integrity mismatch: ' + name)
            return data

        protocol = json.loads(read(PREFIX + 'protocol.json'))
        for name in z.namelist():
            if name.startswith(('src/', 'config/')):
                target = (snapshot / name).resolve()
                if not target.is_relative_to(snapshot.resolve()):
                    raise ValueError('Archive path escapes snapshot')
                data = read(name)
                if name in protocol['source'] and sha(data) != protocol['source'][name]:
                    raise ValueError('Source differs from frozen protocol')
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError('Existing diagnostic source changed')
                target.write_bytes(data)

        # Only source/configuration are extracted. Checkpoints are loaded in memory.
        sys.path.insert(0, str(snapshot))
        sys.path.insert(0, str(ROOT / '.local/sumo-1.27.1/tools'))
        os.environ['PATH'] = str(ROOT / '.local/sumo-1.27.1/bin') + os.pathsep + os.environ['PATH']
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        import torch
        import traci
        from src.traffic_control import scenario
        from src.traffic_control.ppo_repair import pilot as base
        from src.traffic_control.fairness import environment
        from src.traffic_control.guided_exploration.pilot import episode
        from src.traffic_control.ppo_repair.learning import RepairNetworks
        from src.traffic_control.fairness.signals import SafeSignal
        if scenario.ROOT.resolve() != snapshot.resolve():
            raise AssertionError('Did not import isolated historical source')
        torch.set_num_threads(1)
        audit = {'archive_sha256': sha(archive.read_bytes()),
                 'protocol_fingerprint': protocol['fingerprint'],
                 'source': 'Hash-verified archived source, unchanged simulation and policy',
                 'instrumentation': 'Read-only sample callback and signal advance wrapper',
                 'cases': CASES, 'final_test_seeds_used': False,
                 'limits': 'Targeted post-hoc failure diagnosis, not an estimate of failure frequency. Historical collision settings retained for exact replay.'}
        write_json(output / 'protocol.json', audit)
        for case in args.case if args.case is not None else range(len(CASES)):
            label, seed = CASES[case]
            folder = output / f'{label}-{seed}'
            if (folder / 'complete.json').exists():
                print('Already replayed', label, seed, flush=True)
                continue
            # A completed replay may only differ in CPU floating-point diagnostic
            # reductions (e.g. 7e-9 entropy). Preserve the differences and verify
            # exact outcome identity rather than repeating hours of simulation.
            if (folder / 'diagnosis.json').exists():
                prior = json.loads((folder / 'diagnosis.json').read_text())
                remaining, roundoff = split_diagnostic_roundoff(prior['differences'])
                if (not remaining and prior['trace_sha256'] == sha((folder / 'positions.jsonl.gz').read_bytes())
                        and prior['normalized_demand_sha256'] == prior['historical_demand_hash']):
                    prior['exact_measured_outcomes'] = True
                    prior['diagnostic_roundoff_only'] = roundoff
                    # Never relabel approximate diagnostics as exactly identical.
                    write_json(folder / 'diagnosis.json', prior)
                    write_json(folder / 'complete.json', {'exact_measured_replay': True,
                        'diagnostics_within_numerical_tolerance': True,
                        'archive_sha256': audit['archive_sha256'],
                        'checkpoint_sha256': prior['checkpoint_sha256']})
                    print('Verified existing exact outcomes; diagnostic roundoff only', label, seed, flush=True)
                    continue
                raise ValueError('Existing replay has material mismatches; inspect before rerunning')
            folder.mkdir(exist_ok=True)
            model = None
            checkpoint_sha = None
            if label != 'actuated':
                ck = (f'teacher/final-r{label.split("-r")[-1]}.pt' if label.startswith('imitation-')
                      else f'models/{label}/final.pt')
                data = read(PREFIX + ck)
                checkpoint_sha = sha(data)
                saved = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
                if saved['fingerprint'] != protocol['fingerprint']:
                    raise ValueError('Checkpoint has wrong protocol')
                model = RepairNetworks.from_snapshot(saved['model'])
                model.eval()
                weights = {k: v.clone() for k, v in model.state_dict().items()}
                ref = json.loads(read(PREFIX + f'evaluation/{label}-short-spacing-{seed}.json'))
                if checkpoint_sha != ref['checkpoint_sha256']:
                    raise ValueError('Wrong evaluated checkpoint')
            else:
                rows = json.loads(read(PREFIX + f'baselines/short-spacing-{seed}.json'))['records']
                ref = next(r for r in rows if r['controller'] == 'actuated')
            print('Replaying', case, label, seed, flush=True)
            signals, stalled, maximum, blocks = {}, {}, {}, defaultdict(list)
            collision_counts = []
            original_run, original_advance = base.run, SafeSignal.advance
            with gzip.open(folder / 'positions.jsonl.gz', 'wt', encoding='utf-8') as stream:

                def sample(now, people, endpoints):
                    active = {p: s for p, s in people.items() if s.road in endpoints}
                    for person in set(stalled) - set(active):
                        stalled.pop(person)
                    for person, s in active.items():
                        if s.speed < .05:
                            old = stalled.get(person)
                            start = old['start'] if old and old['road'] == s.road else now
                            stalled[person] = {'start': start, 'road': s.road}
                            elapsed = now - start
                            if elapsed > maximum.get(person, {}).get('seconds', -1):
                                maximum[person] = dict(start=start, end=now, seconds=elapsed,
                                                       road=s.road, position=s.position)
                        else:
                            stalled.pop(person, None)
                    row = {'time': now, 'people': {p: asdict(s) for p, s in people.items()
                                                 if s.road.startswith(':')}}
                    # Sampling occurs before this tick's signal advance, matching
                    # the state under which the observed movement just occurred.
                    row['signals'] = {tls: {'stage': s.stage, 'since': s.since,
                                           'target': s.target, 'empty_since': s.empty_since}
                                      for tls, s in signals.items()}
                    prolonged = any(now - v['start'] >= 30 for v in stalled.values())
                    if prolonged and int(round(now * 10)) % 100 == 0:
                        row['internal_vehicles'] = {}
                        for vehicle in traci.vehicle.getIDList():
                            road = traci.vehicle.getRoadID(vehicle)
                            if road.startswith(':'):
                                row['internal_vehicles'][vehicle] = {
                                    'road': road, 'position': traci.vehicle.getPosition(vehicle),
                                    'speed': traci.vehicle.getSpeed(vehicle),
                                    'angle': traci.vehicle.getAngle(vehicle),
                                    'length': traci.vehicle.getLength(vehicle),
                                    'width': traci.vehicle.getWidth(vehicle)}
                    count = traci.simulation.getCollidingVehiclesNumber()
                    if count:
                        collision_counts.append([now, count])
                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')

                def advance(signal, now, occupied):
                    before = (signal.stage, signal.since)
                    result = original_advance(signal, now, occupied)
                    # SignalPlan has no junction id: creation/advance order is C0,C1,C2.
                    if signal not in signals.values():
                        signals[f'C{len(signals)}'] = signal
                    tls = next(k for k, s in signals.items() if s is signal)
                    if before[0] == 'PED_CLEARANCE' and (signal.stage != before[0]):
                        blocks[tls].append({'start': before[1], 'end': now,
                                            'seconds': now - before[1], 'ended': True})
                    return result

                captured = {}

                original_demand = environment.make_demand

                def demand(path, *demand_args, **kwargs):
                    value = original_demand(path, *demand_args, **kwargs)
                    captured['lf_demand_sha256'] = sha(Path(path).read_bytes().replace(b'\r\n', b'\n'))
                    return value

                def run(*run_args, **kwargs):
                    kwargs['sample_sink'] = sample
                    value = original_run(*run_args, **kwargs)
                    captured['phase_history'] = value['phase_history']
                    captured['events'] = json.loads((Path(run_args[2]) / 'events.json').read_text())
                    demand_bytes = (Path(run_args[2]) / 'demand_manifest.json').read_bytes()
                    captured['demand'] = json.loads(demand_bytes)
                    return value

                SafeSignal.advance, base.run = advance, run
                environment.make_demand = demand
                try:
                    out = episode(folder, protocol, 'short-spacing', seed,
                                  'actuated' if model is None else 'policy', model)
                finally:
                    SafeSignal.advance, base.run = original_advance, original_run
                    environment.make_demand = original_demand
            horizon = out['rollout']['horizon']
            for tls, s in signals.items():
                if s.stage == 'PED_CLEARANCE':
                    blocks[tls].append({'start': s.since, 'end': horizon,
                                        'seconds': horizon-s.since, 'ended': False})
            if model is not None and any(not torch.equal(v, model.state_dict()[k]) for k,v in weights.items()):
                raise AssertionError('Diagnostic changed model')
            # Old Windows demand output uses CRLF; semantic traffic is checked too.
            a, b = dict(ref['result']), dict(out['result'])
            old_hash, new_hash = a.pop('demand_sha256'), b.pop('demand_sha256')
            mismatches = differences(a, b)
            if old_hash != captured['lf_demand_sha256']:
                mismatches['/normalized_demand_sha256'] = [old_hash, captured['lf_demand_sha256']]
            mismatches.update(differences(ref['diagnostics'], out['diagnostics'], '/diagnostics'))
            mismatches.update(differences(ref['network_sha256'], out['network_sha256'], '/network'))
            if ref['sumo_version'] != out['sumo_version']:
                mismatches['/sumo_version'] = [ref['sumo_version'], out['sumo_version']]
            result = {'label': label, 'seed': seed, 'checkpoint_sha256': checkpoint_sha,
                      'historical_demand_hash': old_hash, 'local_demand_hash': new_hash,
                      'normalized_demand_sha256': captured['lf_demand_sha256'],
                      'exact_measured_outcomes_and_diagnostics': not mismatches,
                      'differences': mismatches, 'result': out['result'],
                      'phase_history': captured['phase_history'],
                      'clearance_intervals': dict(blocks),
                      'stationary_crossing_people': sorted(maximum.items(), key=lambda kv: -kv[1]['seconds'])[:20],
                      'detected_collision_counts': collision_counts,
                      'trace_sha256': sha((folder / 'positions.jsonl.gz').read_bytes())}
            material, roundoff = split_diagnostic_roundoff(mismatches)
            result['exact_measured_outcomes'] = not any(not k.startswith('/diagnostics/') for k in mismatches)
            result['diagnostic_roundoff_only'] = roundoff
            write_json(folder / 'diagnosis.json', result)
            with gzip.open(folder / 'events.json.gz', 'wt', encoding='utf-8') as f:
                json.dump(captured['events'], f)
            write_json(folder / 'demand.json', captured['demand'])
            if material:
                raise AssertionError('Replay differs; inspect diagnosis.json before causal interpretation')
            write_json(folder / 'complete.json', {'exact_measured_replay': True,
                        'diagnostics_within_numerical_tolerance': True,
                        'archive_sha256': audit['archive_sha256'], 'checkpoint_sha256': checkpoint_sha})
            print('Exact outcomes and diagnostics:', label, seed,
                  'longest stationary crossing seconds:', result['stationary_crossing_people'][:1], flush=True)


if __name__ == '__main__':
    main()
