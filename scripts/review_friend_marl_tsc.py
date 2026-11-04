"""Read-only audit of the separately extracted friend archive.

Recomputes reported statistics without importing its analysis or loading model
pickles. Small deterministic probes exercise the inspected tracker/signal code;
they are unit counterexamples, not new SUMO performance experiments.
"""
import csv
from collections import defaultdict, Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/friend-marl-tsc-review/marl_tsc'
OUT = ROOT / 'results/friend-marl-tsc-review'
CONTROLLERS = ('ped_actuated', 'fixed_time', 'ippo_veh', 'ippo_ped', 'mappo_veh', 'mappo_ped')


def quantile(values, p=.95):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p
    lower, upper = math.floor(x), math.ceil(x)
    return values[lower] + (values[upper] - values[lower]) * (x - lower)


def rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def bool_value(value):
    return value.lower() == 'true'


def verify_archive_files():
    manifest = json.loads((SOURCE.parent / 'archive-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((SOURCE.parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Extracted archive source changed: ' + name)
    return manifest['archive_sha256']


def raw_statistics():
    thresholds = json.loads((SOURCE / 'results/thresholds.json').read_text())
    supplied_p95 = {(r['controller'], r['scenario'], r['level'], r['crossing']): r
                    for r in rows(SOURCE / 'results/report/ped_crossing_p95.csv')}
    supplied_vehicle = {(r['controller'], r['scenario'], r['level']): r
                        for r in rows(SOURCE / 'results/report/vehicle_metrics.csv')}
    episode_tables = {}
    pedestrian_stats = {}
    ref_groups = None
    for name in CONTROLLERS:
        records = rows(SOURCE / f'results/eval/{name}.episodes.csv')
        keys = {(r['scenario'], r['level'], int(r['seed'])) for r in records}
        expected_keys = {(sc, f'v{v}-p{ped}', seed) for sc in ('normal', 'slow')
                         for v in ('low', 'med', 'high') for ped in ('low', 'med', 'high')
                         for seed in range(10, 20)}
        assert len(records) == 180 and keys == expected_keys
        assert all(r['controller'] == name for r in records)
        for r in records:
            assert int(r['veh_demand']) == int(r['veh_arrived']) + int(r['veh_unfinished'])
        episode_tables[name] = {k: r for k, r in zip(
            [(r['scenario'], r['level'], int(r['seed'])) for r in records], records)}
        if name != 'ped_actuated':
            for key, r in episode_tables[name].items():
                ref = episode_tables['ped_actuated'][key]
                assert (r['veh_demand'], r['persons']) == (ref['veh_demand'], ref['persons'])
        groups, per_seed, slow = defaultdict(list), defaultdict(list), defaultdict(list)
        budget_exceedances, censored, events = [], Counter(), 0
        with gzip.open(SOURCE / f'results/eval/{name}.peds.csv.gz', 'rt', newline='') as f:
            for r in csv.DictReader(f):
                key = (r['scenario'], r['level'], r['crossing'])
                wait = int(r['wait'])
                groups[key].append(wait)
                per_seed[(*key, int(r['seed']))].append(wait)
                if bool_value(r['slow']):
                    slow[key].append(wait)
                if bool_value(r['censored']):
                    censored['|'.join(key[:2])] += 1
                budget = thresholds['budgets'][r['scenario']][r['level']][r['crossing']]
                if wait > budget:
                    if name not in ('fixed_time', 'ped_actuated'):
                        budget_exceedances.append({**r, 'budget': budget})
                events += 1
        q = [{k: quantile(v) for k, v in item.items()} for item in (groups, per_seed, slow)]
        if ref_groups is None:
            ref_groups = q
        result = dict(events=events, pooled_cells=len(groups), per_seed_cells=len(per_seed),
                      pooled_pass=sum(v <= ref_groups[0][k] + 1e-9 for k, v in q[0].items()),
                      per_seed_pass=sum(v <= ref_groups[1][k] + 1e-9 for k, v in q[1].items()),
                      slow_cells=len(slow), slow_pass=sum(v <= ref_groups[2][k] for k, v in q[2].items()),
                      worst_p95_ratio=max(v / ref_groups[0][k] for k, v in q[0].items()),
                      minimum_pooled_entries=min(map(len, groups.values())),
                      minimum_per_seed_entries=min(map(len, per_seed.values())),
                      censored=dict(censored), budget_exceedances=budget_exceedances)
        result['per_seed_failures'] = [dict(scenario=k[0], level=k[1], crossing=k[2], seed=k[3],
            controller_p95=v, reference_p95=ref_groups[1][k], delta_seconds=v-ref_groups[1][k],
            controller_entries=len(per_seed[k])) for k, v in sorted(q[1].items())
            if v > ref_groups[1][k] + 1e-9]
        result['per_seed_pass_percent'] = 100 * result['per_seed_pass'] / result['per_seed_cells']
        result['maximum_supplied_p95_difference'] = max(abs(v - float(supplied_p95[(name, *k)]['p95']))
                                                       for k, v in q[0].items())
        assert result['maximum_supplied_p95_difference'] < 1e-8
        pedestrian_stats[name] = result
        print(name, 'p95', result['pooled_pass'], '/', result['pooled_cells'], 'verified', flush=True)

    vehicle_summary, cells = {}, []
    for name in CONTROLLERS:
        records = episode_tables[name]
        values = list(records.values())
        vehicle_summary[name] = dict(episodes=len(values),
            teleport_events=sum(int(r['teleports']) for r in values),
            episodes_with_teleports=sum(int(r['teleports']) > 0 for r in values),
            unfinished_total=sum(int(r['veh_unfinished']) for r in values),
            shield_mask_restriction_fraction=st.mean(float(r['shield_rate']) for r in values))
        for sc in ('normal', 'slow'):
            for v in ('low', 'med', 'high'):
                for ped in ('low', 'med', 'high'):
                    lv = f'v{v}-p{ped}'
                    model = [records[sc, lv, seed] for seed in range(10, 20)]
                    ref = [episode_tables['ped_actuated'][sc, lv, seed] for seed in range(10, 20)]
                    means = lambda rows_, field: st.mean(float(r[field]) for r in rows_)
                    diffs = [float(m['veh_mean_time_loss']) - float(b['veh_mean_time_loss']) for m, b in zip(model, ref)]
                    rng = random.Random(20261104)
                    boot = [st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(2000)]
                    cell = dict(controller=name, scenario=sc, level=lv,
                        completed_vehicle_timeLoss_plus_departDelay=means(model, 'veh_mean_time_loss'),
                        reference=means(ref, 'veh_mean_time_loss'),
                        vehicle_change_percent=100 * (means(model, 'veh_mean_time_loss') / means(ref, 'veh_mean_time_loss') - 1),
                        paired_difference=st.mean(diffs),
                        paired_seed_bootstrap_95=[quantile(boot, .025), quantile(boot, .975)],
                        scheduled_mean=means(model, 'veh_demand'), completed_mean=means(model, 'veh_arrived'),
                        unfinished_mean=means(model, 'veh_unfinished'), reference_unfinished_mean=means(ref, 'veh_unfinished'),
                        teleport_events=sum(int(r['teleports']) for r in model), reference_teleports=sum(int(r['teleports']) for r in ref),
                        paired_vehicle_service_deteriorations=sum(int(m['veh_unfinished']) > int(b['veh_unfinished']) for m, b in zip(model, ref)),
                        shield_rate=means(model, 'shield_rate'))
                    assert abs(cell['completed_vehicle_timeLoss_plus_departDelay'] - float(supplied_vehicle[name, sc, lv]['veh_time_loss'])) < 1e-8
                    cells.append(cell)
    training = {}
    for name in CONTROLLERS[2:]:
        cfg = json.loads((SOURCE / f'runs/{name}/config.json').read_text())
        log = rows(SOURCE / f'runs/{name}/train_log.csv')
        assert [int(r['iteration']) for r in log] == list(range(1, cfg['iterations'] + 1))
        training[name] = dict(seed=cfg['seed'], iterations=cfg['iterations'], parallel_environments=cfg['n_envs'],
            episodes=cfg['iterations'] * cfg['n_envs'], seconds_per_episode=cfg['episode_seconds'])
    return dict(calibration_seeds=thresholds['seeds'], evaluation_seeds=list(range(10, 20)),
                calibration_budget_clamped=thresholds['clamped'],
                scenarios=2, demand_combinations=9, conditions=18,
                evaluations_per_controller=180, evaluations_total=1080, training=training,
                pedestrian=pedestrian_stats, vehicles=vehicle_summary, cells=cells,
                verification='Supplied pooled p95 and vehicle cell means independently reproduced; demand counts match. Route hashes and original simulator traces absent.',
                bootstrap_scope='Exploratory paired demand-seed intervals for one trained model; not training-seed uncertainty')


def unit_probes():
    # Inspected imports only; no policy pickle, training/evaluation entrypoint or SUMO start.
    sys.path.insert(0, str(SOURCE))
    from tsc.env import PedTracker
    from tsc.network import parse_network
    from tsc.config import SignalTiming
    from tsc.signals import JunctionSignal, EW, PED
    infos = parse_network(str(SOURCE / 'scenario/corridor.net.xml'))
    c = infos['J0'].crossings[0]
    wa = infos['J0'].walkingareas[0]

    class FakeSim:
        present = True
        speed = .2
        road = wa
        def __init__(self):
            self.edge = SimpleNamespace(getLastStepPersonIDs=lambda edge: ['p0'] if edge == wa and self.present else [])
            self.person = SimpleNamespace(getNextEdge=lambda pid: c.edge, getSpeed=lambda pid: self.speed,
                                          getRoadID=lambda pid: self.road)

    sim, tracker = FakeSim(), PedTracker(infos)
    tracker.update(sim)  # entrance reached at time 0
    for t in range(1, 10):
        sim.speed = 0. if t in (1, 2, 7, 8, 9) else .2
        tracker.update(sim)
    sim.present, sim.road = False, c.edge  # actually enters at time 10
    tracker.update(sim)
    assert len(tracker.samples) == 1 and tracker.samples[0].wait == 5
    probes = {'stop_go_at_entrance': {'elapsed_arrival_to_entry_seconds': 10,
                                     'recorded_stationary_seconds': tracker.samples[0].wait}}

    sim, tracker = FakeSim(), PedTracker(infos)
    sim.speed = 0
    tracker.update(sim)
    sim.present, sim.road = False, 'removed_without_crossing'
    tracker.update(sim)
    assert len(tracker.samples) == 1 and not tracker.samples[0].censored
    probes['disappearance_without_entry'] = {'recorded_as_uncensored': True,
                                            'crossing_entry_confirmed': False}
    sim, tracker = FakeSim(), PedTracker(infos)
    tracker.update(sim)
    tracker.finish()
    assert tracker.samples == []
    probes['moving_request_at_horizon'] = {'request_still_present': True, 'records_retained': 0}

    sig = JunctionSignal(infos['J0'], SignalTiming(), clearance=15, elapsed=5)
    sig.box_occupied = lambda: True
    sig.apply(PED)
    states = [sig.tick() for _ in range(15)]
    first_walk = states.index(sig.states['PED'])
    assert first_walk == 11
    probes['persistently_occupied_vehicle_box'] = {'first_WALK_after_seconds': first_walk,
                                                  'occupied_callback_still_true': sig.box_occupied()}
    sig = JunctionSignal(infos['J0'], SignalTiming(), clearance=15, phase=PED, target=PED, elapsed=10)
    sig.apply(EW)
    states = [sig.tick() for _ in range(16)]
    assert states[-1] == sig.states['EW']
    probes['clearance_timer_only'] = {'vehicle_green_after_seconds': 15,
        'pedestrian_occupancy_query_in_signal_interface': False,
        'scope': 'No claim of an observed collision; this demonstrates absence of an occupancy hold in the state machine.'}
    return probes


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True, parents=True)
    sha = verify_archive_files()
    result = raw_statistics()
    result['archive_sha256'] = sha
    result['unit_probes'] = unit_probes()
    result['tests'] = dict(source_verified=True, statistics_verified=True, code_probes_passed=5,
                           new_sumo_runs=0, checkpoints_loaded=False)
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    cells = result['cells']
    with (OUT / 'verified_vehicle_cells.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    failures = [dict(controller=c, **row) for c in CONTROLLERS[2:]
                for row in result['pedestrian'][c]['per_seed_failures']]
    with (OUT / 'verified_per_seed_failures.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(failures[0]))
        writer.writeheader()
        writer.writerows(failures)
    print('Training:', result['training'])
    print('Probes:', json.dumps(result['unit_probes']))
    print('Saved', OUT / 'audit.json')


if __name__ == '__main__':
    main()
