"""Two matched SUMO runs prove observation augmentation leaves physics unchanged."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.local/sumo-1.27.1/tools'))

from src.traffic_control.fairness.environment import run
from src.traffic_control.fairness.control_observation import FEATURES


def main():
    folder = ROOT / 'runs/deep-audit'
    folder.mkdir(parents=True, exist_ok=True)
    cfg = dict(end_seconds=180, warmup_seconds=0, measurement_end_seconds=150,
               demand_end_seconds=120)
    results, traces = [], []
    for schema in ('legacy-v2', 'controller-memory-v1'):
        trace = []
        with tempfile.TemporaryDirectory(prefix='memory-check-', dir=folder) as scratch:
            metrics = run('actuated', 920201, scratch,
                          overrides={**cfg, 'observation_schema': schema}, trace_sink=trace)
        results.append(metrics)
        traces.append(trace)
    a, b = results
    keys = ('vehicles_end', 'vehicles_at_measurement_cutoff', 'vehicles_measurement_cohort',
            'pedestrians_by_crossing_end', 'persons', 'phase_history', 'demand_sha256', 'network_sha256')
    for key in keys:
        if a[key] != b[key]:
            raise AssertionError('Observation addition changed baseline outcome: ' + key)
    if len(traces[0]) != len(traces[1]):
        raise AssertionError('Decision sample count changed')
    targets = timers = 0
    for old, new in zip(*traces):
        tls = old['junction']
        before = dict(zip(a['observation_features'][tls], old['observation']))
        after = dict(zip(b['observation_features'][tls], new['observation']))
        if {k: v for k, v in after.items() if k not in FEATURES} != before:
            raise AssertionError('Legacy observation values changed')
        if any(old[k] != new[k] for k in ('time', 'junction', 'action', 'mask')):
            raise AssertionError('Decision changed')
        targets += any(after['committed_' + p] for p in ('EW', 'NS', 'PED'))
        timers += bool(after['clearance_empty_known'])
    if not targets or not timers:
        raise AssertionError('Fixture did not exercise controller memory')
    for result in results:
        for key in ('vehicles_end', 'vehicles_at_measurement_cutoff', 'vehicles_measurement_cohort'):
            v = result[key]
            if v['completed_depart_delay_observed_count'] != v['completed_count']:
                raise AssertionError('SUMO did not report every completed insertion delay')
    evidence = dict(seed=920201, runs=2, matched_fields=list(keys), decision_rows=len(traces[0]),
                    legacy_dimensions=len(a['observation_features']['C0']),
                    augmented_dimensions=len(b['observation_features']['C0']),
                    target_visible_rows=targets, empty_timer_visible_rows=timers,
                    completed_mean_depart_delay_seconds=b['vehicles_end']['completed_mean_depart_delay_seconds'],
                    outcome='passed', note='Engineering equivalence only; no RL training or performance claim')
    (folder / 'control-memory-verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
