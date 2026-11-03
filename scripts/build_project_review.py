"""Build a read-only data index from existing results; run no simulations."""
from pathlib import Path
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.traffic_control.fairness.demand import scenarios


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def link(path, label=None):
    return f"[{label or path}]({(ROOT / path).as_posix()})"


def number(value):
    return "unavailable" if value is None else f"{value:.2f}"


def interval(metric):
    return f"{number(metric['mean'])} [{number(metric['ci95_low'])}, {number(metric['ci95_high'])}]"


def absolute_tables(lines, summary, order=None):
    for scenario, controllers in summary.items():
        lines += [f"### {scenario}", "", "| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label in order or controllers:
            if label not in controllers:
                continue
            row = controllers[label]
            values = [interval(row['vehicle_time_loss']), interval(row['pedestrian_mean_wait'])]
            values += [number(row.get(k, {}).get('mean')) for k in ('scheduled', 'completed', 'unfinished', 'uninserted', 'unserved_requests')]
            lines.append('| ' + ' | '.join([label, *values]) + ' |')
        lines.append("")


def main():
    destination = ROOT / 'results/project-review'
    destination.mkdir(parents=True, exist_ok=True)
    protocol = read('runs/cluster-final-review/protocol.json')
    primary = read('runs/cluster-final-review/primary_inference.json')
    old = read('results/final-study/summary.json')
    initial = read('results/summary.json')
    singapore = read('results/singapore-transfer/summary.json')
    report = (ROOT / 'runs/cluster-final-review/REPORT.md').read_text()
    comparisons = {}
    for scenario, controller, difference, failed in re.findall(r'^\| ([^|]+) \| ([^|]+) \| (-?[\d.]+) \| (\d+) \|$', report, re.M):
        comparisons[(scenario, controller)] = (float(difference), int(failed))
    assert len(comparisons) == len(protocol['scenarios']) * 6
    source_paths = ['src/traffic_control/fairness/environment.py', 'src/traffic_control/connected.py', 'src/traffic_control/fairness/demand.py', 'src/traffic_control/fairness/signals.py', 'config/fairness_v2.json']
    source_matches = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == protocol['source_sha256'][p] for p in source_paths}
    assert all(source_matches.values()), 'Local environment definitions differ from the cluster protocol'
    net = ET.parse(ROOT / 'runs/3j-fixed-seed38/connected.net.xml').getroot()
    signals = sorted({x.attrib['id'] for x in net.findall('tlLogic')})
    crossing_count = sum(e.get('function') == 'crossing' for e in net.findall('edge'))
    assert signals == ['C0', 'C1', 'C2'] and crossing_count == 12
    lines = ['# Project data review - 2 November 2026', '',
        'This index reads saved results. It does not rerun experiments or combine different study versions.', '',
        '## What is available', '',
        '| Study | Setting | Available evidence |', '|---|---|---|',
        '| Single-junction milestone | One signal, four crossings | Local raw network, demand, events and metrics |',
        '| Initial exploratory study | Three signals, ten demand/speed scenarios | Complete local summaries; one trained model per learned configuration |',
        '| Earlier repeated-training study | Three signals, 13 scenarios; linear hold/change policies; five training seeds, ten evaluation seeds | Complete local absolute results and event tables |',
        '| Fairness v2 cluster study | Three signals, 21 scenarios; neural phase-selection policies; five training seeds, twenty evaluation seeds | Learned-controller vehicle differences for all scenarios and primary crossing differences. Complete held-out baseline/raw evaluation records are still on the cluster |',
        '| NUS-derived transfer | Three synthetic junctions positioned using approximate map-derived centroids; synthetic demand | Complete local earlier-controller transfer results; not a calibrated Singapore network |',
        '| Service-priority follow-up | Same physical v2 environment; short learning smoke and two full-length conventional development seeds | Engineering checks and pilot results only |', '',
        'The latest cluster run has 21 x 20 = 420 evaluation groups. Each group contains three conventional controllers and six learned configurations x five replicas = 33 instances, or 13,860 episodes. The four original learned configurations remain the core comparison; the two constrained configurations are extensions.', '',
        'The downloaded comparisons.json is a truncated tail, not valid complete JSON. Do not reconstruct missing absolute results from it. The full REPORT.md, protocol.json and primary_inference.json are readable. No latest held-out absolute baseline values are invented here.', '',
        '## Junction audit', '',
        f"Saved corridor network: signals {', '.join(signals)}; {crossing_count} crossing edges. The cluster protocol independently records actor observations for C0, C1 and C2. Four crossings per junction is not four junctions.", '',
        'The five environment/demand/signal/configuration source hashes checked here exactly match the cluster protocol. This verifies which layout and rules the pipeline specified; it is not a fresh end-to-end audit of every cluster trajectory.', '',
        '## Latest study: primary crossing result', '',
        'A crossing is descriptively no worse when its pooled p95 point difference is <= 0. This is a secondary count, not proof of statistical non-inferiority. Missing/insufficient events must remain unevaluable, never successful zero waits. The original all-crossings target and service counts remain primary.', '',
        '| Configuration | Vehicle difference vs actuated, s | Crossings no worse / 12 | Largest p95 difference, s | Minimum reported pooled crossing entries | Service/measurement gate failed runs / 100 |', '|---|---:|---:|---:|---:|---:|']
    for label, row in primary.items():
        point = row['point_estimates']
        delta, failed = comparisons[('vmedium-pmedium', label)]
        count = f"{sum(x <= 0 for x in point[1])}/12" if point else 'unevaluable from available summary'
        worst = number(max(point[1])) if point else 'unavailable'
        lines.append(f"| {label} | {delta:+.3f} | {count} | {worst} | {row['minimum_crossing_entries']} | {failed} |")
    lines += ['', 'Service/measurement flags combine several checks. They are not collision counts, and their summed score is not a count of unique unserved people.', '',
        '### Every measured primary crossing', '', '| Crossing | IPPO pedestrian p95 difference, s | MAPPO pedestrian p95 difference, s |', '|---|---:|---:|']
    for c, a, b in zip(primary['ippo-pedestrian']['crossing_order'], primary['ippo-pedestrian']['point_estimates'][1], primary['mappo-pedestrian']['point_estimates'][1]):
        lines.append(f'| {c} | {a:+.2f} | {b:+.2f} |')
    lines += ['', 'Positive means a longer wait than actuated. Differences are point estimates, not confidence intervals. Primary crossing data: ' + link('runs/cluster-final-review/primary_inference.json'), '',
        '## Latest study: all 21 scenarios', '',
        'Rates below are synthetic expected arrivals per hour across the network, not per junction or lane. The default demand interval is 0-2100 s; run end is 2400 s. Crossing-arrival measurement cohort: 300-2100 s, with follow-up to 2400 s. Baseline vehicle/pedestrian labels are scenario assumptions, not field calibration or guaranteed saturation levels.', '',
        '| Scenario | Vehicles/hour | People/hour | Additional change |', '|---|---:|---:|---|']
    for name, cfg in scenarios().items():
        extra = {k:v for k,v in cfg.items() if k not in ('vehicle_rate_per_hour','pedestrian_rate_per_hour')}
        lines.append(f"| {name} | {cfg['vehicle_rate_per_hour']} | {cfg['pedestrian_rate_per_hour']} | {json.dumps(extra) if extra else 'none'} |")
    labels = ['ippo-vehicle','ippo-pedestrian','mappo-vehicle','mappo-pedestrian']
    lines += ['', '### Vehicle time-loss differences vs actuated (seconds)', '', '| Scenario | IPPO vehicle | IPPO pedestrian | MAPPO vehicle | MAPPO pedestrian |', '|---|---:|---:|---:|---:|']
    for scenario in protocol['scenarios']:
        lines.append('| ' + ' | '.join([scenario, *[f'{comparisons[(scenario,label)][0]:+.3f}' for label in labels]]) + ' |')
    lines += ['', 'Read the differences with the service failures below. A negative completed-trip mean does not establish success when traffic remains unserved.', '',
        '### Paired runs failing the service/measurement gate (out of 100)', '', '| Scenario | IPPO vehicle | IPPO pedestrian | MAPPO vehicle | MAPPO pedestrian |', '|---|---:|---:|---:|---:|']
    for scenario in protocol['scenarios']:
        lines.append('| ' + ' | '.join([scenario, *[str(comparisons[(scenario,label)][1]) for label in labels]]) + ' |')
    lines += ['', 'Full six-configuration comparison: ' + link('runs/cluster-final-review/REPORT.md'), '',
        '## Earlier repeated-training study: absolute results', '',
        'Historical results, NOT the latest cluster study. Five independent training replicas, ten demand seeds, 150 training episodes; linear hold/change controllers. Pedestrian weights 0, 3 and 10. Means and existing 95% bootstrap intervals are reproduced from the original summary; no new uncertainty calculation is performed here. Counts are mean counts per run.', '',
        'Source: ' + link('results/final-study/summary.json'), '']
    absolute_tables(lines, old['summary'], old['controllers'])
    lines += ['### Earlier primary crossing counts', '', '| Controller | Crossings no worse / 12 (point estimates) | Original all-crossings descriptive target |', '|---|---:|---|']
    for c, row in old['primary_crossing_criterion'].items():
        adequate = row['maximum_crossing_p95_difference']['all_crossings_measured']
        count = str(12 - len(row['crossings_with_higher_p95'])) if adequate else 'unevaluable'
        lines.append(f"| {c} | {count}/12 | {row['descriptive_criterion_met']} |")
    lines += ['', 'Actuated is the reference: 12/12 by identity, not an improvement result.', '',
        '## NUS-derived transfer: earlier controllers', '', *['- ' + x for x in singapore['limits']], '',
        'The raw OSM import contains 20 traffic-light program IDs; these may include separate signal nodes within a physical junction. That import is separate from the three-agent synthetic derived evaluation network. It is not evidence that the main experiment used twenty or four learning junctions.', '',
        'Source: ' + link('results/singapore-transfer/summary.json'), '']
    absolute_tables(lines, {singapore['scenario']:singapore['summary']})
    lines += ['## Initial exploratory study: historical only', '',
        'One trained model per learned configuration and five evaluation seeds; sparse pedestrian samples. Keep separate from the later repeated-training and v2 studies. Scheduled counts were not included in this summary and remain unavailable here.', '',
        'Source: ' + link('results/summary.json'), '']
    absolute_tables(lines, initial['summary'])
    lines += ['## Raw data and development checks', '']
    for p in ['runs/fixed-seed38/metrics.json','runs/actuated-seed38/metrics.json','runs/3j-fixed-seed38/metrics.json','runs/3j-fixed-seed38/connected.net.xml','results/final-study/runs.csv','results/final-study/crossing_events.csv','results/final-study/crossing_comparison.csv','results/final-study/phase_history.csv','results/measurement-resolution','results/singapore-transfer','results/service-priority/LOCAL_PILOT.md','runs/service-priority-medium-pilot/paired_diagnostics.json']:
        if (ROOT/p).exists(): lines.append('- ' + link(p))
    (destination/'DATA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest = {'source_matches_cluster':source_matches,'signals':signals,'crossings':crossing_count,
        'cluster_protocol_fingerprint':protocol['fingerprint'],'comparison_rows':len(comparisons),
        'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['runs/cluster-final-review/REPORT.md','runs/cluster-final-review/primary_inference.json','runs/cluster-final-review/protocol.json','results/final-study/summary.json','results/summary.json','results/singapore-transfer/summary.json']}}
    (destination/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    values = np.array([primary[label]['point_estimates'][1] for label in labels if label.endswith('pedestrian')])
    fig, ax = plt.subplots(figsize=(11.5,3.3),layout='constrained')
    ax.imshow(values,cmap='RdBu_r',vmin=-1.25,vmax=1.25,aspect='auto')
    ax.set_xticks(range(12),[c.replace(':','').replace('_',' / ') for c in primary['ippo-pedestrian']['crossing_order']],rotation=45,ha='right')
    ax.set_yticks([0,1],['IPPO pedestrian\n3 of 12 no worse','MAPPO pedestrian\n3 of 12 no worse'])
    for i in range(2):
        for j in range(12): ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',color='white' if abs(values[i,j])>.8 else '#17212b')
    ax.set_title('Primary medium-demand scenario: crossing p95 difference vs actuated (seconds)',loc='left',pad=14)
    fig.supxlabel('Blue: shorter wait. Red: longer wait. Point estimates only; no uncertainty intervals shown.',fontsize=10)
    fig.savefig(destination/'crossing_differences.png',dpi=180)
    fig.savefig(destination/'crossing_differences.svg')
    plt.close(fig)
    print(f'Wrote {destination / "DATA.md"}; {len(comparisons)} cluster comparisons; three signals verified.')


if __name__ == '__main__':
    main()
