"""Static scientific plots from the completed, audited demand experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/demand-service-review'
SCENARIOS = ['vlow-plow', 'vlow-pmedium', 'vmedium-plow', 'vmedium-pmedium']
TITLES = ['Low vehicles / low pedestrians', 'Low vehicles / medium pedestrians',
          'Medium vehicles / low pedestrians', 'Medium vehicles / medium pedestrians']
NAMES = {'fixed': 'Fixed time', 'pressure': 'Pressure-inspired', 'mappo-original': 'Original MAPPO',
         'mappo-continue': 'Continued MAPPO', 'mappo-tail': 'Tail penalty',
         'mappo-guided': 'Penalty + guidance', 'mappo-early': 'Early-service hybrid', 'actuated': 'Actuated'}
COLORS = {'fixed': '#777777', 'pressure': '#009E73', 'mappo-original': '#56B4E9',
          'mappo-continue': '#0072B2', 'mappo-tail': '#E69F00', 'mappo-guided': '#8E44AD',
          'mappo-early': '#D55E00', 'actuated': '#222222'}


def save(fig, name):
    for extension in ['png', 'svg']:
        fig.savefig(DEST / f'{name}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    summary = json.loads((ROOT / 'runs/demand-service-test/summary.json').read_text())
    audit = json.loads((DEST / 'audit.json').read_text())
    assert audit['fingerprint'] == summary['protocol']['fingerprint']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    groups = list(NAMES)[:-1]
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True)
    for ax, scenario, title in zip(axs.flat, SCENARIOS, TITLES):
        data = summary['scenarios'][scenario]
        rows = {**data['baselines'], **data['learned']}
        for y, name in enumerate(groups):
            row = rows[name]
            c = row['versus_actuated'] if name in data['baselines'] else row['comparisons']['actuated']
            value = c['point']['vehicle_delta_seconds']
            lo, hi = c['exploratory_intervals']['vehicle_delta_seconds']
            ax.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt='o', color=COLORS[name], capsize=3, markersize=5)
        ax.set_yticks(range(len(groups)), [NAMES[g] for g in groups])
        ax.invert_yaxis()
        ax.axvline(0, color='#333333', linewidth=.9)
        ax.set_xlim(-21, 11)
        ax.set_xticks([-20, -15, -10, -5, 0, 5, 10])
        ax.grid(axis='x', alpha=.18)
        ax.set_title(title, fontsize=11)
    for ax in axs[-1]:
        ax.set_xlabel('Vehicle time loss minus actuated (seconds)\nNegative = vehicle gain; positive = vehicle cost')
    fig.suptitle('Earlier pedestrian service gives up the vehicle advantage', fontsize=15, y=.98)
    fig.text(.5, .025, 'Points: mean differences. Bars: exploratory paired 95% bootstrap intervals.\n'
             'Three trained replicas and ten demand seeds; vehicle loss is conditional on completion. Service counts are in the report.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.16, right=.98, bottom=.16, top=.90, wspace=.66, hspace=.35)
    save(fig, 'vehicle_changes')

    groups = ['mappo-continue', 'mappo-guided', 'mappo-early']
    crossings = sorted(summary['scenarios'][SCENARIOS[0]]['learned'][groups[0]]['absolute']['crossings'])
    fig, axs = plt.subplots(3, 1, figsize=(13.2, 8.5))
    for ax, name in zip(axs, groups):
        values = [[summary['scenarios'][s]['learned'][name]['comparisons']['actuated']['point']['crossing_p95_delta_seconds'][c]
                   for c in crossings] for s in SCENARIOS]
        im = ax.imshow(values, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-28, vcenter=0, vmax=28), aspect='auto')
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                ax.text(x, y, f'{value:+.1f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(value) >= 18 else '#111111')
        ax.set_yticks(range(4), ['Low / low', 'Low / medium', 'Medium / low', 'Medium / medium'])
        ax.set_xticks(range(12), [c.removeprefix(':').replace('_', '/') for c in crossings])
        ax.set_title(NAMES[name], fontsize=12, loc='left')
        ax.set_ylabel('Vehicles / pedestrians', fontsize=9)
        for x in [3.5, 7.5]:
            ax.axvline(x, color='white', linewidth=2)
    fig.suptitle('Crossing p95 wait minus the actuated reference (seconds)', fontsize=15, y=.99)
    fig.subplots_adjust(left=.17, right=.88, top=.92, bottom=.11, hspace=.48)
    cax = fig.add_axes([.90, .22, .018, .58])
    fig.colorbar(im, cax=cax, label='Positive = worse pedestrian tail')
    fig.text(.5, .025, 'Point estimates: pool ten demand seeds within each replica, then average three replica p95s.\n'
             'Small differences near zero are not statistical guarantees. All pooled crossing samples contain at least 146 entries.',
             ha='center', fontsize=9)
    save(fig, 'crossing_differences')

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.3), sharey=True)
    for ax, scenario, title in zip(axs.flat, SCENARIOS, TITLES):
        for x, name in enumerate(['actuated', 'mappo-continue', 'mappo-guided', 'mappo-early']):
            m = audit['mechanism'][scenario][name]
            ax.bar(x, m.get('measurement_empty_pedestrian_requests', 0) / m['runs'], color=COLORS[name], alpha=.32,
                   label='All requests made with no pedestrian waiting' if x == 0 else None)
            ax.bar(x, m.get('measurement_fully_observed_empty_admissions_zero_entries', 0) / m['runs'], color=COLORS[name], alpha=1,
                   label='Subset: completed admission had no entries' if x == 0 else None)
        ax.set_xticks(range(4), ['Actuated', 'Continue', 'Guided', 'Early service'])
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.018, .55, 'Mean requests per network measurement window', rotation=90,
             va='center', ha='center', fontsize=10)
    fig.suptitle('Empty pedestrian requests during the main measurement window', fontsize=14, y=.98)
    handles = [plt.Rectangle((0, 0), 1, 1, color='#777777', alpha=.32), plt.Rectangle((0, 0), 1, 1, color='#777777')]
    fig.legend(handles, ['All empty-at-request decisions', 'Subset with no entries during completed admission'],
               loc='lower center', bbox_to_anchor=(.5, .065), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .025, 'Requests in [300,2100) seconds. Zero-entry subset requires admission to end by 2100; late follow-up phases are excluded.\nAn empty request can serve later arrivals. These counts do not estimate avoidable vehicle delay.',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.08, right=.98, bottom=.19, top=.88, wspace=.15, hspace=.35)
    save(fig, 'empty_pedestrian_service')


if __name__ == '__main__':
    main()
