"""Scientific figures from the reviewed saved summary; no new simulation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/pedestrian-finetune-cluster-review'


def main():
    s=json.loads((DEST/'archived_summary.json').read_text())
    medium=s['scenarios']['vmedium-pmedium'];all_rows={**medium['baselines'],**medium['learned']}
    labels={'actuated':'Actuated reference','fixed':'Fixed time','pressure':'Pressure-inspired',
            'mappo-original':'Starting MAPPO','mappo-continue':'Continued MAPPO',
            'mappo-tail':'Long-wait penalty','mappo-guided':'Penalty + guidance'}
    colors={'actuated':'#222222','fixed':'#777777','pressure':'#009E73',
            'mappo-original':'#0072B2','mappo-continue':'#D55E00','mappo-tail':'#E69F00','mappo-guided':'#8E44AD'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.8))
    names=['actuated','mappo-original','mappo-continue','mappo-tail','mappo-guided']
    crossings=sorted(all_rows['actuated']['absolute']['crossings'])
    for name in names:
        yy=[all_rows[name]['absolute']['crossings'][c]['mean_replica_pooled_p95'] for c in crossings]
        ax.plot(range(12),yy,marker='o',markersize=4,color=colors[name],label=labels[name],linewidth=2 if name in ['actuated','mappo-guided'] else 1.4,
                linestyle='--' if name in ['mappo-continue','mappo-tail'] else '-')
    ax.set_xticks(range(12),[c.removeprefix(':').replace('_',' / ') for c in crossings],rotation=40,ha='right')
    ax.set_ylabel('Pedestrian p95 wait (seconds)');ax.set_xlabel('Crossing')
    ax.set_title('Guidance reduces every crossing tail relative to the starting MAPPO\nEvery crossing still has a higher p95 than the actuated reference')
    ax.grid(axis='y',alpha=.2)
    for x in [3.5,7.5]:ax.axvline(x,color='#cccccc',linewidth=.7)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.32,top=.83,left=.09,right=.98)
    fig.text(.5,.018,'Medium demand; point estimates across 3 trained replicas and 4 paired demand seeds. Intervals are in the results tables.',ha='center',fontsize=8)
    for ext in ['png','svg']:fig.savefig(DEST/f'medium_crossing_p95.{ext}',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.3))
    offsets={'fixed':(8,3),'pressure':(-12,-12),'actuated':(-9,10),'mappo-original':(8,4),
             'mappo-continue':(-12,2),'mappo-tail':(8,-12),'mappo-guided':(8,2)}
    for name,row in all_rows.items():
        comp=row.get('versus_actuated',row.get('comparisons',{}).get('actuated'))
        x=row['absolute']['vehicle_mean_time_loss'];y=comp['point']['worst_crossing_delta_seconds']
        ax.scatter(x,y,color=colors[name],s=65,zorder=3)
        ax.annotate(labels[name],(x,y),xytext=offsets[name],textcoords='offset points',ha='right' if name in ['actuated','pressure','mappo-continue'] else 'left')
    ax.axhline(0,color='#999999',linewidth=.8)
    ax.set_xlim(30,57);ax.set_ylim(-4,47)
    ax.set_xlabel('Completed-vehicle time loss (seconds; lower is better)')
    ax.set_ylabel('Largest crossing p95 increase over actuated (seconds)')
    ax.set_title('Medium-demand vehicle and pedestrian trade-off\nThree training replicas; four paired demand seeds; point estimates')
    ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(DEST/f'medium_tradeoff.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
