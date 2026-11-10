"""Standalone scientific figure; per-seed points, no significance claims."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/architecture-hypotheses-summary-2026-11-08.json').read_text());rows=r['variants'];names=list(rows)
fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained');colors={'2d':'#3978b4','3d':'#38927b','multiscale':'#9468ae'}
for ax,key,title in zip(axes,('syntheticGainBySeed','realGainBySeed'),('Held-out synthetic plots','One held-out cached Barcelona site')):
    for i,n in enumerate(names):
        vals=[v for v in rows[n][key] if v is not None];color=colors[n.split('_')[0]]
        ax.scatter(i+np.linspace(-.12,.12,len(vals)),vals,color=color,s=32,zorder=3)
        ax.hlines(np.mean(vals),i-.27,i+.27,color='#222222',linewidth=2,zorder=4)
    ax.axhline(0,color='#666666',linewidth=.8);ax.set_title(title,loc='left',fontsize=11);ax.set_ylabel('Quality after training − initial');ax.grid(axis='y',alpha=.2)
axes[-1].set_xticks(range(len(names)),[n.replace('_','\n') for n in names],fontsize=8)
fig.suptitle('Architecture ablation: 512 matched training actions per model\nDots: three seeds; black marks: mean. Quality uses common successful FAR-matched cases.',fontsize=12)
folder=ROOT/'agent_notes/architecture_hypotheses_2026-11-08';fig.savefig(folder/'learning-gains.png',dpi=180);fig.savefig(folder/'learning-gains.svg');plt.close(fig)
