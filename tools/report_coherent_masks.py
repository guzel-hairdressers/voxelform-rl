"""Rebuild descriptive tables and a standalone scientific figure from saved results."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'agent_notes/coherent_masks_2026-11-09'

def main():
    original=json.loads((ROOT/'results/coherent-masks-2026-11-09.json').read_text());sampled=json.loads((ROOT/'results/coherent-sampled-2026-11-09.json').read_text());gpu=json.loads((ROOT/'results/coherent-gpu-adaptive-2026-11-09.json').read_text());guarded=json.loads((ROOT/'results/coherent-guarded-2026-11-09.json').read_text())
    modes=['independent','smooth','correlated','combined'];summary={}
    lines=['| Eligibility | Sampler | FAR completions | Regions/mask | Cells/region | Boundary faces/cell | Synthetic quality | Barcelona quality |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for eligibility in ('facade','depth'):
        for mode in modes:
            ev=[e for x in sampled['cases'] if x['eligibility']==eligibility and x['mode']==mode for e in x['evaluation']];masks=[m['applied'] for e in ev for m in e['masks']]
            s=dict(successes=sum(v['success'] for v in ev),evaluations=len(ev),components=float(np.mean([v['components'] for v in masks])),patchSize=float(np.mean([v['meanPatchSize'] for v in masks])),boundary=float(np.mean([v['boundaryFacesPerVoxel'] for v in masks])),qualitySynthetic=float(np.mean([v['quality'] for v in ev if v['success'] and v['case']!='Barcelona'])),qualityBarcelona=float(np.mean([v['quality'] for v in ev if v['success'] and v['case']=='Barcelona'])))
            summary[eligibility+'_'+mode]=s
            lines.append(f"| {eligibility} | {mode} | {s['successes']}/{s['evaluations']} | {s['components']:.2f} | {s['patchSize']:.2f} | {s['boundary']:.2f} | {s['qualitySynthetic']:.4f} | {s['qualityBarcelona']:.4f} |")
    (FOLDER/'SAMPLED_RESULTS.md').write_text('\n'.join(lines)+'\n\nThese trajectories use min1/unlimited component guards to isolate sampling. Production additionally prunes patches below four cells and limits disconnected patches. Edit sizes vary during episodes; fixed-count probes are a separate comparison. Quality is an uncalibrated proxy, not architectural validation.\n')
    timing=['| Device | Grid | Decoder | Warm mask ms | Cold mask ms | Apply + depth ms | Failed draws /20 | Depth error m |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for c in gpu['cases']:
        if 'generation' in c:timing.append(f"| {c['device']} | {'×'.join(map(str,c['shape']))} | {'adaptive 1+7' if c['batched'] else 'sequential'} | {c['generation']['medianMs']:.2f} | {c['coldGenerationMs']:.2f} | {c.get('applyDepthMs',0):.2f} | {c['failedDraws']} | {c.get('depthMaxErrorMetres',float('nan')):.6f} |")
    (FOLDER/'TIMINGS.md').write_text('\n'.join(timing)+'\n\nFixed full-envelope snapshots; 20 warmed draws. MPS uses native tensor depth, CPU uses SciPy. Geometry safety remains on CPU. This is not browser-render timing; cold setup and backward/update times, memory and raw samples remain in JSON.\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');labels=['Independent','Smoothed\nprobabilities','Correlated','Correlated\n+ Gaussian'];colors=['#9a9a9a','#7e9cb2','#4b9b90','#a578b0']
    for ax,shape in zip(axes[0],[(4,48,64),(8,64,96)]):
        vals=[]
        for mode in modes:
            row=next(x for x in original['fixedCountProbes'] if tuple(x['shape'])==shape and x['mode']==mode);vals.append(float(np.mean([s['components'] for s in row['samples']])))
        bars=ax.bar(labels,vals,color=colors);ax.bar_label(bars,fmt='%.1f');ax.set_ylabel('Connected regions per mask');ax.set_title(f"Fixed count: {row['exactEditBudget']} cells, {shape}");ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
    for ax,key,title in zip(axes[1],['components','patchSize'],['Actual sampled episodes: regions per edit','Actual sampled episodes: cells per region']):
        vals=[summary['facade_'+m][key] for m in modes];bars=ax.bar(labels,vals,color=colors);ax.bar_label(bars,fmt='%.2f');ax.set_title(title);ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Mask coherence: sampler ablation\nTop: 30 fixed-count draws. Bottom: sampled trajectories before production size/count guarantees.',fontsize=12)
    fig.savefig(FOLDER/'coherence.png',dpi=170);fig.savefig(FOLDER/'coherence.svg');plt.close(fig)
    result=dict(sampled=summary,guardedWide=[dict(seed=x['seed'],success=x['success'],areas=x['interiorOpeningAreasByFloor'],fullHeightInteriorVoidColumns=x['fullHeightInteriorVoidColumns']) for x in guarded['wide']],guardedRealSuccesses=sum(x['success'] for x in guarded['real']),guardedRealEvaluations=len(guarded['real']),primaryErrors=original['failures'],sampledErrors=sampled['failures'],gpuErrors=gpu['failures'])
    (ROOT/'results/coherent-summary-2026-11-09.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2('/tmp/coherent-tests.log',FOLDER/'tests.txt')
    paths=[*sorted((ROOT/'voxelform').glob('*.py')),*sorted((ROOT/'tools').glob('*coherent*.py')),ROOT/'tests/test_patch_policy.py',ROOT/'tests/test_facade_depth.py',ROOT/'public/app.js',ROOT/'public/index.html']
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};(FOLDER/'final-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Rebuilt reports: sampled FAR completions',sum(s['successes'] for s in summary.values()),'/240')
if __name__=='__main__':main()
