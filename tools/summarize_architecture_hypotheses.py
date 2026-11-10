"""Paired descriptive analysis; seeds, not scenes, are the replication unit."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def mean(v):return float(np.mean(v)) if v else None
def main():
    raw=ROOT/'results/architecture-hypotheses-2026-11-08.json';r=json.loads(raw.read_text());rows=r['learning'];summary={}
    def outcomes(row,stage):return {(x['case'],x['operation']):x for x in row[stage]}
    def gain(row,real):
        a=outcomes(row,'initial');b=outcomes(row,'trained')
        return [v['quality']-a[k]['quality'] for k,v in b.items() if ('cached' in k[0])==real and v['success'] and a[k]['success']]
    for n in dict.fromkeys(x['variant'] for x in rows):
        group=[x for x in rows if x['variant']==n and 'error' not in x];ev=[v for a in group for v in a['trained']]
        summary[n]=dict(seeds=len(group),successes=sum(v['success'] for v in ev),evaluations=len(ev),
                        subtractSuccesses=sum(v['success'] for v in ev if v['operation']=='subtract'),addSuccesses=sum(v['success'] for v in ev if v['operation']=='add'),
                        syntheticGainBySeed=[mean(gain(a,False)) for a in group],realGainBySeed=[mean(gain(a,True)) for a in group],
                        syntheticGainPairsBySeed=[len(gain(a,False)) for a in group],realGainPairsBySeed=[len(gain(a,True)) for a in group],
                        syntheticQuality=mean([v['quality'] for v in ev if v['success'] and 'cached' not in v['case']]),
                        realQuality=mean([v['quality'] for v in ev if v['success'] and 'cached' in v['case']]),
                        trainingSeconds=[a['training']['seconds'] for a in group],actualUpdates=[len(a['training']['updates']) for a in group],
                        meanProjectionRetention=mean([v['meanRetained'] for v in ev if v['meanRetained'] is not None]),
                        rejectedTrainingEpisodes=sum(v['rejected'] for a in group for v in a['training']['episodes']))
    pairs={}
    tests=[('single_vs_dual','single',False,True),('explicit_facade','facade',False,True),('3d_vs_2d','encoder','2d','3d'),('multiscale_vs_local3d','encoder','3d','multiscale')]
    for title,factor,a,b in tests:
        deltas={False:{},True:{}};gain_deltas={False:{},True:{}};success_delta=[];latency=[]
        for x in rows:
            if x['config'][factor]!=a:continue
            matches=[y for y in rows if y['seed']==x['seed'] and y['config'][factor]==b and all(y['config'][f]==x['config'][f] for f in x['config'] if f!=factor)]
            for y in matches:
                xo,yo=outcomes(x,'trained'),outcomes(y,'trained');xi,yi=outcomes(x,'initial'),outcomes(y,'initial')
                success_delta.append(sum(v['success'] for v in yo.values())-sum(v['success'] for v in xo.values()))
                for k,v in xo.items():
                    z=yo[k];real='cached' in k[0]
                    if v['success'] and z['success']:
                        deltas[real].setdefault(x['seed'],[]).append(z['quality']-v['quality'])
                        if xi[k]['success'] and yi[k]['success']:
                            gain_deltas[real].setdefault(x['seed'],[]).append((z['quality']-yi[k]['quality'])-(v['quality']-xi[k]['quality']))
        pairs[title]=dict(syntheticQualityDeltaBySeed={k:mean(v) for k,v in deltas[False].items()},realQualityDeltaBySeed={k:mean(v) for k,v in deltas[True].items()},
                          syntheticLearningGainDeltaBySeed={k:mean(v) for k,v in gain_deltas[False].items()},realLearningGainDeltaBySeed={k:mean(v) for k,v in gain_deltas[True].items()},
                          totalSuccessCountDelta=sum(success_delta),syntheticPairsBySeed={k:len(v) for k,v in deltas[False].items()},realPairsBySeed={k:len(v) for k,v in deltas[True].items()},note='Matched other factors and seed; quality only on common successful cases, conditional on completion.')
    interactions={}
    for enc in ('2d','3d','multiscale'):
        interactions[enc]={}
        for seed in r['seeds']:
            table={(x['config']['single'],x['config']['facade']):outcomes(x,'trained') for x in rows if x['config']['encoder']==enc and x['seed']==seed}
            if len(table)!=4:continue
            ds=[]
            for k in table[False,False]:
                if all(t[k]['success'] for t in table.values()):ds.append(table[True,True][k]['quality']-table[True,False][k]['quality']-table[False,True][k]['quality']+table[False,False][k]['quality'])
            interactions[enc][seed]=mean(ds)
    control_file=ROOT/'results/architecture-controls-replicated-2026-11-08.json'
    control_rows=json.loads(control_file.read_text())['controls'] if control_file.exists() else r['controls']
    controls={}
    for kind in ('uniform','sky_greedy'):
        group=[x for x in control_rows if x['kind']==kind];ev=[v for x in group for v in x['evaluation']]
        controls[kind]=dict(successes=sum(v['success'] for v in ev),evaluations=len(ev),
            syntheticQuality=mean([v['quality'] for v in ev if v['success'] and 'cached' not in v['case']]),
            realQuality=mean([v['quality'] for v in ev if v['success'] and 'cached' in v['case']]),
            syntheticQualityBySeed=[mean([v['quality'] for v in x['evaluation'] if v['success'] and 'cached' not in v['case']]) for x in group],
            realQualityBySeed=[mean([v['quality'] for v in x['evaluation'] if v['success'] and 'cached' in v['case']]) for x in group])
    result=dict(variants=summary,pairedEffects=pairs,singleFacadeInteraction=interactions,controls=controls,failedExperiments=r['failures'])
    (ROOT/'results/architecture-hypotheses-summary-2026-11-08.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['| Configuration | FAR reached | Synthetic Δ quality | Barcelona Δ quality | CPU L / XL mask ms | MPS L / XL mask ms |','| --- | ---: | ---: | ---: | ---: | ---: |']
    def fmt(v):return '—' if v is None else f'{v:+.4f}'
    for n,s in summary.items():
        times=[]
        for device in ('cpu','mps'):
            vals=[]
            for scale in ('L','XL'):
                c=next((c for c in r['benchmarks'] if c['variant']==n and c['device']==device and c['scale']==scale),{})
                vals.append(f"{c['timings']['generation']['medianMs']:.2f}" if 'timings' in c else 'failed' if c else '—')
            times.append(' / '.join(vals))
        lines.append(f"| {n} | {s['successes']}/{s['evaluations']} | {fmt(mean([v for v in s['syntheticGainBySeed'] if v is not None]))} | {fmt(mean([v for v in s['realGainBySeed'] if v is not None]))} | {times[0]} | {times[1]} |")
    (ROOT/'agent_notes/architecture_hypotheses_2026-11-08/TABLE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(pairedEffects=pairs,controls=controls,interactions=interactions),indent=2))
if __name__=='__main__':main()
