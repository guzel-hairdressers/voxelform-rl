"""Build a reproducible summary of workload ablations; never hide failed trials."""
import json,hashlib,statistics,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.workload_budget import ROOT

def main():
    dest=ROOT/'agent_notes/workload_budget_2026-11-09'
    read=lambda name:json.loads((ROOT/'results'/name).read_text())
    main=read('workload-budget-2026-11-09.json');cpu=read('workload-feedback-cpu-reserve-2026-11-09.json');gpu=read('workload-feedback-mps-reserve-2026-11-09.json');held=read('workload-feedback-cpu-heldout-2026-11-09.json')
    assert len(main['rows'])==288 and len(cpu['rows'])==36 and len(gpu['rows'])==36 and len(held['rows'])==12
    lines=['# Measurements','', 'Accepted masks, not voxel changes or rejected sampling retries. Success means the exact quantized FAR target. Mean steps below includes only successful episodes; compare success counts first.','', '| Variant | Initial success | Initial mean steps | Short-trained success | Trained mean steps |','|---|---:|---:|---:|---:|']
    summary={}
    for name in main['configs']:
        cells=[];summary[name]={}
        for stage in ('initial','trained'):
            xs=[x for x in main['rows'] if x['name']==name and x['stage']==stage];ok=[x for x in xs if x['success']]
            cells.extend([f'{len(ok)}/{len(xs)}',f'{statistics.mean(x["steps"] for x in ok):.1f}'])
            summary[name][stage]=dict(success=len(ok),trials=len(xs),meanSuccessfulSteps=statistics.mean(x['steps'] for x in ok))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    for name,data in [('Adaptive feedback, CPU',cpu),('Adaptive feedback, MPS',gpu),('Adaptive feedback, fourth seed CPU',held)]:
        ok=[x for x in data['rows'] if x['success']];lines.append(f'| {name} | {len(ok)}/{len(data["rows"])} | {statistics.mean(x["steps"] for x in ok):.1f} | Not trained | — |')
    lines.extend(['','Feedback scheduling uses the original random-initialized spatial model; the live preview separately preserves the saved spatial weights. These two observations distinguish scheduling gains from learning.','', '## Initial CPU comparisons by scene','', '| Scene | Operation | Original steps (successes/3) | Adaptive steps (successes/3) | Mean matched-FAR proxy-quality difference |','|---|---|---:|---:|---:|'])
    cases=['S','irregular','wide','L','XL','Barcelona'];fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for j,op in enumerate(('subtract','add')):
        base_means=[];new_means=[];labels=[]
        for tag in cases:
            a=[x for x in main['rows'] if x['name']=='baseline' and x['stage']=='initial' and x['case']==tag and x['operation']==op];b=[x for x in cpu['rows'] if x['case']==tag and x['operation']==op]
            aa=[x for x in a if x['success']];bb=[x for x in b if x['success']]
            pairs=[(x,y) for x in a for y in b if x['seed']==y['seed'] and x['success'] and y['success']]
            q=statistics.mean(y['quality']-x['quality'] for x,y in pairs)
            span=lambda xs:f'{min(x["steps"] for x in xs)}–{max(x["steps"] for x in xs)}'
            lines.append(f'| {tag} | {op} | {span(aa)} ({len(aa)}/3) | {span(bb)} ({len(bb)}/3) | {q:+.4f} |')
            base_means.append(statistics.mean(x['steps'] for x in aa));new_means.append(statistics.mean(x['steps'] for x in bb));labels.append(f'{tag} ({len(aa)}/3 → {len(bb)}/3)')
        ax=axes[j];ys=list(range(len(cases)));ax.barh([y-.18 for y in ys],base_means,height=.34,color='#a4aab2',label='Original probability area');ax.barh([y+.18 for y in ys],new_means,height=.34,color='#237e76',label='Adaptive workload')
        ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_xlabel('Mean accepted edits among exact-FAR successes');ax.set_title(op.capitalize());ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='lower right',fontsize=8);fig.suptitle('Fewer edits through workload scheduling · three initial-policy seeds');fig.tight_layout();fig.savefig(dest/'steps.png',dpi=170);fig.savefig(dest/'steps.svg');plt.close(fig)
    lines.extend(['', 'Proxy quality is uncalibrated. Differences are descriptive and use only pairs that both reached FAR. They do not establish architectural quality, daylight compliance or generalization.','', '## Final adaptive failures',''])
    for source,data in [('CPU calibration',cpu),('MPS',gpu),('CPU fourth seed',held)]:
        for x in data['rows']:
            if not x['success']:lines.append(f'- {source}: seed {x["seed"]}, {x["case"]} {x["operation"]}, {x["termination"]} after {x["steps"]} steps with {x["remaining"]}/{x["required"]} voxels remaining. Budget {x["budget"]["limit"]}.')
    lines.extend(['','## Training budgets','', '| Variant | Seed | Actual actions | Optimizer updates |','|---|---:|---:|---:|'])
    for x in main['training']:lines.append(f'| {x["name"]} | {x["seed"]} | {x["actions"]} | {x["updates"]} |')
    lines.extend(['','Training stops at an episode boundary after at least 768 accepted actions (actual counts above). The short runs use small synthetic shapes and FAR progress rewards; the live session also retains its environmental-audit reward. This is a trainability/short-run comparison, not a convergence study.','', '## Run provenance','', 'The main run was resumed after preserving every completed row and checkpoint, to reuse immutable scene/shadow setup. Its JSON records both source hash sets and the starting row count. Model/action semantics in the four main configurations are unchanged by the feedback-only follow-ups. Runtime measurements include concurrent test activity and are not an isolated hardware speed benchmark.'])
    (dest/'MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'results/workload-summary-2026-11-09.json').write_text(json.dumps(summary,indent=2)+'\n')
    source=dest/'source-final';source.mkdir(exist_ok=True)
    manifest={}
    for p in [*sorted((ROOT/'voxelform').glob('*.py')),*sorted((ROOT/'tests').glob('test_workload.py')),*sorted((ROOT/'tools').glob('*workload*.py')),ROOT/'public/app.js',ROOT/'public/index.html']:
        rel=str(p.relative_to(ROOT));manifest[rel]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,source/(rel.replace('/','__')+'.txt'))
    (dest/'final-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'MEASUREMENTS.md')

if __name__=='__main__':main()
