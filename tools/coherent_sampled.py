"""Actual generative inference, contrasted with the preserved greedy-first diagnostic."""
import json,gc,functools,traceback
from pathlib import Path
import numpy as np,torch
from voxelform.patch_policy import PatchConfig,PatchLearner,PatchFacades,MODES
from voxelform.environment import Environment
from voxelform.context import ContextLibrary
from tools.coherent_masks import episode,ROOT

class SamplingPolicy(PatchLearner):
    def choose(self,env,training=True,greedy=False):return super().choose(env,training=training,greedy=False)

def main():
    torch.set_num_threads(1);rows=[];wide=[];failures=[];out=ROOT/'results/coherent-sampled-2026-11-09.json'
    context=ContextLibrary().load('m_bcn_0093')
    def save():out.write_text(json.dumps(dict(note='Posthoc actual sampled inference. Same 256-action checkpoints as original greedy-first diagnostic; no retraining. Every mode draws from its distribution; three paired seeds. Guarded cases test min4/max4 separately.',cases=rows,wide=wide,failures=failures),indent=2)+'\n')
    for elig in ('facade','depth'):
        for mode in MODES:
            cfg=PatchConfig(mode=mode,eligibility=elig);factory=functools.partial(PatchFacades,config=cfg);cases=[]
            for i,plan in enumerate(('rectangle','triangle','courtyard','branch')):
                e=Environment((4,16,20),seed=9100+i,plan=plan,sun_count=4,frontier_factory=factory);cases.extend([(plan,e,'subtract'),(plan,e,'add')])
            real=Environment(context=context,seed=9200,sun_count=8,frontier_factory=factory);cases.extend([('Barcelona',real,'subtract'),('Barcelona',real,'add')])
            for seed in (123,456,789):
                row=dict(eligibility=elig,mode=mode,seed=seed)
                try:
                    p=SamplingPolicy();p.load(torch.load(ROOT/f'results/checkpoints/coherent-masks-2026-11-09/{elig}-{mode}-{seed}.pt',weights_only=True))
                    row['evaluation']=[dict(case=tag,**episode(p,e,op,seed+10000+i)) for i,(tag,e,op) in enumerate(cases)]
                    print('sampled',elig,mode,seed,sum(v['success'] for v in row['evaluation']),'/10',flush=True)
                except Exception:row['error']=traceback.format_exc();failures.append(row['error']);print(row['error'],flush=True)
                rows.append(row);save()
            del cases,real;gc.collect()
    for guarded in (False,True):
        for elig in ('facade','depth'):
            for seed in (123,456,789):
                row=dict(eligibility=elig,mode='combined',seed=seed,guarded=guarded)
                try:
                    p=SamplingPolicy();p.load(torch.load(ROOT/f'results/checkpoints/coherent-masks-2026-11-09/{elig}-combined-{seed}.pt',weights_only=True))
                    p.config=PatchConfig(mode='combined',eligibility=elig,minimum_patch=4 if guarded else 1,maximum_patches=4 if guarded else 0)
                    e=Environment((4,40,48),seed=9300,sun_count=8,frontier_factory=functools.partial(PatchFacades,config=p.config));d=e.depth.cpu().numpy().copy()
                    row.update(episode(p,e,'subtract',seed+50000));row['fullHeightInteriorVoidColumns']=int(((~e.occupied).cpu().numpy().all(0)&(d[0]>8)).sum())
                    print('sampled wide',elig,seed,guarded,row['success'],sum(m['interiorVoxels'] for m in row['masks']),'interior',flush=True)
                except Exception:row['error']=traceback.format_exc();failures.append(row['error'])
                wide.append(row);save()
    # Live default's two operations also checked on the cached real scene.
    guarded=[]
    for seed in (123,456,789):
        p=SamplingPolicy();p.load(torch.load(ROOT/f'results/checkpoints/coherent-masks-2026-11-09/facade-combined-{seed}.pt',weights_only=True));p.config=PatchConfig(minimum_patch=4,maximum_patches=4)
        e=Environment(context=context,sun_count=8,seed=9200,frontier_factory=functools.partial(PatchFacades,config=p.config))
        for op in ('subtract','add'):guarded.append(dict(seed=seed,**episode(p,e,op,seed+70000)))
        save()
    data=json.loads(out.read_text());data['guardedReal']=guarded;out.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
