"""Held-out broad plate exercises deep eligibility absent from small learning plots."""
import json,functools,time,hashlib,traceback
from pathlib import Path
import numpy as np,torch
from voxelform.patch_policy import PatchConfig,PatchLearner,PatchFacades,MODES
from voxelform.environment import Environment
from tools.coherent_masks import episode,ROOT

def main():
    torch.set_num_threads(1);rows=[];failures=[];out=ROOT/'results/coherent-wide-2026-11-09.json'
    for eligibility in ('facade','depth'):
        for mode in MODES:
            for seed in (123,456,789):
                row=dict(eligibility=eligibility,mode=mode,seed=seed,guarded=False)
                try:
                    p=PatchLearner();p.load(torch.load(ROOT/f'results/checkpoints/coherent-masks-2026-11-09/{eligibility}-{mode}-{seed}.pt',weights_only=True))
                    e=Environment((4,40,48),seed=9300,sun_count=8,frontier_factory=functools.partial(PatchFacades,config=p.config))
                    initial_depth=e.frontier.depth_numpy.copy();row['initialMaxDepth']=float(initial_depth.max())
                    result=episode(p,e,'subtract',seed+50000);row.update(result)
                    row['fullHeightInteriorVoidColumns']=int(((~e.occupied).cpu().numpy().all(0)&(initial_depth[0]>8)).sum())
                    print(eligibility,mode,seed,row['success'],sum(m['interiorVoxels'] for m in row['masks']),'interior edits',flush=True)
                except Exception:row['error']=traceback.format_exc();failures.append(row['error']);print(row['error'],flush=True)
                rows.append(row);out.write_text(json.dumps(dict(note='Posthoc wide held-out subtraction plate: 4x40x48, max centre depth19.5m, eight sun samples,16 context boxes. Same previously trained checkpoints; no wide-scene fine tuning.',cases=rows,failures=failures),indent=2))
        for seed in (123,456,789):
            row=dict(eligibility=eligibility,mode='combined',seed=seed,guarded=True)
            try:
                p=PatchLearner();p.load(torch.load(ROOT/f'results/checkpoints/coherent-masks-2026-11-09/{eligibility}-combined-{seed}.pt',weights_only=True));p.config=PatchConfig(mode='combined',eligibility=eligibility,minimum_patch=4,maximum_patches=4)
                e=Environment((4,40,48),seed=9300,sun_count=8,frontier_factory=functools.partial(PatchFacades,config=p.config));d=e.frontier.depth_numpy.copy();row.update(episode(p,e,'subtract',seed+50000));row['fullHeightInteriorVoidColumns']=int(((~e.occupied).cpu().numpy().all(0)&(d[0]>8)).sum());print('guarded',eligibility,seed,row['success'],flush=True)
            except Exception:row['error']=traceback.format_exc();failures.append(row['error'])
            rows.append(row);out.write_text(json.dumps(dict(note='Wide held-out subtraction plate. Guarded extra cases test minimum4/max4 patches separately from sampler ablation.',cases=rows,failures=failures),indent=2))
if __name__=='__main__':main()
