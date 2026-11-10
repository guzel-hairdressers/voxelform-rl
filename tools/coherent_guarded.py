"""Production guard follow-up: minimum footprint and per-floor void count."""
import json,functools,time,gc,hashlib,traceback,argparse
import numpy as np,torch
from voxelform.patch_policy import PatchConfig,PatchFacades,components
from voxelform.environment import Environment
from voxelform.context import ContextLibrary
from tools.coherent_sampled import SamplingPolicy
from tools.coherent_masks import ROOT,episode
from tools.architecture_hypotheses import timed,summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);path=ROOT/'results/coherent-guarded-2026-11-09.json';r=dict(config=dict(minimumPatch=4,maximumPatches=4,maximumVoids=1,minimumVoidArea=4),wide=[],real=[],benchmark=[],failures=[],sourceHashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['voxelform/patch_policy.py','voxelform/environment.py']})
    def save():path.write_text(json.dumps(r,indent=2)+'\n')
    if args.benchmark_only:path=ROOT/'results/coherent-guarded-fastpath-2026-11-09.json'
    cfg=PatchConfig(mode='combined',eligibility='depth',minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4)
    context=ContextLibrary().load('m_bcn_0093')
    for seed in (() if args.benchmark_only else (123,456,789)):
        p=SamplingPolicy();p.load(torch.load(ROOT/f'results/checkpoints/coherent-masks-2026-11-09/depth-combined-{seed}.pt',weights_only=True));p.config=cfg
        e=Environment((4,40,48),sun_count=8,seed=9300,frontier_factory=functools.partial(PatchFacades,config=cfg));d=e.depth.cpu().numpy().copy()
        row=dict(seed=seed,**episode(p,e,'subtract',seed+50000));row['fullHeightInteriorVoidColumns']=int(((~e.occupied).cpu().numpy().all(0)&(d[0]>8)).sum());row['interiorOpeningAreasByFloor']=[[int(sizes[i]) for i in ids] for _,ids,sizes in e.frontier.old_voids]
        assert all(len(x)<=1 and all(v>=4 for v in x) for x in row['interiorOpeningAreasByFloor']);r['wide'].append(row);save();print('guarded wide',seed,row['success'],row['interiorOpeningAreasByFloor'],flush=True)
        e=Environment(context=context,sun_count=8,seed=9200,frontier_factory=functools.partial(PatchFacades,config=cfg))
        for op in ('subtract','add'):r['real'].append(dict(seed=seed,**episode(p,e,op,seed+70000)))
        save();gc.collect()
    for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
        for shape,suns in [((4,48,64),8),((8,64,96),12)]:
            row=dict(device=device,shape=shape,sunSamples=suns,contextBuildings=16)
            try:
                e,setup=timed(lambda:Environment(shape,sun_count=suns,seed=123,device=device,frontier_factory=functools.partial(PatchFacades,config=cfg)),device)
                p=SamplingPolicy(device,config=cfg);p.fields(e);gen=[];apply=[]
                for _ in range(10):
                    a,ms=timed(lambda:p.choose(e,False),device)
                    if a is None:raise RuntimeError('Production guard rejected all draws')
                    gen.append(ms);_,ms=timed(lambda:e.apply(a,audit=False),device);apply.append(ms)
                    assert components(a.mask)['components']<=4
                row.update(setupMs=setup,generation=summary(gen),apply=summary(apply));print('guard benchmark',device,shape,round(row['generation']['medianMs'],2),flush=True)
            except Exception:row['error']=traceback.format_exc();r['failures'].append(row['error']);print(row['error'],flush=True)
            r['benchmark'].append(row);save();gc.collect()
            if device=='mps':torch.mps.empty_cache()
if __name__=='__main__':main()
