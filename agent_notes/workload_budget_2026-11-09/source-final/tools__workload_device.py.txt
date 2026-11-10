"""Final-code CPU/MPS workload rollouts and one native optimizer update."""
import functools,json,time,hashlib,resource,argparse
from pathlib import Path
import torch
from tools.workload_budget import make,episode,ROOT
from voxelform.patch_policy import PatchConfig,PatchLearner
from voxelform.context import ContextLibrary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='all');args=parser.parse_args()
    torch.set_num_threads(1);path=ROOT/'results'/('workload-device-2026-11-09.json' if args.device=='all' else f'workload-device-{args.device}-2026-11-09.json')
    r=dict(rows=[],hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'voxelform').glob('*.py'))]})
    context=ContextLibrary().load('m_bcn_0093')
    for device in (['cpu']+(['mps'] if torch.backends.mps.is_available() else []) if args.device=='all' else [args.device]):
        for allocation in ('quota','learned'):
            c=PatchConfig(allocation=allocation,eligibility='depth',batched=True,vertical_interior=True,minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4)
            for tag,shape,suns,ctx in [('L',(4,48,64),8,None),('XL',(8,64,96),12,None),('Barcelona',context['shape'],8,context)]:
                start=time.perf_counter();e=make(c,shape=shape,context=ctx,device=device);setup=time.perf_counter()-start;p=PatchLearner(device,config=c)
                for op in ('subtract','add'):
                    row=dict(device=device,allocation=allocation,case=tag,shape=shape,sunSamples=suns,contextBuildings=16 if ctx is None else len(ctx['buildings']),setupSeconds=setup,**episode(p,e,op,50123))
                    row['rssHighWaterBytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    row['mpsAllocatedBytes']=torch.mps.current_allocated_memory() if device=='mps' else None
                    r['rows'].append(row);path.write_text(json.dumps(r,indent=2)+'\n');print(device,allocation,tag,op,row['steps'],row['termination'],flush=True)
            e=make(c,shape=(4,12,16),device=device);p=PatchLearner(device,config=c)
            row=episode(p,e,'subtract',123,training=True)
            r.setdefault('training',[]).append(dict(device=device,allocation=allocation,updates=p.updates,**row));path.write_text(json.dumps(r,indent=2)+'\n')
    print(path,flush=True)

if __name__=='__main__':main()
