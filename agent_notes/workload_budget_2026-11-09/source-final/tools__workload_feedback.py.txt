"""Follow-up: compensate for measured projection losses; no additional training."""
import argparse,json,hashlib,gc
from pathlib import Path
import torch
from voxelform.patch_policy import PatchConfig,PatchLearner
from voxelform.context import ContextLibrary
from tools.workload_budget import make,episode,ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--margin',type=int,default=2);ap.add_argument('--reserve',type=float,default=0.);ap.add_argument('--seeds',type=int,nargs='+',default=[123,456,789]);ap.add_argument('--tag',default='');args=ap.parse_args();torch.set_num_threads(1)
    path=ROOT/f'results/workload-feedback-{args.device}{args.tag}-2026-11-09.json';r=dict(device=args.device,margin=args.margin,reserve=args.reserve,rows=[],hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'voxelform').glob('*.py'))]})
    cfg=PatchConfig(allocation='feedback',eligibility='depth',minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=True,budget_margin=args.margin,budget_reserve_fraction=args.reserve)
    ctx=ContextLibrary().load('m_bcn_0093')
    for seed in args.seeds:
        p=PatchLearner(args.device,seed,cfg)
        for tag,shape,plan,context in [('S',(4,16,20),'rectangle',None),('irregular',(4,20,28),'branch',None),('wide',(4,40,48),'rectangle',None),('L',(4,48,64),'rectangle',None),('XL',(8,64,96),'rectangle',None),('Barcelona',ctx['shape'],'rectangle',ctx)]:
            e=make(cfg,shape,plan,args.device,context,reuse=True)
            for op in ('subtract','add'):
                row=dict(seed=seed,case=tag,shape=e.shape,**episode(p,e,op,seed+50000));r['rows'].append(row);path.write_text(json.dumps(r,indent=2)+'\n')
                print(args.device,'feedback',seed,tag,op,row['steps'],row['termination'],row['remaining'],flush=True)
            del e;gc.collect()
    print(path,flush=True)

if __name__=='__main__':main()
