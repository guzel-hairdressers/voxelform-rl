"""Bounded layer ablation and neural paired-edit experiment; no live mutation."""
import json,functools,hashlib,time,argparse
from pathlib import Path
import torch
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner
from voxelform.redistribution import choose,commit
from tools.training_profile import episode

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);r=dict(layers=[],exchanges=[],hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'voxelform/redistribution.py',ROOT/'voxelform/patch_policy.py']})
    path=ROOT/('results/redistribution-fast-2026-11-09.json' if args.fast else 'results/redistribution-2026-11-09.json')
    def save():path.write_text(json.dumps(r,indent=2)+'\n')
    for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
        for shape in ((4,48,64),(8,64,96)):
            for layers in (1,2,3):
                c=PatchConfig(facade_layers=layers,allocation='feedback',budget_fraction=1. if args.fast else .5,budget_reserve_fraction=.2,minimum_patch=4,maximum_patches=4,batched=True)
                e=Environment(shape,device=device,sun_count=8 if shape[0]==4 else 12,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=device))
                for seed in ((123,) if args.fast else (123,456,789)):
                    p=PatchLearner(device,seed,c);row=episode(e,p,seed+60000,False);r['layers'].append(dict(device=device,shape=shape,layers=layers,seed=seed,**row));save()
                print(device,shape,layers,[(x['steps'],x['success']) for x in r['layers'] if x['device']==device and x['shape']==shape and x['layers']==layers],flush=True)
        # Taller permitted envelope, initial occupied height three floors. Once
        # FAR 3 is reached, independent paired refinement remains possible.
        c=PatchConfig(facade_layers=3,minimum_patch=4,maximum_patches=4)
        e=Environment((8,24,32),device=device,sun_count=8,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=device))
        p=PatchLearner(device,123,c);solid=e.allowed.clone();solid[3:]=False;e.restore_occupancy(solid)
        before=e.far;start=time.perf_counter();rows=[]
        for step in range(8):
            move=choose(p,e,96,True)
            if move is None:p.record(-.1);break
            old=float(e.quality.cpu());commit(e,move);p.record(float(e.quality.cpu())-old)
            rows.append(dict(count=move.count,far=e.far,quality=float(e.quality.cpu()),buildings=e.frontier.components))
        if p.tape:stats=p.finish(0.)
        else:stats={}
        r['exchanges'].append(dict(device=device,initialFar=before,finalFar=e.far,steps=rows,seconds=time.perf_counter()-start,learning=stats));save();print('exchange',device,len(rows),before,e.far,flush=True)
    print(path,flush=True)
if __name__=='__main__':main()
