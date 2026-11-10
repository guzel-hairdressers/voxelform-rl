"""Independent checkpoint evaluation with each environmental component retained."""
import functools,json,hashlib,time
from pathlib import Path
import torch
from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner
from tools.train_m4 import run_episode,ROOT

def main():
    torch.set_num_threads(1);source=ROOT/'results/m4-training-pilot-2026-11-09.json';r=json.loads(source.read_text());c=PatchConfig(**r['config']);device=r['device'];lib=ContextLibrary()
    cases=[]
    for tag,shape,plan,site,seed in [('heldout-irregular',(4,36,44),'courtyard',None,19301),('heldout-XL',(8,64,96),'rectangle',None,19302),('heldout-Barcelona',(6,24,32),'rectangle','m_bcn_0094',9301)]:
        e=Environment(shape,plan=plan,seed=seed,context=lib.load(site) if site else None,device=device,sun_count=12 if shape[0]==8 else 8,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=device))
        cases.append((tag,e))
    out=dict(rows=[],source=str(source.relative_to(ROOT)),hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'tools/train_m4.py',*sorted((ROOT/'voxelform').glob('*.py'))]})
    path=ROOT/'results/m4-training-evaluation-2026-11-09.json'
    for row in r['rows']:
        p=PatchLearner(device,row['seed'],c);p.load(torch.load(ROOT/f'results/checkpoints/m4-training-pilot-2026-11-09/{row["name"]}-{row["seed"]}.pt',map_location=device,weights_only=True))
        final=[dict(case=tag,**run_episode(p,e,3.,row['seed']+80000+i,row['mode'],False)) for i,(tag,e) in enumerate(cases)]
        errors=[abs(a['quality']-b['quality']) for a,b in zip(final,row['final'])]
        out['rows'].append(dict(name=row['name'],seed=row['seed'],final=final,originalQualityMaxDifference=max(errors)))
        path.write_text(json.dumps(out,indent=2)+'\n');print(row['name'],row['seed'],'replay error',max(errors),flush=True)
    print(path,flush=True)

if __name__=='__main__':main()
