"""Matched real-site grid study; keep physical coordinates and solar directions."""
import functools,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner
from tools.train_m4 import run_episode

ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(1);path=ROOT/'results/grid-quality-2026-11-09.json'
    if path.exists():raise ValueError('Preserve existing results')
    library=ContextLibrary();c=PatchConfig(allocation='feedback',eligibility='depth',facade_layers=2,budget_reserve_fraction=.2,minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=True)
    r=dict(rows=[],config=vars(c),device='mps',hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'voxelform').glob('*.py'))]})
    sites=['m_bcn_0093','m_bcn_0094','m_nyc_0003','m_prs_0315']
    for site in sites:
        for alignment in ('world','principal_major','principal_minor','longest_edge'):
            start=time.perf_counter();context=library.load(site,grid_alignment=alignment)
            e=Environment(context=context,device='mps',sun_count=8,far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device='mps'))
            setup=time.perf_counter()-start;mask=context['allowed'];perimeter=int(np.abs(np.diff(np.pad(mask,1).astype(int),axis=0)).sum()+np.abs(np.diff(np.pad(mask,1).astype(int),axis=1)).sum())
            p=np.asarray(context['perimeter']);actual_perimeter=float(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum())
            for seed in (123,456,789):
                policy=PatchLearner('mps',seed,c);result=run_episode(policy,e,3.,seed+80000,'current',False)
                # Fixed physical denominators; legacy Q includes bounding-grid
                # roof normalization, which is not suitable for orientation ranking.
                m=result['environment'];usable=m['potentialUsableArea']/max(1,m['occupiedFloorArea'])
                roof=e.receiver_data[-1]==4;roof_area=float((e.visible[:,roof].float().mean(0)>.4).sum().cpu())
                score=.7*usable+.18*m['directSunProxy']+.12*roof_area/e.plot_area
                row=dict(site=site,alignment=alignment,angle=context['gridAngle'],shape=e.shape,seed=seed,
                    rasterArea=int(mask.sum()),siteArea=e.plot_area,rasterPerimeter=perimeter,referencePerimeter=actual_perimeter,
                    setupSeconds=setup,usableFraction=usable,sourceNormalizedScore=score,roofSunArea=roof_area,**result)
                r['rows'].append(row);path.write_text(json.dumps(r,indent=2)+'\n');print(site,alignment,seed,result['success'],result['steps'],round(score,4),flush=True)
            del e;torch.mps.empty_cache()
    print(path,flush=True)
if __name__=='__main__':main()
