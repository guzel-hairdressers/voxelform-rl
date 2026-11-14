"""Bounded geometry/feasibility comparison; not a learned-quality benchmark."""
import argparse,functools,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades
from voxelform.unified import UnifiedLearner
from tools.unified_training import unified_episode

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='');args=ap.parse_args()
    if args.tag and not args.tag.replace('-','').isalnum():raise ValueError('Use a simple filename tag')
    torch.set_num_threads(1);device='mps' if torch.backends.mps.is_available() else 'cpu'
    config=PatchConfig(mode='combined',allocation='feedback',eligibility='depth',facade_layers=2,minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=True,budget_reserve_fraction=.2)
    result=dict(device=device,note='Untrained matched-seed controls. This tests action geometry and FAR feasibility, not convergence or residential compliance.',setup=[],rows=[])
    result['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'voxelform/unified.py',ROOT/'unified_server.py']}
    out=ROOT/f"results/floor-growth-preview{'-'+args.tag if args.tag else ''}-2026-11-14.json"
    def save():out.write_text(json.dumps(result,indent=2)+'\n')
    if out.exists():raise ValueError('Preserve existing results')
    lib=ContextLibrary()
    for site in ['m_nyc_0003','m_bcn_0094']:
        start=time.perf_counter();context=lib.load(site,floors=10,grid_alignment='auto_fit')
        env=Environment(context=context,shape=context['shape'],device=device,far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=config,depth_device=device))
        external=env.precompute_external()
        result['setup'].append(dict(site=site,shape=env.shape,buildings=len(context['buildings']),seconds=time.perf_counter()-start,external=external));save()
        for seed in [123,456]:
            for copy in [False,True]:
                p=UnifiedLearner(device,seed,config,amount_feedback=True,net_guided=True,floor_copy=copy)
                row=unified_episode(p,env,3.,seed,False)
                solid=env.frontier.occupied
                row.update(site=site,seed=seed,arm='floor-copy-linked' if copy else 'roof-patches',unsupportedCells=int((solid[1:]&~solid[:-1]).sum()),floorAreas=solid.sum((1,2)).tolist())
                result['rows'].append(row);save()
                print(site,seed,row['arm'],row['success'],round(row['far'],3),row['steps'],row['attempts'],round(row['seconds'],2),row['unsupportedCells'],flush=True)
        # Demonstrate actual on-policy learning through the active decoder.
        p=UnifiedLearner(device,123,config,amount_feedback=True,net_guided=True,floor_copy=True)
        start=time.perf_counter();deadline=start+5;episodes=[]
        while time.perf_counter()<deadline:episodes.append(unified_episode(p,env,3.,123+len(episodes),True,deadline))
        result.setdefault('learning',[]).append(dict(site=site,seconds=time.perf_counter()-start,updates=p.updates,episodes=episodes));save()
    print(out,flush=True)
if __name__=='__main__':main()
