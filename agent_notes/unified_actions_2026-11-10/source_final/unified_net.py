"""Paired untrained control: isolate net FAR guidance with growth allowed in every exchange."""
import argparse,functools,hashlib,json,time
import numpy as np
from voxelform.context import ContextLibrary,public_scene
from pathlib import Path
import torch
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades
from voxelform.unified import UnifiedLearner
from tools.unified_training import unified_episode

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--real',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);device='mps';tag='real' if args.real else 'xl';path=ROOT/('results/unified-net-real-2026-11-10.json' if args.real else 'results/unified-net-2026-11-10.json')
    if path.exists():raise ValueError('Preserve prior results')
    c=PatchConfig(mode='combined',allocation='feedback',eligibility='depth',facade_layers=2,minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=True,budget_reserve_fraction=.2)
    start=time.perf_counter();context=ContextLibrary().load('m_bcn_0094',floors=10,grid_alignment='auto_fit') if args.real else None
    e=Environment((10,64,96),context=context,device=device,sun_count=8 if args.real else 12,far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=device));e.precompute_external()
    snapshots=ROOT/'results/checkpoints/unified-net-proposals';snapshots.mkdir(parents=True,exist_ok=True)
    if context:(snapshots/f'{tag}-context.json').write_text(json.dumps(public_scene(context)))
    result=dict(rows=[],device=device,setupSeconds=time.perf_counter()-start,note='Identical initialized networks and sampled seeds; no training. Live GPU training may run concurrently.',hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'voxelform/unified.py',ROOT/'tools/unified_training.py']})
    for seed,trained in [(123,False),(456,False),(789,False),(123,True),(456,True)]:
        for feedback in [False,True]:
            p=UnifiedLearner(device,seed,c,amount_feedback=True,repair_remove=True,net_guided=feedback)
            if trained:
                state=torch.load(ROOT/f'results/checkpoints/unified-2026-11-10/unified-{seed}.pt',map_location=device,weights_only=True)
                p.load(state['policy']);p.amount_feedback=True;p.repair_remove=True;p.net_guided=feedback
            row=unified_episode(p,e,3.,seed+80000+(1 if args.real else 0),False)
            np.savez_compressed(snapshots/f'{tag}-{seed}-{trained}-{feedback}.npz',occupied=e.occupied.cpu().numpy(),allowed=e.allowed.cpu().numpy())
            result['rows'].append(dict(seed=seed,trained=trained,netGuided=feedback,**row));path.write_text(json.dumps(result,indent=2)+'\n')
            print(seed,'trained',trained,'netGuided',feedback,row['success'],round(row['far'],4),row['steps'],round(row['quality'],4),flush=True)
    print(path,flush=True)
if __name__=='__main__':main()
