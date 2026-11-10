"""Independent diffuse-sky visibility check: hemisphere rays, exact voxel DDA.

Not the diffusion/usable-area proxy and not solar-path exposure. Held aside from
the grid score so a proxy/raster artefact cannot be called a daylight gain.
"""
import functools,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner
from voxelform.fields import receivers
from voxelform.shadows import MeshShadows,dda_paths
from tools.train_m4 import run_episode

ROOT=Path(__file__).resolve().parents[1]
def diffuse_reference(e,samples=64,face_samples=256):
    ids,points,n,cells,faces=receivers(e.occupied.cpu());side=(faces<4).numpy();eligible=np.flatnonzero(side)
    rng=np.random.default_rng(6601);chosen=rng.choice(eligible,min(len(eligible),face_samples),replace=False)
    p=points[chosen].numpy();normal=n[chosen].numpy();i=np.arange(samples);z=(i+.5)/samples;phi=i*np.pi*(3-np.sqrt(5));r=np.sqrt(1-z*z)
    directions=np.stack((r*np.cos(phi),r*np.sin(phi),z),1).astype('float32')
    a=e.context.get('gridAngle',0);c,s=np.cos(a),np.sin(a);directions[:,:2]=directions[:,:2]@np.array([[c,-s],[s,c]])
    outside=MeshShadows(e.context['buildings'],directions).query(torch.tensor(p)).numpy()
    origins=np.repeat(p+normal*1e-4,samples,axis=0);rays=np.tile(directions,(len(p),1))
    paths,_=dda_paths(origins,rays,e.shape);occupied=np.r_[e.occupied.cpu().numpy().ravel(),False]
    own=(~occupied[paths].any(1)).reshape(len(p),samples).T
    weights=np.maximum(directions@normal.T,0);visible=outside&own
    per_face=(visible*weights).sum(0)/np.maximum(weights.sum(0),1e-12)
    return dict(diffuseReference=float((visible*weights).sum()/weights.sum()),meanFace=float(per_face.mean()),faceStdError=float(per_face.std()/np.sqrt(len(p))),sampledFaces=len(p),hemisphereDirections=samples)

def main():
    torch.set_num_threads(1);lib=ContextLibrary();rows=[]
    c=PatchConfig(allocation='feedback',eligibility='depth',facade_layers=2,budget_reserve_fraction=.2,minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=True)
    path=ROOT/'results/grid-sky-validation-2026-11-09.json'
    if path.exists():raise ValueError('Preserve existing results')
    for site in ['m_bcn_0093','m_bcn_0094','m_nyc_0003','m_prs_0315']:
        for mode in ['world','longest_edge']:
            e=Environment(context=lib.load(site,grid_alignment=mode),device='mps',sun_count=8,far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device='mps'));p=PatchLearner('mps',123,c)
            result=run_episode(p,e,3.,80123,'current',False)
            ref=diffuse_reference(e);rows.append(dict(site=site,alignment=mode,success=result['success'],far=e.far,diffusionFacade=float(e.values['diffuseFacadeProxy'].cpu()),**ref))
            path.write_text(json.dumps(dict(rows=rows,sourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n');print(site,mode,ref,flush=True)
            del e;torch.mps.empty_cache()
    print(path,flush=True)
if __name__=='__main__':main()
