"""Vectorized current-facade constraints for a directly generated neural mask.

No patch library, candidate batches or mask-ranking policy lives here. The model
supplies one dense mask; these compiled CPU kernels enforce physical eligibility
and prune unsafe selections without exposing another facade layer mid-action.
"""
from dataclasses import dataclass,field
import time
import numpy as np
from scipy import ndimage as ndi
import torch

STRUCTURE=ndi.generate_binary_structure(3,1)
DIRECTIONS=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
ACTION_SPACE='direct_facade_mask_far_v1'


@dataclass
class FacadeMaskAction:
    mask:np.ndarray
    operation:int
    version:int
    generated_voxels:int
    projection_rounds:int=0
    _tensors:dict=field(default_factory=dict,repr=False)
    @property
    def count(self):return int(self.mask.sum())
    def tensor(self,device):
        key=str(device)
        if key not in self._tensors:self._tensors[key]=torch.as_tensor(self.mask,device=device)
        return self._tensors[key]


def exterior_air(occupied):
    air=~occupied;seeds=np.zeros_like(air)
    seeds[:,0,:]=air[:,0,:];seeds[:,-1,:]=air[:,-1,:]
    seeds[:,:,0]=air[:,:,0];seeds[:,:,-1]=air[:,:,-1];seeds[-1]=air[-1]
    return ndi.binary_propagation(seeds,structure=STRUCTURE,mask=air)


def neighbors_horizontal(value,outside=False):
    p=np.pad(value,((0,0),(1,1),(1,1)),constant_values=outside)
    return np.stack((p[:,1:-1,:-2],p[:,1:-1,2:],p[:,:-2,1:-1],p[:,2:,1:-1]),axis=0)


class ExteriorFacades:
    def __init__(self,occupied,allowed,seed=123):
        self.occupied=np.array(occupied,dtype=bool,copy=True);self.allowed=np.array(allowed,dtype=bool,copy=True)
        self.shape=self.occupied.shape;self.version=0;self.projection_seconds=0.;self.commit_seconds=0.
        self.refresh()

    def refresh(self):
        stamp=time.perf_counter();self.exterior_air=exterior_air(self.occupied)
        sides=self.occupied[None]&neighbors_horizontal(self.exterior_air,outside=True)
        self.remove_faces=np.concatenate((sides,np.zeros((2,*self.shape),dtype=bool)))
        near=neighbors_horizontal(self.occupied,outside=False)
        side_growth=(self.allowed&self.exterior_air)[None]&near[[1,0,3,2]]
        # Growth at a current facade's upper/lower edge can raise/lower its extent.
        facade=sides.any(axis=0);edge_growth=np.zeros((2,*self.shape),dtype=bool)
        edge_growth[0,1:]=facade[:-1];edge_growth[1,:-1]=facade[1:]
        edge_growth&=(self.allowed&self.exterior_air)[None]
        self.add_faces=np.concatenate((side_growth,edge_growth))
        self.remove_skin=self.remove_faces.any(axis=0);self.add_skin=self.add_faces.any(axis=0)
        _,self.components=ndi.label(self.occupied,structure=STRUCTURE)
        self.count=int(self.occupied.sum());self.facade_seconds=time.perf_counter()-stamp

    def eligible(self,operation):return self.remove_skin if operation==-1 else self.add_skin

    def _unsafe(self,changed):
        labels,count=ndi.label(changed,structure=STRUCTURE)
        ground=np.unique(labels[0]);ground=ground[ground!=0]
        safe=count>0 and count<=self.components and len(ground)==count
        return safe,labels,count,ground

    def validate_mask(self,mask,operation):
        if operation not in (-1,1) or mask.shape!=self.shape or not mask.any():return False
        if np.any(mask&~self.eligible(operation)):return False
        changed=self.occupied.copy();changed[mask]=operation==1
        if not self._unsafe(changed)[0]:return False
        if operation==1 and np.any(self.exterior_air&~mask&~exterior_air(changed)):return False
        return True

    def project_mask(self,generated,scores,operation,remaining):
        """Constrain one neural output; never synthesize a geometric patch.

        Every retained bit comes from the generated mask and the frozen facade.
        FAR clipping uses neural confidence. Connectivity repair only restores
        unsafe selected bits; it cannot add unselected edits or peel new layers.
        """
        stamp=time.perf_counter();eligible=self.eligible(operation)
        mask=np.asarray(generated,dtype=bool)&eligible
        generated_count=int(mask.sum());limit=int(remaining);minimum=min(2,limit)
        if generated_count>limit:
            ids=np.flatnonzero(mask);chosen=np.argpartition(-scores.ravel()[ids],limit-1)[:limit]
            mask=np.zeros(self.shape,dtype=bool);mask.ravel()[ids[chosen]]=True
        anchors=None
        for iteration in range(10):
            if mask.sum()<minimum:break
            changed=self.occupied.copy();changed[mask]=operation==1
            safe,labels,count,ground=self._unsafe(changed)
            restore=np.zeros_like(mask)
            if not safe:
                if not len(ground):restore[0]=mask[0]
                sizes=np.bincount(labels.ravel())
                if anchors is None and len(ground):
                    roots=ground[np.argsort(-sizes[ground])[:self.components]]
                    anchors=[tuple(np.argwhere(labels==root)[0]) for root in roots]
                roots=np.unique([labels[point] for point in anchors]) if anchors else np.empty(0,int)
                disconnected=changed&~np.isin(labels,roots)
                restore|=mask&ndi.binary_dilation(disconnected,structure=STRUCTURE)
            if operation==1:
                sealed=self.exterior_air&~mask&~exterior_air(changed)
                if sealed.any():restore|=mask&ndi.binary_dilation(sealed,structure=STRUCTURE)
            if safe and not restore.any():
                self.projection_seconds=time.perf_counter()-stamp
                return FacadeMaskAction(mask,operation,self.version,generated_count,iteration)
            if not restore.any():break
            mask&=~restore
        self.projection_seconds=time.perf_counter()-stamp
        return None

    def apply(self,action):
        stamp=time.perf_counter()
        if action.version!=self.version:raise ValueError('Neural mask is stale; use the current facade snapshot')
        if not self.validate_mask(action.mask,action.operation):raise ValueError('Generated mask violates current-facade or topology constraints')
        changed=action.count;self.occupied[action.mask]=action.operation==1
        self.version+=1;self.refresh();self.commit_seconds=time.perf_counter()-stamp
        return changed
