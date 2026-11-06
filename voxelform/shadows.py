"""Directional external-shadow caches, implemented as device-resident tensors.

Sun vectors point TOWARD the sun. Boxes are oriented prisms, not axis-aligned
approximations of rotated neighbours. Maps are nearest-texel approximations;
the direct slab query is the point-sun reference for this geometry.
"""
import math
import numpy as np
import torch


def device_name(requested='auto'):
    if requested!='auto':return requested
    if torch.cuda.is_available():return 'cuda'
    if torch.backends.mps.is_available():return 'mps'
    return 'cpu'


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    elif str(device)=='mps':torch.mps.synchronize()


def sun_path(count=12,azimuth=(90.,270.),peak=55.,grid_yaw=0.):
    """Synthetic directions for method tests, NOT an astronomical ephemeris."""
    t=np.linspace(0.,1.,count);az=np.radians(np.linspace(*azimuth,count)-grid_yaw)
    elevation=np.radians(8.+(peak-8.)*np.sin(np.pi*t))
    return np.stack((np.sin(az)*np.cos(elevation),np.cos(az)*np.cos(elevation),np.sin(elevation)),axis=1)


def box_corners(boxes):
    # cx,cy,cz,half-x,half-y,half-z,cos(yaw),sin(yaw)
    signs=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    result=[]
    for b in np.asarray(boxes):
        p=signs*b[3:6];r=np.array([[b[6],-b[7]],[b[7],b[6]]])
        p[:,:2]=p[:,:2]@r.T;result.append(p+b[:3])
    return np.asarray(result).reshape(-1,3)


def ray_boxes(points,direction,boxes,depth=False,chunk=8192):
    """Batched oriented-box slab tests. Returns blocked, or first-hit distance."""
    if not len(boxes):return torch.full((len(points),),float('inf'),device=points.device) if depth else torch.zeros(len(points),dtype=torch.bool,device=points.device)
    out=[];c=boxes[:,6];s=boxes[:,7]
    d=torch.stack((direction[0]*c+direction[1]*s,-direction[0]*s+direction[1]*c,direction[2].expand_as(c)),dim=-1)
    parallel=d.abs()<1e-8;denom=torch.where(parallel,torch.ones_like(d),d)
    for start in range(0,len(points),chunk):
        delta=points[start:start+chunk,None,:]-boxes[None,:,:3]
        p=torch.stack((delta[:,:,0]*c+delta[:,:,1]*s,-delta[:,:,0]*s+delta[:,:,1]*c,delta[:,:,2]),dim=-1)
        a=(-boxes[None,:,3:6]-p)/denom;b=(boxes[None,:,3:6]-p)/denom
        lo=torch.where(parallel,float('-inf'),torch.minimum(a,b)).amax(dim=-1)
        hi=torch.where(parallel,float('inf'),torch.maximum(a,b)).amin(dim=-1)
        valid=~(parallel&(p.abs()>boxes[None,:,3:6]+1e-6)).any(dim=-1)
        hit=valid&(hi>torch.maximum(lo,torch.full_like(lo,1e-5)))
        if depth:out.append(torch.where(hit,lo.clamp_min(0),float('inf')).amin(dim=1))
        else:out.append(hit.any(dim=1))
    return torch.cat(out) if out else torch.empty(0,device=points.device,dtype=torch.float32 if depth else torch.bool)


class DirectShadows:
    def __init__(self,boxes,suns,device='cpu'):
        self.boxes=torch.as_tensor(boxes,dtype=torch.float32,device=device).reshape(-1,8)
        self.suns=torch.as_tensor(suns,dtype=torch.float32,device=device)
    def query(self,points):return torch.stack([~ray_boxes(points,s,self.boxes) for s in self.suns])
    @property
    def bytes(self):return self.boxes.numel()*4+self.suns.numel()*4


def internal_indices(boxes,bounds):
    """Exact OBB/AABB SAT at setup; touching the boundary is not intrusion."""
    lo,hi=np.array(bounds);center=(lo+hi)/2;half=(hi-lo)/2;indices=[]
    for k,b in enumerate(boxes):
        if abs(b[2]-center[2])>=b[5]+half[2]-1e-6:continue
        axes=np.array([[1,0],[0,1],[b[6],b[7]],[-b[7],b[6]]],float)
        d=np.abs(axes@(b[:2]-center[:2]));ra=np.abs(axes)@half[:2]
        rb=np.abs(axes@np.array([[b[6],-b[7]],[b[7],b[6]]]))@b[3:5]
        if np.all(d<ra+rb-1e-6):indices.append(k)
    return indices


class BoundaryShadows(DirectShadows):
    """User's box-face method; bake only up-sun faces, one set per sun.

    Occluders intruding into the box are queried directly as a documented
    fallback. A pure boundary mask cannot represent them.
    """
    def __init__(self,boxes,suns,bounds,pitch=1.,device='cpu',fallback=True):
        super().__init__(boxes,suns,device);self.bounds=torch.tensor(bounds,dtype=torch.float32,device=device)
        self.maps=[];self.face_union=set();self.pitch=pitch
        inside=internal_indices(np.asarray(boxes),bounds);self.intruders=self.boxes[inside]
        outside=[k for k in range(len(boxes)) if k not in inside];bakers=self.boxes[outside]
        self.intruder_count=len(inside);self.fallback=fallback
        lower,upper=np.asarray(bounds)
        for raw,s in zip(np.asarray(suns),self.suns):
            faces={}
            for axis in range(3):
                if abs(raw[axis])<1e-8:continue
                side=1 if raw[axis]>0 else 0;uv=[k for k in range(3) if k!=axis]
                sizes=np.maximum(1,np.ceil((upper-lower)[uv]/pitch).astype(int));step=(upper-lower)[uv]/sizes
                u=lower[uv[0]]+(torch.arange(int(sizes[0]),device=device)+.5)*step[0]
                v=lower[uv[1]]+(torch.arange(int(sizes[1]),device=device)+.5)*step[1]
                a,b=torch.meshgrid(u,v,indexing='ij');p=torch.zeros((a.numel(),3),device=device)
                p[:,axis]=upper[axis] if side else lower[axis];p[:,uv[0]]=a.flatten();p[:,uv[1]]=b.flatten()
                image=(~ray_boxes(p,s,bakers)).reshape(tuple(sizes))
                faces[axis]=(uv,step,sizes,image);self.face_union.add((axis,side))
            self.maps.append(faces)

    def query(self,points):
        result=[]
        for s,faces in zip(self.suns,self.maps):
            endpoint=torch.where(s>=0,self.bounds[1],self.bounds[0]);denom=torch.where(s.abs()<1e-8,torch.ones_like(s),s)
            t=torch.where(s.abs()<1e-8,float('inf'),(endpoint-points)/denom)
            distance,axis=t.min(dim=1);q=points+s*distance.clamp_min(0)[:,None];visible=torch.ones(len(points),dtype=torch.bool,device=points.device)
            for k,(uv,step,sizes,image) in faces.items():
                u=((q[:,uv[0]]-self.bounds[0,uv[0]])/step[0]).floor().long().clamp(0,int(sizes[0])-1)
                v=((q[:,uv[1]]-self.bounds[0,uv[1]])/step[1]).floor().long().clamp(0,int(sizes[1])-1)
                visible=torch.where(axis==k,image[u,v],visible)
            if self.fallback and self.intruder_count:visible&=~ray_boxes(points,s,self.intruders)
            result.append(visible)
        return torch.stack(result)
    @property
    def bytes(self):return sum(v[3].numel() for faces in self.maps for v in faces.values())+super().bytes


class DepthShadows(DirectShadows):
    """Sun-aligned orthographic external depth maps, closest surface to sun."""
    def __init__(self,boxes,suns,bounds,pitch=1.,device='cpu'):
        super().__init__(boxes,suns,device);self.maps=[]
        corners=np.array([(x,y,z) for x in (bounds[0][0],bounds[1][0]) for y in (bounds[0][1],bounds[1][1]) for z in (bounds[0][2],bounds[1][2])])
        geometry=box_corners(boxes);self.anchor=np.mean(bounds,axis=0)
        for raw,s in zip(np.asarray(suns),self.suns):
            u=np.cross(raw,[0,0,1]) if abs(raw[2])<.99 else np.cross(raw,[0,1,0]);u=u/np.linalg.norm(u);v=np.cross(raw,u)
            basis=np.stack((u,v,raw),axis=1);projected=(corners-self.anchor)@basis
            low=projected[:,:2].min(0)-pitch;high=projected[:,:2].max(0)+pitch
            sizes=np.ceil((high-low)/pitch).astype(int);w=float(np.max((geometry-self.anchor)@raw)+10) if len(geometry) else float(projected[:,2].max()+10)
            ut=torch.as_tensor(u,dtype=torch.float32,device=device);vt=torch.as_tensor(v,dtype=torch.float32,device=device)
            a=low[0]+(torch.arange(int(sizes[0]),device=device)+.5)*pitch;b=low[1]+(torch.arange(int(sizes[1]),device=device)+.5)*pitch
            x,y=torch.meshgrid(a,b,indexing='ij')
            origin=torch.as_tensor(self.anchor,dtype=torch.float32,device=device)+x.flatten()[:,None]*ut+y.flatten()[:,None]*vt+w*s
            depth=(w-ray_boxes(origin,-s,self.boxes,depth=True)).reshape(tuple(sizes))
            self.maps.append((ut,vt,torch.tensor(low,dtype=torch.float32,device=device),pitch,sizes,depth))
        self.anchor=torch.tensor(self.anchor,dtype=torch.float32,device=device)

    def query(self,points):
        result=[];p=points-self.anchor
        for s,(u,v,low,pitch,sizes,depth) in zip(self.suns,self.maps):
            a=((p@u-low[0])/pitch).floor().long().clamp(0,int(sizes[0])-1);b=((p@v-low[1])/pitch).floor().long().clamp(0,int(sizes[1])-1)
            result.append(p@s>=depth[a,b]-1e-4)
        return torch.stack(result)
    @property
    def bytes(self):return sum(v[-1].numel()*4 for v in self.maps)+super().bytes


def dda_paths(origins,directions,shape,spacing=(1.,1.,3.)):
    """Independent NumPy exact grid traversal; returns cell IDs and entry t.

    Layout is z,y,x. Simultaneous edge crossings advance all tied axes.
    Used for dynamic-map precomputation and point-reference validation.
    """
    o=np.asarray(origins,float);d=np.broadcast_to(directions,o.shape).copy();size=np.array(shape[::-1]);h=np.array(spacing)
    parallel=np.abs(d)<1e-12;denom=np.where(parallel,1,d)
    t0=(0-o)/denom;t1=(size*h-o)/denom
    entry=np.maximum(np.max(np.where(parallel,-np.inf,np.minimum(t0,t1)),axis=1),0.)
    leave=np.min(np.where(parallel,np.inf,np.maximum(t0,t1)),axis=1)
    active=(leave>entry+1e-8)&~np.any(parallel&((o<0)|(o>size*h)),axis=1)
    p=o+d*(entry+1e-6)[:,None];cell=np.floor(p/h).astype(int);step=np.sign(d).astype(int)
    boundary=(cell+(step>0))*h;next_t=np.where(parallel,np.inf,(boundary-o)/denom)
    increment=np.where(parallel,np.inf,h/np.abs(denom));dummy=int(np.prod(shape));ids=[];depths=[];current=entry
    for _ in range(int(sum(shape))+3):
        valid=active&np.all((cell>=0)&(cell<size),axis=1)
        index=(cell[:,2]*shape[1]+cell[:,1])*shape[2]+cell[:,0]
        ids.append(np.where(valid,index,dummy));depths.append(current.copy())
        minimum=next_t.min(1);advance=next_t<=minimum[:,None]+1e-9
        cell=cell+step*advance;next_t=next_t+np.where(advance,increment,0.)
        current=minimum;active=valid&(current<leave-1e-8)
    return np.stack(ids,axis=1),np.stack(depths,axis=1)


class SelfShadowMaps:
    """Fixed Sun-plane ray paths, dynamic occupancy gather/reduction on GPU.

    Exact voxel entry depths along cached texel rays; receiver lookup still has
    map-resolution error, measured against exact receiver DDA in benchmarks.
    """
    def __init__(self,shape,suns,pitch=1.,spacing=(1.,1.,3.),device='cpu'):
        self.maps=[];bounds=np.array([[0,0,0],np.array(shape[::-1])*spacing]);self.suns=torch.tensor(suns,dtype=torch.float32,device=device)
        empty=DepthShadows([],suns,bounds,pitch,device);self.anchor=empty.anchor
        for raw,parameters in zip(suns,empty.maps):
            u,v,low,pitch,sizes,_=parameters;un=u.cpu().numpy();vn=v.cpu().numpy();lo=low.cpu().numpy()
            corners=np.array([(x,y,z) for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
            w=float(np.max((corners-bounds.mean(0))@raw)+2.)
            x,y=np.meshgrid(lo[0]+(np.arange(sizes[0])+.5)*pitch,lo[1]+(np.arange(sizes[1])+.5)*pitch,indexing='ij')
            origins=bounds.mean(0)+x.reshape(-1,1)*un+y.reshape(-1,1)*vn+w*raw
            ids,t=dda_paths(origins,-np.asarray(raw),shape,spacing)
            self.maps.append(dict(parameters=parameters,ids=torch.tensor(ids,dtype=torch.long,device=device),
                entry=torch.tensor(w-t,dtype=torch.float32,device=device),image=None))

    def update(self,occupied):
        flat=torch.cat((occupied.flatten(),torch.zeros(1,dtype=torch.bool,device=occupied.device)))
        for m in self.maps:
            m['image']=torch.where(flat[m['ids']],m['entry'],float('-inf')).amax(dim=1).reshape(m['parameters'][-2].tolist())

    def query(self,points,normals=None):
        result=[];p=points-self.anchor
        for s,m in zip(self.suns,self.maps):
            u,v,low,pitch,sizes,_=m['parameters'];a=((p@u-low[0])/pitch).floor().long().clamp(0,int(sizes[0])-1);b=((p@v-low[1])/pitch).floor().long().clamp(0,int(sizes[1])-1)
            receiver_depth=p@s
            if normals is not None:
                # Receiver-plane correction avoids flat faces shadowing
                # themselves solely because the nearest texel ray is offset.
                du=low[0]+(a+.5)*pitch-p@u;dv=low[1]+(b+.5)*pitch-p@v
                denominator=(normals@s)
                receiver_depth=receiver_depth-((normals@u)*du+(normals@v)*dv)/torch.where(denominator.abs()>1e-6,denominator,torch.ones_like(denominator))
            result.append(receiver_depth>=m['image'][a,b]-1e-3)
        return torch.stack(result)
    @property
    def bytes(self):return sum(m['ids'].numel()*8+m['entry'].numel()*4 for m in self.maps)


class ReceiverCache:
    """External visibility is immutable: query only newly exposed receiver IDs."""
    def __init__(self,method,max_receivers):
        self.method=method;dev=method.suns.device;self.known=torch.zeros(max_receivers,dtype=torch.bool,device=dev)
        self.values=torch.zeros((len(method.suns),max_receivers),dtype=torch.bool,device=dev)
    def query(self,ids,points):
        new=~self.known[ids];fresh=ids[new]
        if len(fresh):self.values[:,fresh]=self.method.query(points[new]);self.known[fresh]=True
        return self.values[:,ids],len(fresh)
