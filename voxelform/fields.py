"""Device-resident relative environmental fields; no dBA/lux claims."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .shadows import DirectShadows


def grid_points(shape,device='cpu',spacing=(1.,1.,3.)):
    z,y,x=torch.meshgrid(*[torch.arange(n,device=device,dtype=torch.float32)+.5 for n in shape],indexing='ij')
    return torch.stack((x*spacing[0],y*spacing[1],z*spacing[2]),dim=-1)


def noise_prior(points,segments,soft_radius=2.):
    """Analytic integral of 1/(r²+r0²) over finite line sources.

    Near a long line it decays approximately 1/d; far from a finite line,
    approximately 1/d². Strength is relative; no calibrated acoustic units,
    diffraction, ground effects or building shielding are included.
    """
    value=torch.zeros(points.shape[:-1],device=points.device)
    for start,end,strength in segments:
        if strength<0:raise ValueError('negative source strength')
        a=points.new_tensor(start);b=points.new_tensor(end);v=b-a;length=v.norm()
        if float(length)<=0:raise ValueError('zero-length source')
        direction=v/length;delta=points-a;t=(delta*direction).sum(-1)
        rho=((delta-t[...,None]*direction).square().sum(-1)+soft_radius**2).sqrt()
        value+=strength*(torch.atan((length-t)/rho)+torch.atan(t/rho))/rho
    return value


def sky_boundary(shape,boxes,device='cpu',samples=12,spacing=(1.,1.,3.)):
    """Scalar external sky-openness boundary for the diffusion proxy."""
    i=np.arange(samples);r=np.sqrt((i+.5)/samples);phi=i*math.pi*(3-math.sqrt(5))
    suns=np.stack((r*np.cos(phi),r*np.sin(phi),np.sqrt(1-r*r)),axis=1)
    padded=tuple(n+2 for n in shape);p=grid_points(padded,device,spacing)-torch.tensor(spacing,device=device)
    boundary=torch.ones(padded,dtype=torch.bool,device=device);boundary[1:-1,1:-1,1:-1]=False;boundary[0]=False
    out=torch.zeros(padded,device=device)
    out[boundary]=DirectShadows(boxes,suns,device).query(p[boundary]).float().mean(0)
    return out


def diffuse_access(occupied,boundary=None,iterations=64,decay_length=12.,spacing=(1.,1.,3.)):
    """Screened Laplace/Jacobi accessibility in exterior air, not radiance.

    Solid walls absorb to zero; open boundaries inject scalar sky openness.
    This allows gradual decay in wells but also diffusion around corners.
    It is a deliberately approximate field, NOT daylight factor or lux.
    """
    if boundary is None:
        boundary=torch.ones(tuple(n+2 for n in occupied.shape),device=occupied.device);boundary[0]=0
    field=torch.zeros_like(occupied,dtype=torch.float32);wx,wy,wz=[1/(v*v) for v in spacing]
    denominator=2*(wx+wy+wz)+1/(decay_length**2);air=~occupied
    for _ in range(iterations):
        padded=boundary.clone();padded[1:-1,1:-1,1:-1]=field
        field=(wx*(padded[1:-1,1:-1,2:]+padded[1:-1,1:-1,:-2])+
               wy*(padded[1:-1,2:,1:-1]+padded[1:-1,:-2,1:-1])+
               wz*(padded[2:,1:-1,1:-1]+padded[:-2,1:-1,1:-1]))/denominator
        field=field*air
    return field


NORMALS=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))


def receivers(occupied,spacing=(1.,1.,3.)):
    """Persistent IDs and physical face-centre samples. Ground underside omitted."""
    pad=F.pad(occupied,(1,1,1,1,1,1),value=False)
    near=(pad[1:-1,1:-1,2:],pad[1:-1,1:-1,:-2],pad[1:-1,2:,1:-1],pad[1:-1,:-2,1:-1],pad[2:,1:-1,1:-1],pad[:-2,1:-1,1:-1])
    exposed=torch.stack([occupied&~v for v in near]);exposed[5,0]=False
    ids=exposed.flatten().nonzero().flatten();volume=occupied.numel();face=ids//volume;cell=ids%volume
    nz,ny,nx=occupied.shape;x=cell%nx;y=(cell//nx)%ny;z=cell//(nx*ny)
    normal=torch.tensor(NORMALS,dtype=torch.float32,device=occupied.device)[face]
    p=torch.stack((x+.5,y+.5,z+.5),dim=1)*torch.tensor(spacing,device=occupied.device)+normal*torch.tensor(spacing,device=occupied.device)/2
    return ids,p,normal,cell,face


def usable_depth_proxy(occupied,air_light,depth=7,boundary=None):
    """Potential floor area within a bounded 4-neighbour distance of lit facade.

    No room subdivision, corridor or furniture feasibility is implied.
    Horizontal distance is in 1m grid steps. Roof exposure is not a window.
    """
    air=(~occupied)&(air_light>.15);pad=F.pad(air,(1,1,1,1),value=boundary is None)
    if boundary is not None:
        # Use measured external sky openness at envelope sides as well as in
        # internal air. Do not silently assume obstructed envelope edges are lit.
        pad[:,0,:]=boundary[1:-1,0,:]>.15;pad[:,-1,:]=boundary[1:-1,-1,:]>.15
        pad[:,:,0]=boundary[1:-1,:,0]>.15;pad[:,:,-1]=boundary[1:-1,:,-1]>.15
    seed=occupied&(pad[:,1:-1,2:]|pad[:,1:-1,:-2]|pad[:,2:,1:-1]|pad[:,:-2,1:-1])
    reach=seed
    for _ in range(max(0,depth-1)):
        p=F.pad(reach,(1,1,1,1),value=False)
        reach=occupied&(reach|p[:,1:-1,2:]|p[:,1:-1,:-2]|p[:,2:,1:-1]|p[:,:-2,1:-1])
    return reach
