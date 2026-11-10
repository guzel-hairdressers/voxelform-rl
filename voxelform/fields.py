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


def sky_boundary(shape,boxes,device='cpu',samples=12,spacing=(1.,1.,3.),buildings=None,grid_angle=0.):
    """Scalar external sky-openness boundary for the diffusion proxy."""
    i=np.arange(samples);r=np.sqrt((i+.5)/samples);phi=i*math.pi*(3-math.sqrt(5))
    suns=np.stack((r*np.cos(phi),r*np.sin(phi),np.sqrt(1-r*r)),axis=1)
    if grid_angle:
        c,s=math.cos(grid_angle),math.sin(grid_angle);suns[:,:2]=suns[:,:2]@np.array([[c,-s],[s,c]])
    padded=tuple(n+2 for n in shape);p=grid_points(padded,device,spacing)-torch.tensor(spacing,device=device)
    boundary=torch.ones(padded,dtype=torch.bool,device=device);boundary[1:-1,1:-1,1:-1]=False;boundary[0]=False
    out=torch.zeros(padded,device=device)
    if buildings is None:
        method=DirectShadows(boxes,suns,device)
    else:
        from .shadows import MeshShadows
        method=MeshShadows(buildings,suns,device)
    out[boundary]=method.query(p[boundary]).float().mean(0)
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


def facade_sky_access(occupied,air_light,boundary,facades=None):
    """Diffuse-sky proxy on lateral facades, sampled from adjacent exterior air.

    Returns four oriented face fields and their per-occupied-cell mean. Roofs
    are excluded. This reuses sky diffusion, not directional solar visibility;
    no sun-path samples or new ray casts are involved. Air-cell-centre sampling
    is a numerical proxy, not facade illuminance or daylight compliance.
    """
    padded=boundary.clone();padded[1:-1,1:-1,1:-1]=air_light
    adjacent=torch.stack((padded[1:-1,1:-1,2:],padded[1:-1,1:-1,:-2],
                          padded[1:-1,2:,1:-1],padded[1:-1,:-2,1:-1]))
    if facades is None:
        from .actions import exterior_air,neighbors_horizontal
        solid=occupied.detach().cpu().numpy()
        facades=torch.as_tensor(solid[None]&neighbors_horizontal(exterior_air(solid),outside=True),device=occupied.device)
        # neighbors_horizontal returns -X,+X,-Y,+Y; this function uses NORMALS.
        facades=facades[[1,0,3,2]]
    faces=adjacent*facades
    return faces,faces.sum(0)/facades.sum(0).clamp_min(1)


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


def facade_depth(occupied,exterior=None,pitch=1.):
    """Exact horizontal Euclidean centre-to-exterior-wall distance, in metres.

    CPU SciPy EDT on a half-cell lattice includes every closest point on the
    axis-aligned wall segments for cell-centre queries. Floors are independent:
    roof/underside exposure is not window frontage. Sealed air is not exterior.
    Empty cells return zero. No sky/lighting sufficiency is asserted here.
    """
    from scipy import ndimage as ndi
    from .actions import exterior_air,neighbors_horizontal
    occupied=np.asarray(occupied,dtype=bool)
    if not math.isfinite(pitch) or pitch<=0:raise ValueError('pitch must be positive and finite')
    exterior=exterior_air(occupied) if exterior is None else np.asarray(exterior,dtype=bool)
    faces=occupied[None]&neighbors_horizontal(exterior,outside=True)
    nz,ny,nx=occupied.shape;result=np.zeros(occupied.shape,dtype=np.float32)
    for z in range(nz):
        if not occupied[z].any():continue
        walls=np.zeros((2*ny+1,2*nx+1),dtype=bool)
        for side in range(4):
            y,x=np.nonzero(faces[side,z])
            if side<2:
                xx=2*x+(2 if side==1 else 0)
                for offset in (0,1,2):walls[2*y+offset,xx]=True
            else:
                yy=2*y+(2 if side==3 else 0)
                for offset in (0,1,2):walls[yy,2*x+offset]=True
        if walls.any():
            distance=ndi.distance_transform_edt(~walls,sampling=pitch/2)[1::2,1::2]
            result[z,occupied[z]]=distance[occupied[z]]
        else:result[z,occupied[z]]=np.inf
    return result


def depth_badness(depth):
    """Zero through 8m; linear ramp to one at 16m; continuous beyond threshold."""
    if isinstance(depth,torch.Tensor):return ((depth-8.)/8.).clamp(0,1)
    return np.clip((np.asarray(depth)-8.)/8.,0.,1.)


_DEPTH_TENSOR_GRIDS={}
def facade_depth_tensor(occupied,exterior):
    """Exact half-lattice Euclidean wall distance using device-native reductions.

    No CPU EDT or per-voxel Python work. Two separable squared-distance passes
    on the same wall lattice as facade_depth: plan-row minima, then plan-column
    minima at voxel centres. The floor axis is batched independently. Chunk floors only to bound temporary memory.
    Exterior reachability is supplied by the caller's geometry safeguard.
    """
    occupied=occupied.bool();exterior=exterior.bool();nz,ny,nx=occupied.shape
    p=F.pad(exterior,(1,1,1,1),value=True)
    left=occupied&p[:,1:-1,:-2];right=occupied&p[:,1:-1,2:]
    up=occupied&p[:,:-2,1:-1];down=occupied&p[:,2:,1:-1]
    vertical=torch.zeros((nz,1,2*ny+1,2*nx+1),device=occupied.device)
    horizontal=torch.zeros_like(vertical)
    vertical[:,0,1::2,:-1:2]=left.float()
    vertical[:,0,1::2,2::2]=torch.maximum(vertical[:,0,1::2,2::2],right.float())
    horizontal[:,0,:-1:2,1::2]=up.float()
    horizontal[:,0,2::2,1::2]=torch.maximum(horizontal[:,0,2::2,1::2],down.float())
    walls=(F.max_pool2d(vertical,(3,1),stride=1,padding=(1,0))+F.max_pool2d(horizontal,(1,3),stride=1,padding=(0,1)))[:,0]>0
    key=(str(occupied.device),ny,nx)
    if key not in _DEPTH_TENSOR_GRIDS:
        ys=torch.arange(2*ny+1,device=occupied.device,dtype=torch.int32)
        query_x=2*torch.arange(nx,device=occupied.device,dtype=torch.float32)+1
        vertical_x=2*torch.arange(nx+1,device=occupied.device,dtype=torch.float32)
        dx2=(query_x[:,None]-vertical_x[None,:]).square()
        _DEPTH_TENSOR_GRIDS[key]=(ys,dx2)
    ys,dx2=_DEPTH_TENSOR_GRIDS[key]
    stripes=walls.transpose(1,2).contiguous();seed=ys[None,None]
    before=torch.where(stripes,seed,-1048576).cummax(-1).values
    after=-torch.where(stripes,-seed,-1048576).flip(-1).cummax(-1).values.flip(-1)
    dy=torch.minimum(seed-before,after-seed).float()
    dy=torch.where(stripes.any(-1,keepdim=True),dy,torch.inf)
    by_column=dy[:,:,1::2].square().transpose(1,2)
    # Odd-x wall points are horizontal-face midpoints. For an odd-x query,
    # only its own column can be the closest such midpoint; otherwise a
    # neighbouring endpoint (even x) on that segment is closer. Thus only
    # even-x columns require the horizontal min-plus reduction.
    group=max(1,20_000_000//(ny*nx*(nx+1)));parts=[]
    for z in range(0,nz,group):
        from_vertical=(by_column[z:z+group,:,None,::2]+dx2[None,None]).amin(-1)
        d2=torch.minimum(from_vertical,by_column[z:z+group,:,1::2])
        parts.append(d2.sqrt()*.5)
    result=torch.cat(parts,dim=0)
    return torch.where(occupied,result,0.)
