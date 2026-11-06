"""Surface-anchored volume edits and explicit proxy evaluation."""
import math,random,time
import numpy as np
import torch
from .shadows import DirectShadows,BoundaryShadows,DepthShadows,SelfShadowMaps,ReceiverCache,sun_path,sync
from .fields import grid_points,noise_prior,sky_boundary,diffuse_access,receivers,usable_depth_proxy


def context_boxes(nx,ny,count=16,seed=123,height=24.):
    rng=random.Random(seed);rows=[]
    for k in range(count):
        side=k%4;hx=rng.uniform(4,9);hy=rng.uniform(4,9);yaw=rng.uniform(-.6,.6);radius=math.hypot(hx,hy)+rng.uniform(3,14)
        x,y=(rng.uniform(-10,nx+10),-radius) if side==0 else (nx+radius,rng.uniform(-10,ny+10)) if side==1 else (rng.uniform(-10,nx+10),ny+radius) if side==2 else (-radius,rng.uniform(-10,ny+10))
        h=rng.uniform(7,max(8,height*1.8));rows.append((x,y,h/2,hx,hy,h/2,math.cos(yaw),math.sin(yaw)))
    return np.asarray(rows,dtype=np.float32)


class Environment:
    def __init__(self,shape=(6,24,32),seed=123,device='cpu',shadow_method='boundary',sun_count=8,pitch=1.,plan='rectangle'):
        self.shape=tuple(shape);self.device=torch.device(device);self.seed=seed;self.pitch=pitch;self.sun_count=sun_count
        self.nz,self.ny,self.nx=self.shape;self.plan=plan
        self.points=grid_points(shape,self.device);self.x=self.points[...,0];self.y=self.points[...,1];self.z=self.points[...,2]/3
        allowed=torch.ones(shape,dtype=torch.bool,device=self.device)
        if plan=='triangle':allowed&=self.x/self.nx+self.y/self.ny<1.
        elif plan=='courtyard':allowed&=~((self.x>.65*self.nx)&(self.y>.65*self.ny))
        elif plan=='branch':allowed&=(self.y<self.ny*.55)|((self.x>.35*self.nx)&(self.x<self.nx*.65))
        self.allowed=allowed;self.total_area=allowed.sum();self.boxes=context_boxes(self.nx,self.ny,16,seed,self.nz*3)
        self.suns_np=sun_path(sun_count);self.suns=torch.tensor(self.suns_np,dtype=torch.float32,device=self.device)
        self.bounds=((0,0,0),(self.nx,self.ny,self.nz*3))
        cls={'boundary':BoundaryShadows,'depth':DepthShadows,'direct':DirectShadows}[shadow_method]
        self.static=cls(self.boxes,self.suns_np,self.bounds,pitch,self.device) if shadow_method!='direct' else cls(self.boxes,self.suns_np,self.device)
        self.static_receivers=ReceiverCache(self.static,6*int(np.prod(shape)))
        self.dynamic=SelfShadowMaps(shape,self.suns_np,pitch,device=self.device)
        self.sky_boundary=sky_boundary(shape,self.boxes,self.device)
        road=((-15.,-8.,0.),(self.nx+15.,-8.,0.),1.)
        rail=((-11.,-15.,0.),(-11.,self.ny+15.,0.),.6)
        self.noise_raw=noise_prior(self.points,[road,rail]);self.noise=self.noise_raw/(self.noise_raw+.12)
        self.history=[];self.reset()

    def reset(self):
        self.occupied=self.allowed.clone();self.history=[];self.edits=0
        self.evaluate();return self.observation()

    def evaluate(self):
        stamp=time.perf_counter();self.light=diffuse_access(self.occupied,self.sky_boundary,iterations=48)
        self.usable=usable_depth_proxy(self.occupied,self.light,boundary=self.sky_boundary)
        ids,p,n,cells,faces=receivers(self.occupied)
        outside,new=self.static_receivers.query(ids,p);self.dynamic.update(self.occupied)
        own=self.dynamic.query(p,normals=n);front=(self.suns@n.T)>1e-6
        self.visible=outside&own&front;self.receiver_data=(ids,p,n,cells,faces)
        side=faces<4;roof=faces==4;areas=torch.where(side,3.,1.)
        weights=(self.suns@n.T).clamp_min(0)*areas[None]
        sunlight=(self.visible.float()*weights).sum()/weights.sum().clamp_min(1)
        roof_sun=self.visible[:,roof].float().mean(0) if bool(roof.any()) else torch.zeros(0,device=self.device)
        roof_opportunity=(roof_sun>.4).float().sum()/(self.nx*self.ny)
        retention=self.occupied.sum()/self.total_area;usable=self.usable.sum()/self.total_area
        exposure=self.noise.flatten()[cells[side]].mean() if bool(side.any()) else torch.zeros((),device=self.device)
        quality=.70*usable+.18*sunlight+.12*roof_opportunity-.08*exposure-2*torch.relu(.45-retention)
        self.quality=quality;self.values=dict(quality=quality,retention=retention,potentialUsableFraction=usable,
            directSunProxy=sunlight,roofOpportunity=roof_opportunity,noisePrior=exposure,
            occupiedFloorArea=self.occupied.sum(),potentialUsableArea=self.usable.sum())
        sync(self.device)
        self.last_new_receivers=new;self.last_receiver_count=len(ids);self.last_eval_seconds=time.perf_counter()-stamp
        return quality

    def observation(self):
        # All channels stay on the chosen device; no per-voxel Python work.
        return torch.stack((self.occupied.float(),self.light,self.noise,self.allowed.float(),self.z/self.nz),dim=1)

    def candidates(self,count=24,params=None,kind=None):
        """Coherent tools, not architectural recipes or isolated voxel flips.

        Rectangle profiles are a bounded baseline; learned free profiles are a
        later comparison. Restoration currently means undo of the latest edit.
        """
        k=count;d=self.device
        params=torch.rand((k,7),device=d) if params is None else torch.as_tensor(params,device=d)
        kind=torch.randint(0,5,(k,),device=d) if kind is None else torch.as_tensor(kind,device=d)
        a=params[:,0]*.8+.05;b=params[:,1]*.8+.05;w=params[:,2]*.45+.12;h=params[:,3]*.45+.12
        low=(params[:,4]*self.nz).floor();high=(low+1+(params[:,5]*(self.nz-low)).floor()).clamp_max(self.nz)
        depth=params[:,6]*.55+.12
        x=self.x[None]/self.nx;y=self.y[None]/self.ny;z=self.z[None]
        side=((z>=low[:,None,None,None])&(z<high[:,None,None,None]))
        mask=torch.where((kind<2)[:,None,None,None],
            (y>=b[:,None,None,None])&(y<b[:,None,None,None]+h[:,None,None,None])&torch.where((kind==0)[:,None,None,None],x<depth[:,None,None,None],x>1-depth[:,None,None,None]),
            (x>=a[:,None,None,None])&(x<a[:,None,None,None]+w[:,None,None,None])&torch.where((kind==2)[:,None,None,None],y<depth[:,None,None,None],y>1-depth[:,None,None,None]))&side
        roof=(kind==4)[:,None,None,None]
        mask=torch.where(roof,(x>=a[:,None,None,None])&(x<a[:,None,None,None]+w[:,None,None,None])&
            (y>=b[:,None,None,None])&(y<b[:,None,None,None]+h[:,None,None,None])&(z>=low[:,None,None,None]),mask)
        changed=mask&self.occupied[None];valid=changed.flatten(1).sum(1)>=4
        features=torch.cat((torch.nn.functional.one_hot(kind,5).float(),params),dim=1)
        # Stop action and exact undo are explicit, not building-state labels.
        zero=torch.zeros_like(self.occupied);masks=torch.cat((zero[None],changed,zero[None]),dim=0)
        extra=torch.zeros((2,12),device=d);extra[0,0]=-1;extra[1,0]=-2
        if self.history:masks[-1]=self.history[-1]
        features=torch.cat((extra[:1],features,extra[1:]),dim=0)
        valid=torch.cat((torch.ones(1,dtype=torch.bool,device=d),valid,torch.tensor([bool(self.history)],device=d)))
        return masks,features,valid

    def apply(self,index,masks):
        if index==0:return True
        if index==len(masks)-1:
            if not self.history:return False
            self.occupied|=self.history.pop()
        else:
            removed=self.occupied&masks[index];self.occupied&=~removed;self.history.append(removed)
        self.edits+=1;self.evaluate();return False

    def metrics(self):
        sync(self.device)
        result={k:float(v.detach().cpu()) for k,v in self.values.items()}
        result.update(edits=self.edits,receivers=self.last_receiver_count,newReceivers=self.last_new_receivers,
            evaluationSeconds=self.last_eval_seconds,device=str(self.device),fieldStatus='uncalibrated_massing_proxies')
        return result
