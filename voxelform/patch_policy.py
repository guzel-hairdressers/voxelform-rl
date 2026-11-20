"""Direct coherent neural masks. Gaussian latents are RL actions, not patch candidates."""
from dataclasses import dataclass,asdict
import math,time
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal,Bernoulli
from .actions import ExteriorFacades,exterior_air,STRUCTURE,neighbors_horizontal
from .fields import facade_depth,depth_badness,facade_depth_tensor
from .policy import Learner,Network
from .shadows import sync

MODES=('independent','smooth','correlated','combined')
_GAUSSIAN_KERNELS={}

@dataclass(frozen=True)
class PatchConfig:
    mode:str='combined'
    eligibility:str='facade'
    stride:tuple=(2,4,4)  # floors,y-metres,x-metres for fixed 1x1x3m cells
    sigma:tuple=(2.,2.,2.)  # index units: 6m vertical, 2m horizontal
    minimum_patch:int=1
    maximum_patches:int=0
    maximum_voids:int=0
    minimum_void_area:int=1
    batched:bool=False
    maximum_edit_voxels:int=0
    vertical_interior:bool=False
    allocation:str='probability'
    budget_fraction:float=.5
    budget_margin:int=2
    budget_reserve_fraction:float=0.
    facade_layers:int=1
    def __post_init__(self):
        object.__setattr__(self,'stride',tuple(self.stride));object.__setattr__(self,'sigma',tuple(self.sigma))
        if not isinstance(self.batched,bool) or not isinstance(self.vertical_interior,bool):raise ValueError('batch/vertical flags must be boolean')
        if isinstance(self.maximum_edit_voxels,bool) or int(self.maximum_edit_voxels)!=self.maximum_edit_voxels or self.maximum_edit_voxels<0:raise ValueError('Invalid edit limit')
        object.__setattr__(self,'maximum_edit_voxels',int(self.maximum_edit_voxels))
        if isinstance(self.facade_layers,bool) or self.facade_layers not in (1,2,3):raise ValueError('Facade layers must be 1, 2 or 3')
        object.__setattr__(self,'facade_layers',int(self.facade_layers))
        if self.allocation not in ('probability','quota','learned','feedback'):raise ValueError('Invalid edit allocation')
        if self.allocation!='probability' and self.mode not in ('correlated','combined'):raise ValueError('Workload allocation requires correlated masks')
        if not math.isfinite(self.budget_fraction) or not 0<self.budget_fraction<=1:raise ValueError('Invalid capacity fraction')
        if not math.isfinite(self.budget_reserve_fraction) or not 0<=self.budget_reserve_fraction<=1:raise ValueError('Invalid reserve fraction')
        if isinstance(self.budget_margin,bool) or int(self.budget_margin)!=self.budget_margin or self.budget_margin<0:raise ValueError('Invalid budget margin')
        if self.mode not in MODES or self.eligibility not in ('facade','depth'):raise ValueError('Unknown patch mode/eligibility')
        if len(self.stride)!=3 or any(isinstance(v,bool) or int(v)!=v or v<1 for v in self.stride):raise ValueError('Invalid coarse stride')
        if len(self.sigma)!=3 or any(not math.isfinite(v) or v<0 or v>8 for v in self.sigma):raise ValueError('Invalid Gaussian sigma')
        if any(isinstance(v,bool) or int(v)!=v for v in (self.minimum_patch,self.maximum_patches)) or self.minimum_patch<1 or self.maximum_patches<0:raise ValueError('Invalid patch limits')
        if isinstance(self.maximum_voids,bool) or int(self.maximum_voids)!=self.maximum_voids or not 0<=self.maximum_voids<=8 or isinstance(self.minimum_void_area,bool) or int(self.minimum_void_area)!=self.minimum_void_area or self.minimum_void_area<1:raise ValueError('Invalid interior opening limits')
        object.__setattr__(self,'stride',tuple(int(v) for v in self.stride))
        for key in ('minimum_patch','maximum_patches','maximum_voids','minimum_void_area'):object.__setattr__(self,key,int(getattr(self,key)))


def gaussian_field(field,sigma):
    """Differentiable separable Gaussian smoothing, CPU/MPS-native 1D/2D kernels."""
    batched=field.ndim==4
    out=field if batched else field[None]
    B,Z,Y,X=out.shape
    sz,sy,sx=(float(s) for s in sigma)
    def _kernel(s):
        key=(str(field.device),field.dtype,s)
        if key not in _GAUSSIAN_KERNELS:
            radius=max(1,int(math.ceil(3*s)))
            x=torch.arange(-radius,radius+1,device=field.device,dtype=field.dtype)
            k=torch.exp(-.5*(x/s).square())
            _GAUSSIAN_KERNELS[key]=k/k.sum()
        return _GAUSSIAN_KERNELS[key]
    if sz>0:
        k=_kernel(sz);r=len(k)//2
        t=out.permute(0,2,3,1).reshape(B*Y*X,1,Z)
        t=torch.cat((t[:,:,:1].expand(-1,-1,r),t,t[:,:,-1:].expand(-1,-1,r)),dim=2)
        out=F.conv1d(t,k.view(1,1,-1)).reshape(B,Y,X,Z).permute(0,3,1,2)
    if sy>0 or sx>0:
        t=out.reshape(B*Z,1,Y,X)
        if sy>0:
            k=_kernel(sy);r=len(k)//2
            t=torch.cat((t[:,:,:1,:].expand(-1,-1,r,-1),t,t[:,:,-1:,:].expand(-1,-1,r,-1)),dim=2)
            t=F.conv2d(t,k.view(1,1,-1,1))
        if sx>0:
            k=_kernel(sx);r=len(k)//2
            t=torch.cat((t[:,:,:,:1].expand(-1,-1,-1,r),t,t[:,:,:,-1:].expand(-1,-1,-1,r)),dim=3)
            t=F.conv2d(t,k.view(1,1,1,-1))
        out=t.reshape(B,Z,Y,X)
    return out if batched else out[0]


def components(mask):
    labels,n=ndi.label(mask,structure=np.ones((3,3,3),bool));sizes=np.bincount(labels.ravel())[1:]
    return dict(components=n,largestShare=float(sizes.max()/sizes.sum()) if n else 0.,
                singletonShare=float(sizes[sizes==1].sum()/max(sizes.sum(),1)),
                meanPatchSize=float(sizes.mean()) if n else 0.)


class PatchFacades(ExteriorFacades):
    def __init__(self,occupied,allowed,seed=123,config=None,depth_device=None):
        self.config=config or PatchConfig();self.depth_device=torch.device(depth_device) if depth_device is not None else None
        self._prev_occ=None;self._prev_air=None;self._empty_voids=None;self.plot_edge=None
        super().__init__(occupied,allowed,seed)
    @property
    def action_space(self):return 'neural_patch_'+self.config.eligibility+'_v1'
    def refresh(self):
        stamp=time.perf_counter()
        prev_occ=self._prev_occ;prev_air=self._prev_air
        super().refresh();self._weight_tensors={}
        if self.plot_edge is None or self.plot_edge.shape!=self.shape:
            self.plot_edge=self.allowed&neighbors_horizontal(~self.allowed,outside=True).any(0)
            self._empty_voids=None
        occ_z=np.flatnonzero(self.occupied.any((1,2)))
        z_mask=((self.occupied!=prev_occ)|(self.exterior_air!=prev_air)) if (prev_occ is not None and prev_occ.shape==self.shape) else None
        changed_z=np.flatnonzero(z_mask.any((1,2))) if z_mask is not None else None
        if self.depth_device is None:
            if changed_z is not None and getattr(self,'depth_numpy',None) is not None:
                if len(changed_z):
                    self.depth_numpy[changed_z]=facade_depth(self.occupied[changed_z],self.exterior_air[changed_z])
            else:
                self.depth_numpy=np.zeros(self.shape,dtype=np.float32)
                if len(occ_z):
                    self.depth_numpy[occ_z]=facade_depth(self.occupied[occ_z],self.exterior_air[occ_z])
            self.depth_tensor=None
        else:
            if changed_z is not None and getattr(self,'depth_tensor',None) is not None:
                if len(changed_z):
                    occ_sub=torch.as_tensor(self.occupied[changed_z],device=self.depth_device)
                    air_sub=torch.as_tensor(self.exterior_air[changed_z],device=self.depth_device)
                    self.depth_tensor[changed_z]=facade_depth_tensor(occ_sub,air_sub)
            else:
                self.depth_tensor=torch.zeros(self.shape,dtype=torch.float32,device=self.depth_device)
                if len(occ_z):
                    occ_sub=torch.as_tensor(self.occupied[occ_z],device=self.depth_device)
                    air_sub=torch.as_tensor(self.exterior_air[occ_z],device=self.depth_device)
                    self.depth_tensor[occ_z]=facade_depth_tensor(occ_sub,air_sub)
            self.depth_numpy=None
        if self.config.maximum_voids or self.config.minimum_void_area>1:
            if changed_z is not None and getattr(self,'old_voids',None) is not None and len(self.old_voids)==self.shape[0]:
                for z in changed_z:
                    zi=int(z);self.old_voids[zi]=self._floor_voids(self.occupied[zi],zi)
            else:
                uniform=bool((self.allowed==self.allowed[:1]).all())
                if uniform and self._empty_voids is None:
                    self._empty_voids=self._floor_voids(np.zeros_like(self.occupied[0]),0)
                if uniform:
                    el,ei,es=self._empty_voids
                    self.old_voids=[self._floor_voids(self.occupied[z],z) if self.occupied[z].any() else (el.copy(),list(ei),es.copy()) for z in range(self.shape[0])]
                else:
                    self.old_voids=[self._floor_voids(self.occupied[z],z) for z in range(self.shape[0])]
        else:
            self.old_voids=None
        self.facade_band=self.remove_skin.copy()
        for _ in range(self.config.facade_layers-1):self.facade_band|=self.occupied&neighbors_horizontal(self.facade_band).any(0)
        self.remove_weights=self.facade_band.astype(np.float32)
        self._remove_eligible=self.facade_band.copy()
        if self.config.eligibility=='depth':
            if self.depth_tensor is None:
                self.remove_weights=np.maximum(self.remove_weights,depth_badness(self.depth_numpy)*self.occupied);self._remove_eligible=self.remove_weights>0
            else:
                self._remove_eligible|=(self.depth_tensor>8).cpu().numpy();self.remove_weights=None
        self._prev_occ=self.occupied.copy();self._prev_air=self.exterior_air.copy()
        self.facade_seconds=time.perf_counter()-stamp
    def eligible(self,operation):return self._remove_eligible if operation==-1 else self.add_skin
    def weights(self,operation):
        if operation==1:return self.add_skin.astype(np.float32)
        if self.remove_weights is None:self.remove_weights=np.maximum(self.facade_band,depth_badness(self.depth_tensor.cpu().numpy()))
        return self.remove_weights
    def weights_tensor(self,operation,device):
        key=(operation,str(device))
        if key not in self._weight_tensors:
            if operation==-1 and self.depth_tensor is not None and self.depth_tensor.device.type==torch.device(device).type and (torch.device(device).index is None or self.depth_tensor.device.index==torch.device(device).index):
                weights=torch.as_tensor(self.facade_band,device=device,dtype=torch.float32)
                if self.config.eligibility=='depth':weights=torch.maximum(weights,depth_badness(self.depth_tensor))
            else:weights=torch.as_tensor(self.weights(operation),device=device)
            self._weight_tensors[key]=weights
        return self._weight_tensors[key]
    def _floor_voids(self,solid,z):
        labels,n=ndi.label(self.allowed[z]&~solid,structure=ndi.generate_binary_structure(2,1))
        outside=set(np.unique(labels[self.plot_edge[z]]));sizes=np.bincount(labels.ravel())
        ids=[i for i in range(1,n+1) if i not in outside]
        return labels,ids,sizes
    def _void_filter(self,mask,operation,check_only=False):
        if self.old_voids is None:return True if check_only else mask
        # A frozen lateral-facade removal only enlarges/merges existing air
        # components on each floor, or grows a bay touching the plot edge.
        # It cannot create a new isolated opening or split an existing one.
        if operation==-1 and not (mask&~self.remove_skin).any():return True if check_only else mask
        changed=self.occupied.copy();changed[mask]=operation==1;keep=mask.copy()
        for z in range(self.shape[0]):
            labels,ids,sizes=self._floor_voids(changed[z],z);old_labels,old_ids,old_sizes=self.old_voids[z]
            old_air=np.isin(old_labels,old_ids);existing=set(np.unique(labels[old_air]));existing.discard(0)
            accepted=[]
            for i in ids:
                minimum=self.config.minimum_void_area
                if i in existing:
                    overlaps=np.unique(old_labels[(labels==i)&old_air]);overlaps=overlaps[overlaps!=0]
                    if len(overlaps):minimum=min(minimum,int(old_sizes[overlaps].max()))
                if sizes[i]>=minimum:accepted.append(i)
            if self.config.maximum_voids:
                limit=max(self.config.maximum_voids,len(old_ids))
                accepted=sorted(accepted,key=lambda i:(i not in existing,-sizes[i]))[:limit]
            rejected=set(ids)-set(accepted)
            if rejected:
                if check_only or operation==1:return False if check_only else None
                keep[z]&=~np.isin(labels,list(rejected))
        return True if check_only else keep
    def validate_mask(self,mask,operation):
        if not super().validate_mask(mask,operation):return False
        if operation==-1 and (mask&~self.remove_skin).any():
            changed=self.occupied&~mask
            if (mask&~exterior_air(changed)).any():return False
        changed=self.occupied.copy();changed[mask]=operation==1
        return self._void_filter(mask,operation,check_only=True)
    def _patch_filter(self,mask,remaining):
        minimum=min(self.config.minimum_patch,int(remaining));labels,n=ndi.label(mask,structure=np.ones((3,3,3),bool))
        # Only the adaptive allocator gets a small final-adjustment phase;
        # courtyard size/topology and maximum region count still apply.
        if self.config.allocation=='feedback' and remaining<=self.config.minimum_patch*max(1,self.config.maximum_patches):minimum=1
        sizes=np.bincount(labels.ravel());ids=np.arange(1,n+1);ids=ids[sizes[ids]>=minimum]
        if self.config.maximum_patches:ids=ids[np.argsort(-sizes[ids])[:self.config.maximum_patches]]
        return mask&np.isin(labels,ids)
    def project_mask(self,generated,scores,operation,remaining):
        start=time.perf_counter();raw=np.asarray(generated,dtype=bool)&self.eligible(operation);original_count=int(raw.sum())
        for _ in range(4):
            mask=self._patch_filter(raw,remaining)
            action=super().project_mask(mask,scores,operation,remaining)
            if action is None:break
            mask=action.mask
            if operation==-1 and (mask&~self.remove_skin).any():
                reached=exterior_air(self.occupied&~mask);mask=mask&reached
            mask=self._void_filter(mask,operation)
            if mask is None:break
            mask=self._patch_filter(mask,remaining)
            if mask.sum()<min(2,int(remaining)):break
            if self.validate_mask(mask,operation):
                action.mask=mask;action.generated_voxels=original_count
                action.facade_only=not bool((mask&~self.remove_skin).any()) if operation==-1 else True
                self.projection_seconds=time.perf_counter()-start;return action
            raw=mask
        self.projection_seconds=time.perf_counter()-start;return None


class PatchNetwork(Network):
    def __init__(self):super().__init__();self.patch_log_std=nn.Parameter(torch.zeros(2))


class BudgetNetwork(PatchNetwork):
    def __init__(self):
        super().__init__()
        self.budget_encoder=nn.Linear(5,24)
        self.amount_head=nn.Linear(24,2)
        self.budget_critic=nn.Linear(24,1)
        self.amount_log_std=nn.Parameter(torch.full((2,),-1.4))
        nn.init.zeros_(self.amount_head.weight);nn.init.zeros_(self.amount_head.bias)
        nn.init.zeros_(self.budget_critic.weight);nn.init.zeros_(self.budget_critic.bias)
    def forward(self,observation,goal,budget):
        encoded=self.encoder(observation);floor=encoded.mean((-1,-2));context=torch.cat((floor.mean(0),floor.amax(0)))
        b=torch.tanh(self.budget_encoder(budget))
        conditioned=encoded+self.goal_encoder(goal)[None,:,None,None]+b[None,:,None,None]
        fields=self.mask_head(conditioned).permute(1,0,2,3)
        value=self.critic(torch.cat((context,goal))).squeeze()+self.budget_critic(b).squeeze()
        return fields,value,self.amount_head(b)


class PatchLearner(Learner):
    def __init__(self,device='cpu',seed=123,config=None):
        super().__init__(device,seed);base=self.model;self.config=config or PatchConfig()
        with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);self.model=(BudgetNetwork() if self.config.allocation=='learned' else PatchNetwork()).to(device)
        self.model.load_state_dict(base.state_dict(),strict=False)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3);self.raw_structure={};self.applied_structure={}
    def fields(self,env):
        self._env=env
        if self.config.allocation!='learned':return super().fields(env)
        operation=-1 if env.count>env.goal_count else 1
        remaining=abs(env.count-env.goal_count);budget=env.workload
        features=[remaining/env.capacity,budget.left(env.edits)/max(1,budget.limit),
            float(env.frontier.eligible(operation).sum())/env.capacity,budget.quota(remaining,env.edits)/env.capacity,float(operation)]
        fields,value,self._amount_mean=self.model(env.observation(),torch.tensor(env.goal_features,device=self.device),torch.tensor(features,device=self.device))
        return fields,value
    def allocation_counts(self,scores,legal,budget,training=False,greedy=False,retry_offset=0):
        """Count decoding is part of the action, never another spatial proposal."""
        batched=scores.ndim==4;batch=scores.shape[0] if batched else 1
        if self.config.allocation=='probability':
            counts=scores.reshape(batch,-1).sum(1).round().clamp(min=2).long()
            return torch.minimum(torch.minimum(counts,torch.as_tensor(budget,device=self.device)),legal.sum()),None,None
        env=self._env;quota=env.workload.quota(abs(env.count-env.goal_count),env.edits)
        if self.config.allocation in ('quota','feedback'):
            requested=math.ceil(quota/env.workload.retention) if self.config.allocation=='feedback' else quota
            counts=torch.full((batch,),requested,device=self.device,dtype=torch.long);logp=entropy=None
        else:
            channel=0 if self._operation==-1 else 1
            dist=Normal(self._amount_mean[channel],self.model.amount_log_std[channel].clamp(-2.5,0).exp(),validate_args=False)
            latent=dist.loc.detach().expand(batch) if greedy else dist.sample((batch,))
            counts=(quota*latent.clamp(-1.4,1.4).exp()).round().clamp(min=1).long()
            logp=dist.log_prob(latent) if training else None;entropy=dist.entropy() if training else None
        # Conditional retries use fresh sampled fields at smaller scales. There
        # is no mask scoring/ranking bank and no unsafe bits are synthesized.
        scales=(1.,1.,.75,.5,.35,.25,.125,.0625)
        scale=torch.tensor(scales[retry_offset:retry_offset+batch],device=self.device)
        counts=(counts*scale).round().clamp(min=1).long()
        return torch.minimum(torch.minimum(counts,torch.as_tensor(budget,device=self.device)),legal.sum()),logp,entropy
    def draw(self,field,weights,remaining,training,greedy,retry_offset=0):
        """Return raw neural mask and correct sampling score/entropy, before topology."""
        c=self.config;legal=weights>0;channel=0 if self._operation==-1 else 1
        if c.mode in ('independent','smooth'):
            mean=gaussian_field(field,c.sigma) if c.mode=='smooth' else field
            p=(mean.sigmoid()*weights).clamp(1e-6,1-1e-6);dist=Bernoulli(probs=p[legal],validate_args=False)
            if greedy:
                k=max(min(2,remaining,int(legal.sum())),min(remaining,int(legal.sum()),int(p[legal].sum().round().detach().cpu())))
                selected=torch.zeros_like(p,dtype=torch.bool);ids=p.masked_fill(~legal,-1).flatten().topk(k).indices;selected.flatten()[ids]=True
            else:
                selected=torch.zeros_like(p,dtype=torch.bool);selected[legal]=dist.sample().bool()
            logp=dist.log_prob(selected[legal].float()).sum()/math.sqrt(int(legal.sum())) if training else None
            entropy=dist.entropy().mean() if training else None
            scores=p
        else:
            # All action decoding depends on the sampled latent and the frozen
            # environment. In particular the area k is decoded from this draw,
            # never directly from a separate parameter-dependent neural mean.
            coarse=tuple(max(1,math.ceil(n/s)) for n,s in zip(field.shape,c.stride))
            kernel=tuple(min(n,s) for n,s in zip(field.shape,c.stride))
            mean=F.avg_pool3d(field[None,None],kernel_size=kernel,stride=c.stride,ceil_mode=True,count_include_pad=False)[0,0]
            std=self.model.patch_log_std[channel].clamp(-2,1).exp();dist=Normal(mean,std,validate_args=False)
            latent=mean.detach() if greedy else dist.sample()
            decoded=F.interpolate(latent[None,None],size=field.shape,mode='trilinear',align_corners=False)[0,0]
            if c.mode=='combined':decoded=gaussian_field(decoded,c.sigma)
            if c.vertical_interior and self._operation==-1:
                interior=self._interior_tensor
                decoded=torch.where(interior,decoded.mean(0,keepdim=True).expand_as(decoded),decoded)
            scores=decoded.sigmoid()*weights
            n=int(legal.sum());k=max(min(2,remaining,n),min(remaining,n,int(scores[legal].sum().round().detach().cpu())))
            selected=torch.zeros_like(legal);ids=scores.masked_fill(~legal,-1).flatten().topk(k).indices;selected.flatten()[ids]=True
            logp=dist.log_prob(latent).sum()/math.sqrt(latent.numel()) if training else None
            entropy=dist.entropy().mean() if training else None
        if c.allocation!='probability':
            if c.mode not in ('correlated','combined'):raise ValueError('Workload allocation requires a correlated sampler')
            counts,amount_logp,amount_entropy=self.allocation_counts(scores,legal,remaining,training,greedy,retry_offset)
            k=int(counts[0].detach().cpu());selected=torch.zeros_like(legal)
            selected.flatten()[scores.masked_fill(~legal,-1).flatten().topk(k).indices]=True
            if amount_logp is not None:logp=logp+amount_logp[0];entropy=entropy+amount_entropy
        return selected,logp,entropy,scores
    def _choose_batched(self,env,training):
        self.last_attempt_recorded=False;self.projection_seconds=0.;self.generation_attempts=0
        self.sampling_seconds=0.;self.transfer_seconds=0.
        operation=-1 if env.count>env.goal_count else 1;self._operation=operation
        if not isinstance(env.frontier,PatchFacades) or env.frontier.config!=self.config:raise ValueError('Patch policy and environment configurations differ')
        eligible=env.frontier.eligible(operation);n=int(eligible.sum());remaining=abs(env.count-env.goal_count)
        if not n:self.last_rejection='no_admissible_edits';return None
        weights=env.frontier.weights_tensor(operation,self.device)
        if self.config.maximum_edit_voxels:remaining=min(remaining,self.config.maximum_edit_voxels)
        sync(self.device);start=time.perf_counter();action=None;used_logps=[]
        with torch.set_grad_enabled(training):
            fields,value=self.fields(env);sync(self.device);self.model_seconds=time.perf_counter()-start
            field=fields[0 if operation==-1 else 1];kernel=tuple(min(a,b) for a,b in zip(field.shape,self.config.stride))
            mean=F.avg_pool3d(field[None,None],kernel_size=kernel,stride=self.config.stride,ceil_mode=True,count_include_pad=False)[0,0]
            dist=Normal(mean,self.model.patch_log_std[0 if operation==-1 else 1].clamp(-2,1).exp(),validate_args=False)
            legal=weights>0;budget=(env.occupied.sum()-env.goal_count).abs()
            if self.config.maximum_edit_voxels:budget=budget.clamp_max(self.config.maximum_edit_voxels)
            interior=torch.as_tensor(eligible&~env.frontier.remove_skin,device=self.device) if self.config.vertical_interior and operation==-1 else None
            for offset,batch_size in ((0,1),(1,7)):
                sampling=time.perf_counter();latent=dist.sample((batch_size,))
                decoded=F.interpolate(latent[:,None],size=env.shape,mode='trilinear',align_corners=False)[:,0]
                if self.config.mode=='combined':decoded=gaussian_field(decoded,self.config.sigma)
                if interior is not None:decoded=torch.where(interior[None],decoded.mean(1,keepdim=True).expand_as(decoded),decoded)
                scores=decoded.sigmoid()*weights[None]
                counts,amount_logps,amount_entropy=self.allocation_counts(scores,legal,budget,training,retry_offset=offset)
                largest=min(remaining,n)
                ids=scores.masked_fill(~legal[None],-1).flatten(1).topk(largest,dim=1).indices
                active=torch.arange(largest,device=self.device)[None]<counts[:,None]
                masks=torch.zeros((batch_size,int(np.prod(env.shape))),device=self.device,dtype=torch.bool).scatter_(1,ids,active).reshape(batch_size,*env.shape)
                if training:
                    logps=dist.log_prob(latent).flatten(1).sum(1)/math.sqrt(mean.numel())
                    if amount_logps is not None:logps=logps+amount_logps
                sync(self.device);self.sampling_seconds+=time.perf_counter()-sampling;transfer=time.perf_counter()
                cpu=masks.detach().cpu().numpy();self.transfer_seconds+=time.perf_counter()-transfer
                for i in range(batch_size):
                    self.generation_attempts=offset+i+1
                    if int(cpu[i].sum())>remaining:raise RuntimeError('Batched decoder exceeded FAR budget')
                    action=env.frontier.project_mask(cpu[i],None,operation,remaining)
                    self.projection_seconds+=env.frontier.projection_seconds
                    if action is not None:break
                if training:used_logps.append(logps[:i+1].sum())
                self.raw_structure=components(cpu[i])
                if action is not None:break
            if training:
                self.tape.append([torch.stack(used_logps).sum(),value,dist.entropy().mean()+(amount_entropy if amount_entropy is not None else 0),None]);self.last_attempt_recorded=True
        if action is not None:
            self.applied_structure=components(action.mask);action.mask_source='neural_'+self.config.mode
            action.raw_structure=self.raw_structure;action.applied_structure=self.applied_structure
        sync(self.device);self.generation_seconds=time.perf_counter()-start;self.last_rejection=None if action is not None else 'mask_generation_failed'
        return action
    def choose(self,env,training=True,greedy=False):
        if self.config.batched and not greedy and self.config.mode in ('correlated','combined'):return self._choose_batched(env,training)
        self.last_attempt_recorded=False;self.projection_seconds=0.;self.generation_attempts=0;self._operation=-1 if env.count>env.goal_count else 1
        if not isinstance(env.frontier,PatchFacades) or env.frontier.config!=self.config:raise ValueError('Patch policy and environment configurations differ')
        legal=env.frontier.eligible(self._operation)
        if not legal.any():self.last_rejection='no_admissible_edits';return None
        weights=env.frontier.weights_tensor(self._operation,self.device);remaining=abs(env.count-env.goal_count)
        if self.config.maximum_edit_voxels:remaining=min(remaining,self.config.maximum_edit_voxels)
        if self.config.vertical_interior:self._interior_tensor=torch.as_tensor(legal&~env.frontier.remove_skin,device=self.device)
        sync(self.device);start=time.perf_counter();logps=[];entropies=[];action=None
        with torch.set_grad_enabled(training):
            fields,value=self.fields(env);sync(self.device);self.model_seconds=time.perf_counter()-start
            field=fields[0 if self._operation==-1 else 1]
            for attempt in range(8):
                raw,logp,entropy,scores=self.draw(field,weights,remaining,training,greedy and attempt==0,attempt)
                if training:logps.append(logp);entropies.append(entropy)
                cpu=raw.detach().cpu().numpy();self.raw_structure=components(cpu)
                action=env.frontier.project_mask(cpu,scores.detach().cpu().numpy(),self._operation,remaining)
                self.projection_seconds+=env.frontier.projection_seconds;self.generation_attempts=attempt+1
                if action is not None:break
            if training:
                self.tape.append([torch.stack(logps).sum(),value,torch.stack(entropies).mean(),None]);self.last_attempt_recorded=True
        if action is not None:
            self.applied_structure=components(action.mask);action.mask_source='neural_'+self.config.mode
            action.raw_structure=self.raw_structure;action.applied_structure=self.applied_structure
        sync(self.device);self.generation_seconds=time.perf_counter()-start;self.last_rejection=None if action is not None else 'mask_generation_failed'
        return action
    def state(self):
        return dict(format='voxelform.patch.v1',config=asdict(self.config),model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
                    updates=self.updates,directMaskUpdates=self.direct_updates,importedUpdates=self.imported_updates,migration=self.migration)
    def load(self,state):
        if state.get('format')=='voxelform.patch.v1':
            config=PatchConfig(**state['config']);replacement=PatchLearner(str(self.device),config=config)
            replacement.model.load_state_dict(state['model']);replacement.optimizer.load_state_dict(state['optimizer'])
            replacement.updates=int(state['updates']);replacement.direct_updates=int(state['directMaskUpdates']);replacement.imported_updates=int(state['importedUpdates']);replacement.migration=state.get('migration')
        else:
            old=Learner(str(self.device));old.load(state);replacement=PatchLearner(str(self.device),config=self.config)
            replacement.model.load_state_dict(old.model.state_dict(),strict=False);replacement.updates=old.updates;replacement.imported_updates=old.updates
            replacement.migration='Retained dense encoder/head/critic as initialization; reset optimizer for new sampling semantics'
        self.__dict__.update(replacement.__dict__)
