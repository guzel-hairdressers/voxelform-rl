"""Research action space: neural remove/grow/exchange/finish from step zero.

Independent of live FAR-only episodes and checkpoint semantics. No phase gate
or direction chosen from FAR. Masks use frozen eligibility and whole-edit guards.
"""
from dataclasses import asdict
import math,time
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical,Normal
from .policy import Learner,Network
from .patch_policy import PatchConfig,gaussian_field
from .redistribution import Exchange,growth_mask
from .actions import STRUCTURE,exterior_air

OPERATIONS=('remove','grow','exchange','finish')

def valid_move(env,move,maximum_buildings=2):
    f=env.frontier;r,a=move.remove,move.add
    if move.version!=f.version or r.shape!=env.shape or a.shape!=env.shape:return False
    if not (r.any() or a.any()) or (r&a).any():return False
    if (r&~f.eligible(-1)).any() or (a&~growth_mask(env)).any():return False
    for mask in (r,a):
        if not mask.any():continue
        labels,n=ndi.label(mask,np.ones((3,3,3),bool))
        if f.config.maximum_patches and n>f.config.maximum_patches:return False
        minimum=1 if f.config.allocation=='feedback' and int(mask.sum())<=f.config.minimum_patch*max(1,f.config.maximum_patches) else f.config.minimum_patch
        if (np.bincount(labels.ravel())[1:]<minimum).any():return False
    changed=(f.occupied&~r)|a;labels,n=ndi.label(changed,STRUCTURE)
    grounded=np.unique(labels[0]);grounded=grounded[grounded!=0]
    if not 0<n<=maximum_buildings or len(grounded)!=n:return False
    air=exterior_air(changed)
    if (r&~air).any() or (f.exterior_air&~a&~air).any():return False
    if f.old_voids is not None:
        for z in range(env.nz):
            lab,ids,sizes=f._floor_voids(changed[z],z);oldlab,oldids,oldsizes=f.old_voids[z]
            if f.config.maximum_voids and len(ids)>max(f.config.maximum_voids,len(oldids)):return False
            for i in ids:
                prior=[oldsizes[j] for j in np.unique(oldlab[lab==i]) if j in oldids]
                if sizes[i]<min([f.config.minimum_void_area,*prior]):return False
    return True

def project(env,masks,scores,operation):
    """Only prune neural selections. Exchanges balance; pure moves change FAR."""
    r,a=masks.copy();f=env.frontier
    for _ in range(4):
        r&=exterior_air(f.occupied&~r)
        r=f._patch_filter(r,int(r.sum())) if r.any() else r
        a=f._patch_filter(a,int(a.sum())) if a.any() else a
        if operation=='exchange':
            k=min(int(r.sum()),int(a.sum()))
            if not k:return None
            for mask,confidence in ((r,scores[0]),(a,scores[1])):
                if mask.sum()>k:
                    ids=np.flatnonzero(mask);keep=ids[np.argsort(-confidence.ravel()[ids],kind='stable')[:k]]
                    mask[:]=False;mask.ravel()[keep]=True
        if operation=='remove' and not r.any() or operation=='grow' and not a.any():return None
        move=Exchange(r.copy(),a.copy(),f.version)
        if valid_move(env,move):return move
    return None

def commit(env,move,audit=False):
    if not valid_move(env,move):raise ValueError('Invalid/stale unified edit')
    f=env.frontier;f.occupied[move.remove]=False;f.occupied[move.add]=True;f.version+=1;f.refresh()
    env.occupied.masked_fill_(torch.as_tensor(move.remove,device=env.device),False)
    env.occupied.masked_fill_(torch.as_tensor(move.add,device=env.device),True)
    env.count+=int(move.add.sum())-int(move.remove.sum());env.edits+=1
    env.last_changed_voxels=int(move.add.sum()+move.remove.sum());env.voxels_edited+=env.last_changed_voxels
    env.last_operation='exchange' if move.remove.any() and move.add.any() else 'remove' if move.remove.any() else 'grow'
    env.termination=None  # FAR is acceptance, not an automatic episode boundary.
    env.update_depth()
    if audit:env.evaluate()

class UnifiedNetwork(Network):
    def __init__(self):
        super().__init__(10)
        self.goal_encoder=nn.Linear(5,24)
        self.critic=nn.Sequential(nn.Linear(53,32),nn.SiLU(),nn.Linear(32,1))
        self.operation_head=nn.Linear(53,4)
        self.patch_log_std=nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.operation_head.weight);nn.init.zeros_(self.operation_head.bias)
        # Learnable initial tendency toward FAR, never an availability/phase
        # rule. Exchange and upward growth remain possible above target.
        with torch.no_grad():
            self.operation_head.weight[0,48:50]=torch.tensor([8.,-8.])
            self.operation_head.weight[1,48:50]=torch.tensor([-8.,8.])
    def forward(self,observation,goal):
        encoded=self.encoder(observation);floor=encoded.mean((-1,-2));context=torch.cat((floor.mean(0),floor.amax(0),goal))
        fields=self.mask_head(encoded+self.goal_encoder(goal)[None,:,None,None]).permute(1,0,2,3)
        return fields,self.critic(context).squeeze(),self.operation_head(context)

class UnifiedLearner(Learner):
    def __init__(self,device='cpu',seed=123,config=None):
        super().__init__(device,seed);self.config=config or PatchConfig(mode='combined',allocation='feedback')
        if self.config.mode not in ('combined','correlated'):raise ValueError('Unified policy requires correlated masks')
        with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);self.model=UnifiedNetwork().to(device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)
    def observation(self,e):
        # Cache immutable external exposure once. Keep direct sun and diffuse
        # facade access separate; both edit eligibility fields are explicit.
        if not hasattr(e,'external_potential'):
            e.precompute_external()
            normals=e.suns.new_tensor(((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)))
            w=(e.suns@normals.T).clamp_min(0)*e.suns.new_tensor((3.,3.,3.,3.,1.,1.))[None]
            e.external_potential=(e.static_receivers.values.reshape(len(e.suns),6,*e.shape).float()*w[:,:,None,None,None]).sum((0,1))/w.sum().clamp_min(1)
        remove=e.frontier.weights_tensor(-1,self.device)
        grow=torch.as_tensor(growth_mask(e),device=self.device,dtype=torch.float32)
        return torch.cat((e.observation(include_depth=True),remove[:,None],grow[:,None],e.external_potential[:,None],e.facade_light[:,None]),1)
    def forward(self,e):
        limit=getattr(e,'unified_limit',1);spent=getattr(e,'unified_actions',e.edits)
        goal=torch.tensor([*e.goal_features,max(0,limit-spent)/max(1,limit),spent/max(1,limit)],device=self.device,dtype=torch.float32)
        return self.model(self.observation(e),goal)
    def fields(self,e):
        fields,value,_=self.forward(e);return fields,value
    def legal_operations(self,e):
        r=e.frontier.eligible(-1).any();a=growth_mask(e).any()
        return torch.tensor([r,a,r and a,e.far_satisfied],device=self.device,dtype=torch.bool)
    def choose(self,e,training=True,force_operation=None,retries=8):
        start=time.perf_counter();self.last_attempt_recorded=False;self.last_rejection=None
        with torch.set_grad_enabled(training):
            fields,value,logits=self.forward(e);legal=self.legal_operations(e)
            if not bool(legal.any()):self.last_rejection='no_admissible_edits';return None
            distribution=Categorical(logits=logits.masked_fill(~legal,float('-inf')))
            if force_operation is None:op=distribution.sample()
            else:
                if training:raise ValueError('Forced operations are geometry probes, not on-policy training')
                op=torch.tensor(OPERATIONS.index(force_operation),device=self.device)
                if not bool(legal[op]):raise ValueError('Requested operation is unavailable')
            operation=OPERATIONS[int(op.cpu())];self.last_operation=operation
            logp=distribution.log_prob(op);entropy=distribution.entropy()
            move=None;logs=[]
            if operation!='finish':
                weights=torch.stack((e.frontier.weights_tensor(-1,self.device),torch.as_tensor(growth_mask(e),device=self.device,dtype=torch.float32)))
                kernel=tuple(min(n,s) for n,s in zip(e.shape,self.config.stride))
                mean=F.avg_pool3d(fields[:,None],kernel,stride=self.config.stride,ceil_mode=True,count_include_pad=False)[:,0]
                consumed=[0] if operation=='remove' else [1] if operation=='grow' else [0,1]
                dist=Normal(mean[consumed],self.model.patch_log_std[consumed].clamp(-2,1).exp()[:,None,None,None],validate_args=False)
                remaining=abs(e.count-e.goal_count);left=max(1,getattr(e,'unified_limit',8)-getattr(e,'unified_actions',e.edits)-4)
                toward=(operation=='remove' and e.count>e.goal_count) or (operation=='grow' and e.count<e.goal_count)
                amount=max(4,math.ceil(remaining/left/.65)) if toward else max(4,round(e.target_count*.025))
                if toward:amount=min(amount,remaining)
                if self.config.maximum_edit_voxels:amount=min(amount,self.config.maximum_edit_voxels)
                for attempt in range(retries):
                    latent=dist.sample();decoded=F.interpolate(latent[:,None],size=e.shape,mode='trilinear',align_corners=False)[:,0]
                    if self.config.mode=='combined':decoded=gaussian_field(decoded,self.config.sigma)
                    scores=torch.zeros_like(fields);masks=torch.zeros(fields.shape,device=self.device,dtype=torch.bool)
                    count=max(1,round(amount*(1.,1.,.75,.5,.35,.25,.125,.0625)[min(attempt,7)]))
                    for index,channel in enumerate(consumed):
                        field=decoded[index]
                        if channel==0 and self.config.vertical_interior:
                            interior=torch.as_tensor(e.frontier.eligible(-1)&~e.frontier.facade_band,device=self.device)
                            field=torch.where(interior,field.mean(0,keepdim=True).expand_as(field),field)
                        scores[channel]=field.sigmoid()*weights[channel];available=int((weights[channel]>0).sum().cpu());n=min(count,available)
                        ids=scores[channel].masked_fill(weights[channel]<=0,-1).flatten().topk(n).indices;masks[channel].flatten()[ids]=True
                    if training:logs.append(dist.log_prob(latent).sum()/math.sqrt(latent.numel()))
                    move=project(e,masks.cpu().numpy(),scores.detach().cpu().numpy(),operation)
                    if move is not None:break
                if training:logp=logp+torch.stack(logs).sum();entropy=entropy+dist.entropy().mean()
                if move is None:self.last_rejection='mask_generation_failed'
            if training:self.tape.append([logp,value,entropy,None]);self.last_attempt_recorded=True
        self.generation_seconds=time.perf_counter()-start
        return move
    def state(self):
        return dict(format='voxelform.unified.v1',config=asdict(self.config),model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates,directMaskUpdates=self.direct_updates,importedUpdates=0)
    def load(self,state):
        if state.get('format')!='voxelform.unified.v1':raise ValueError('Unified action checkpoint required')
        other=UnifiedLearner(str(self.device),config=PatchConfig(**state['config']));other.model.load_state_dict(state['model']);other.optimizer.load_state_dict(state['optimizer']);other.updates=int(state['updates']);other.direct_updates=int(state['directMaskUpdates']);self.__dict__.update(other.__dict__)
