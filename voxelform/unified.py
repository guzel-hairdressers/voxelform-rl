"""Unified remove/grow/exchange/finish policy, available from step zero.

The active preview enables linked whole-floor copying and net FAR guidance.
The default floor_copy=False preserves the original roof-patch research decoder.
Masks use frozen eligibility and whole-edit guards, without a phase gate.
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

def copy_plates(env):
    """One complete highest floor per grounded mass, with frozen eligibility.

    Plates are an explicit growth primitive, not a bank of sampled edit masks.
    Their geometry comes directly from the current building.
    """
    f=env.frontier;labels,n=ndi.label(f.occupied,STRUCTURE);plates=[]
    for i in range(1,n+1):
        cells=labels==i;levels=np.flatnonzero(cells.any((1,2)))
        if not len(levels):continue
        z=int(levels[-1])+1
        if z>=env.nz:continue
        plate=cells[z-1]
        if plate.sum()<max(4,f.config.minimum_patch):continue
        if f.config.maximum_edit_voxels and plate.sum()>f.config.maximum_edit_voxels:continue
        if (plate&(~f.allowed[z]|f.occupied[z]|~f.exterior_air[z])).any():continue
        mask=np.zeros(env.shape,bool);mask[z]=plate;plates.append(mask)
    return plates

def copy_eligibility(env):
    plates=copy_plates(env)
    return np.logical_or.reduce(plates) if plates else np.zeros(env.shape,bool)

def supported_removal(env,remove):
    """Prune selected cuts; never invent extra cuts to support an upper floor."""
    r=remove.copy();solid=env.frontier.occupied
    labels=getattr(env,'copy_labels',np.zeros(env.shape,dtype=np.int64))
    # A copied floor remains coupled to its source at each footprint cell. Only
    # discard partial selections; projection never expands a generated mask.
    for _ in range(env.nz+1):
        before=int(r.sum())
        for z in range(env.nz-2,-1,-1):r[z]&=~solid[z+1]|r[z+1]
        if labels.any():
            total=np.bincount(labels[solid]);selected=np.bincount(labels[r],minlength=len(total))
            complete=selected==total;complete[0]=True
            r&=complete[labels]
        if int(r.sum())==before:break
    return r

def balance_copy(env,remove,addition,scores):
    """Prune a neural cut prefix with the exact source/copy area cost.

    Cutting a source cell removes both that cell and its prospective copy.
    Linked stacks add further atomic costs. Search a quota on the ONE generated
    ranking, never sample or score a library of alternative geometric masks.
    """
    def result(r):
        a=addition.copy();a[1:]&=~r[:-1]
        return a,env.count-int(r.sum())+int(a.sum())
    a,after=result(remove);low,high=env.acceptable_counts
    lower=low if env.count>=low else env.count+1
    upper=high if env.count<=high else env.count-1
    if lower<=after<=upper:return remove,a
    if after>upper:return remove,a # Not enough selected cuts; do not invent any.
    target=env.goal_count
    if env.count<low:target=min(target,env.count+max(1,int(addition.sum())//2))
    target=max(lower,min(upper,target))
    ids=np.flatnonzero(remove);ids=ids[np.argsort(-scores.ravel()[ids],kind='stable')]
    lo,hi=0,len(ids);best=np.zeros(env.shape,bool)
    while lo<=hi:
        n=(lo+hi)//2;r=np.zeros(env.shape,bool);r.ravel()[ids[:n]]=True
        r=supported_removal(env,r);_,count=result(r)
        if count>=target:best=r;lo=n+1
        else:hi=n-1
    return best,result(best)[0]

def project_floor_copy(env,masks,scores,operation,net_guided):
    f=env.frontier;r,a=masks.copy()
    if r.any():
        r=supported_removal(env,r)
        previous=f.components
        try:
            f.components=max(2,previous)
            action=f.project_mask(r,scores[0],-1,int(r.sum()))
        finally:f.components=previous
        if action is None:return None
        r=supported_removal(env,action.mask)
    if net_guided and operation=='exchange':r,a=balance_copy(env,r,a,scores[0])
    # A copied floor and its source are edited together on this action. Preserve
    # the complete resulting source plate, including existing/new courtyard holes.
    if a.any():
        for z in range(1,env.nz):a[z]&=~r[z-1]
    if operation=='remove' and not r.any() or operation=='grow' and not a.any():return None
    if operation=='exchange' and (not r.any() or not a.any()):return None
    after=env.count-int(r.sum())+int(a.sum())
    if net_guided:
        low,high=env.acceptable_counts
        if env.count>high and not low<=after<env.count:return None
        if env.count<low and not env.count<after<=high:return None
        if low<=env.count<=high and not low<=after<=high:return None
    move=Exchange(r,a,f.version)
    return move if valid_move(env,move) else None

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
    changed=(f.occupied&~r)|a
    if getattr(env,'floor_copy_growth',False):
        if f.config.maximum_edit_voxels and int(r.sum()+a.sum())>f.config.maximum_edit_voxels:return False
        if (changed[1:]&~changed[:-1]).any():return False
        if not np.array_equal(r,supported_removal(env,r)):return False
        if a.any():
            plates=copy_plates(env)
            if not any(np.array_equal(a,plate & ~np.concatenate((np.zeros_like(r[:1]),r[:-1]))) for plate in plates):return False
    labels,n=ndi.label(changed,STRUCTURE)
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

def project(env,masks,scores,operation,repair_remove=False,net_guided=False):
    """Only prune neural selections. Exchanges balance; pure moves change FAR."""
    r,a=masks.copy();f=env.frontier
    if getattr(env,'floor_copy_growth',False):return project_floor_copy(env,masks,scores,operation,net_guided)
    if repair_remove and (operation=='remove' or net_guided and operation=='exchange'):
        # Reuse the proven selection-pruning path; permit two retained grounded
        # masses rather than inheriting the original one-component assumption.
        previous=f.components
        try:
            f.components=max(2,previous)
            action=f.project_mask(r,scores[0],-1,int(r.sum()))
        finally:f.components=previous
        if action is None:return None
        r=action.mask.copy()
        if operation=='remove':
            move=Exchange(r,a,f.version)
            return move if valid_move(env,move) else None
    for _ in range(4):
        r&=exterior_air(f.occupied&~r)
        r=f._patch_filter(r,int(r.sum())) if r.any() else r
        a=f._patch_filter(a,int(a.sum())) if a.any() else a
        if operation=='exchange':
            nr,na=int(r.sum()),int(a.sum());delta=env.count-env.goal_count
            if net_guided and delta>0:
                na=min(na,nr);nr=min(nr,na+delta)
            elif net_guided and delta<0:
                nr=min(nr,na);na=min(na,nr-delta)
            else:nr=na=min(nr,na)
            k=min(nr,na)
            if not k:return None
            for mask,confidence,limit in ((r,scores[0],nr),(a,scores[1],na)):
                if mask.sum()>limit:
                    ids=np.flatnonzero(mask);keep=ids[np.argsort(-confidence.ravel()[ids],kind='stable')[:limit]]
                    mask[:]=False;mask.ravel()[keep]=True
        if operation=='remove' and not r.any() or operation=='grow' and not a.any():return None
        move=Exchange(r.copy(),a.copy(),f.version)
        if valid_move(env,move):return move
    return None

def commit(env,move,audit=False):
    if not valid_move(env,move):raise ValueError('Invalid/stale unified edit')
    f=env.frontier
    if getattr(env,'floor_copy_growth',False):
        labels=getattr(env,'copy_labels',np.zeros(env.shape,dtype=np.int64)).copy()
        next_id=int(labels.max())+1
        for z in range(1,env.nz):
            plate=move.add[z]
            if not plate.any():continue
            fresh=plate&(labels[z-1]==0);n=int(fresh.sum())
            labels[z-1][fresh]=np.arange(next_id,next_id+n);next_id+=n
            labels[z][plate]=labels[z-1][plate]
        labels[move.remove]=0;env.copy_labels=labels
    f.occupied[move.remove]=False;f.occupied[move.add]=True;f.version+=1;f.refresh()
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
    def __init__(self,device='cpu',seed=123,config=None,amount_feedback=False,repair_remove=True,net_guided=False,floor_copy=False):
        super().__init__(device,seed);self.config=config or PatchConfig(mode='combined',allocation='feedback')
        self.amount_feedback=bool(amount_feedback)
        self.repair_remove=bool(repair_remove)
        self.net_guided=bool(net_guided)
        self.floor_copy=bool(floor_copy)
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
        grow=torch.as_tensor(copy_eligibility(e) if self.floor_copy else growth_mask(e),device=self.device,dtype=torch.float32)
        return torch.cat((e.observation(include_depth=True),remove[:,None],grow[:,None],e.external_potential[:,None],e.facade_light[:,None]),1)
    def forward(self,e):
        limit=getattr(e,'unified_limit',1);spent=getattr(e,'unified_actions',e.edits)
        goal=torch.tensor([*e.goal_features,max(0,limit-spent)/max(1,limit),spent/max(1,limit)],device=self.device,dtype=torch.float32)
        return self.model(self.observation(e),goal)
    def fields(self,e):
        fields,value,_=self.forward(e);return fields,value
    def legal_operations(self,e,net_guided=None):
        guided=self.net_guided if net_guided is None else net_guided
        r=e.frontier.eligible(-1).any();a=(copy_eligibility(e) if self.floor_copy else growth_mask(e)).any()
        remove=r;grow=a
        if guided:
            low,high=e.acceptable_counts
            remove=r and e.count>low;grow=a and e.count<high
        if self.floor_copy and guided:
            grow=any(e.count+int(plate.sum())<=e.acceptable_counts[1] for plate in copy_plates(e))
        return torch.tensor([remove,grow,r and a,e.far_satisfied],device=self.device,dtype=torch.bool)
    def choose(self,e,training=True,force_operation=None,retries=8):
        e.floor_copy_growth=self.floor_copy
        start=time.perf_counter();self.last_attempt_recorded=False;self.last_operation=None;self.last_rejection=None;self.last_generated_count=0
        with torch.set_grad_enabled(training):
            fields,value,logits=self.forward(e);guided=self.net_guided and force_operation is None
            legal=self.legal_operations(e,net_guided=guided)
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
                weights=torch.stack((e.frontier.weights_tensor(-1,self.device),torch.as_tensor(copy_eligibility(e) if self.floor_copy else growth_mask(e),device=self.device,dtype=torch.float32)))
                kernel=tuple(min(n,s) for n,s in zip(e.shape,self.config.stride))
                mean=F.avg_pool3d(fields[:,None],kernel,stride=self.config.stride,ceil_mode=True,count_include_pad=False)[:,0]
                consumed=[0] if operation=='remove' else [1] if operation=='grow' else [0,1]
                dist=Normal(mean[consumed],self.model.patch_log_std[consumed].clamp(-2,1).exp()[:,None,None,None],validate_args=False)
                remaining=abs(e.count-e.goal_count);left=max(1,getattr(e,'unified_limit',8)-getattr(e,'unified_actions',e.edits)-4)
                toward=(operation=='remove' and e.count>e.goal_count) or (operation=='grow' and e.count<e.goal_count)
                amount=max(4,math.ceil(remaining/left/.65)) if toward else max(4,round(e.target_count*.025))
                if toward and self.amount_feedback:
                    productive=max(1,e.workload.limit-e.workload.margin-getattr(e,'unified_actions',e.edits))
                    amount=max(4,math.ceil(remaining/productive/e.workload.retention))
                if toward:amount=min(amount,remaining)
                if guided and e.far_satisfied:
                    low,high=e.acceptable_counts
                    if operation=='remove':amount=min(amount,e.count-low)
                    elif operation=='grow':amount=min(amount,high-e.count)
                net_amount=0
                if guided and operation=='exchange' and remaining:
                    productive=max(1,e.workload.limit-e.workload.margin-getattr(e,'unified_actions',e.edits))
                    net_amount=min(remaining,math.ceil(remaining/productive/e.workload.retention))
                if self.config.maximum_edit_voxels:amount=min(amount,self.config.maximum_edit_voxels)
                plates=copy_plates(e) if self.floor_copy and operation!='remove' else []
                for attempt in range(retries):
                    latent=dist.sample();decoded=F.interpolate(latent[:,None],size=e.shape,mode='trilinear',align_corners=False)[:,0]
                    if self.config.mode=='combined':decoded=gaussian_field(decoded,self.config.sigma)
                    scores=torch.zeros_like(fields);masks=torch.zeros(fields.shape,device=self.device,dtype=torch.bool)
                    count=max(1,round(amount*(1.,1.,.75,.5,.35,.25,.125,.0625)[min(attempt,7)]))
                    # Decode growth as a full plate. The learned spatial field chooses
                    # which existing grounded mass to extend, never isolated roof cells.
                    plate=None
                    if plates:
                        add_field=decoded[consumed.index(1)].sigmoid()
                        plate_scores=torch.stack([add_field[torch.as_tensor(p,device=self.device)].mean() for p in plates])
                        plate=plates[int(plate_scores.argmax().cpu())]
                        masks[1]=torch.as_tensor(plate,device=self.device)
                        scores[1]=add_field*weights[1]
                    for index,channel in enumerate(consumed):
                        if plate is not None and channel==1:continue
                        field=decoded[index]
                        if channel==0 and self.config.vertical_interior:
                            interior=torch.as_tensor(e.frontier.eligible(-1)&~e.frontier.facade_band,device=self.device)
                            field=torch.where(interior,field.mean(0,keepdim=True).expand_as(field),field)
                        scores[channel]=field.sigmoid()*weights[channel];available=int((weights[channel]>0).sum().cpu())
                        if self.floor_copy and channel==0:
                            # Rank cuts top-down within each column. Ties favour
                            # the upper floor, so lower cuts do not strand upper cells.
                            occupied=e.occupied
                            group_ids=torch.as_tensor(getattr(e,'copy_labels',np.zeros(e.shape,dtype=np.int64)),device=self.device).flatten()
                            if bool((group_ids>0).any()):
                                # Native tensor reduction gives linked cells the
                                # same neural score before selecting a mask.
                                total=torch.zeros(int(group_ids.max().cpu())+1,device=self.device)
                                count_group=torch.zeros_like(total)
                                total.scatter_add_(0,group_ids,scores[channel].flatten())
                                count_group.scatter_add_(0,group_ids,torch.ones_like(scores[channel].flatten()))
                                shared=(total/count_group.clamp_min(1))[group_ids].reshape(e.shape)
                                scores[channel]=torch.where(group_ids.reshape(e.shape)>0,shared,scores[channel])
                            ordered=torch.where(occupied,scores[channel],torch.ones_like(field))
                            ordered=torch.cummin(ordered.flip(0),dim=0).values.flip(0)
                            height=torch.arange(e.nz,device=self.device)[:,None,None]*1e-6
                            scores[channel]=torch.where(weights[channel]>0,ordered+height,0)
                        extra=net_amount if channel==(0 if e.count>e.goal_count else 1) else 0
                        n=min(count+round(extra*(1.,1.,.75,.5,.35,.25,.125,.0625)[min(attempt,7)]),available)
                        if plate is not None and channel==0:
                            # Remove enough area to offset the copied plate and
                            # move toward FAR. Retry scaling changes net work,
                            # never fragments the addition into random roof posts.
                            delta=e.count-e.goal_count if self.net_guided else 0
                            net=round(net_amount*(1.,1.,.75,.5,.35,.25,.125,.0625)[min(attempt,7)])
                            desired=int(plate.sum())+(net if delta>0 else -min(net,int(plate.sum())-1) if delta<0 else 0)
                            n=min(available,max(1,desired))
                        if self.config.maximum_edit_voxels:
                            n=min(n,max(0,self.config.maximum_edit_voxels-int(plate.sum())) if plate is not None else self.config.maximum_edit_voxels)
                        ids=scores[channel].masked_fill(weights[channel]<=0,-1).flatten().topk(n).indices;masks[channel].flatten()[ids]=True
                    if training:logs.append(dist.log_prob(latent).sum()/math.sqrt(latent.numel()))
                    self.last_generated_count=int(masks.sum().cpu())
                    move=project(e,masks.cpu().numpy(),scores.detach().cpu().numpy(),operation,self.repair_remove,guided)
                    if move is not None:break
                if training:logp=logp+torch.stack(logs).sum();entropy=entropy+dist.entropy().mean()
                if move is None:self.last_rejection='mask_generation_failed'
            if training:self.tape.append([logp,value,entropy,None]);self.last_attempt_recorded=True
        self.generation_seconds=time.perf_counter()-start
        return move
    def state(self):
        return dict(format='voxelform.unified.v1',config=asdict(self.config),amountFeedback=self.amount_feedback,repairRemove=self.repair_remove,netGuided=self.net_guided,floorCopy=self.floor_copy,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates,directMaskUpdates=self.direct_updates,importedUpdates=0)
    def load(self,state):
        if state.get('format')!='voxelform.unified.v1':raise ValueError('Unified action checkpoint required')
        other=UnifiedLearner(str(self.device),config=PatchConfig(**state['config']),amount_feedback=state.get('amountFeedback',False),repair_remove=state.get('repairRemove',False),net_guided=state.get('netGuided',False),floor_copy=state.get('floorCopy',False));other.model.load_state_dict(state['model']);other.optimizer.load_state_dict(state['optimizer']);other.updates=int(state['updates']);other.direct_updates=int(state['directMaskUpdates']);self.__dict__.update(other.__dict__)
