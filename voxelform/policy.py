"""Direct per-voxel neural add/subtract masks, conditioned on the FAR goal.

There are no sampled rectangle/ellipse proposals and no policy ranking a list
of masks. Two convolutional output channels define the mask distribution itself.
"""
import math
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli
from .actions import ACTION_SPACE
from .shadows import sync


class Network(nn.Module):
    def __init__(self,input_channels=5):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(input_channels,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,24,3,padding=1),nn.SiLU())
        self.goal_encoder=nn.Linear(3,24)
        self.mask_head=nn.Sequential(nn.Conv2d(24,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,2,1))
        self.critic=nn.Sequential(nn.Linear(51,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.goal_encoder.weight);nn.init.zeros_(self.goal_encoder.bias)
        nn.init.normal_(self.mask_head[-1].weight,std=.01);nn.init.constant_(self.mask_head[-1].bias,-3.)
    @property
    def actor(self):return self.mask_head
    def forward(self,observation,goal):
        encoded=self.encoder(observation);floor=encoded.mean((-1,-2));context=torch.cat((floor.mean(0),floor.amax(0)))
        conditioned=encoded+self.goal_encoder(goal)[None,:,None,None]
        fields=self.mask_head(conditioned).permute(1,0,2,3)  # subtract/add, z,y,x
        return fields,self.critic(torch.cat((context,goal))).squeeze()


class Learner:
    def __init__(self,device='cpu',seed=123):
        self.device=torch.device(device)
        with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);self.model=Network().to(device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        self.updates=0;self.direct_updates=0;self.imported_updates=0;self.migration=None
        self.generation_seconds=0.;self.projection_seconds=0.;self.model_seconds=0.;self.generation_attempts=0;self.last_attempt_recorded=False;self.last_rejection=None;self.clear()
    def clear(self):self.tape=[]
    def fields(self,env):
        goal=torch.tensor(env.goal_features,dtype=torch.float32,device=self.device)
        return self.model(env.observation(),goal)
    def choose(self,env,training=True,greedy=False):
        self.last_attempt_recorded=False;self.projection_seconds=0.
        operation=-1 if env.count>env.goal_count else 1
        eligible=np.array(env.frontier.eligible(operation),copy=True)
        if not eligible.any():self.last_rejection='no_admissible_edits';return None
        sync(self.device);stamp=time.perf_counter()
        legal=torch.as_tensor(eligible,device=self.device);remaining=abs(env.count-env.goal_count)
        with torch.set_grad_enabled(training):
            fields,value=self.fields(env);sync(self.device);self.model_seconds=time.perf_counter()-stamp;field=fields[0 if operation==-1 else 1]
            logits=field[legal];distribution=Bernoulli(logits=logits)
            scores=field.detach().cpu().numpy();log_probabilities=[];entropies=[]
            action=None
            for attempt in range(8):
                if greedy and attempt==0:
                    # Deterministic decoding of the neural probability field:
                    # its expected active area sets k; its logits choose pixels.
                    k=max(min(2,remaining,len(logits)),min(remaining,len(logits),int(distribution.probs.sum().round().detach().cpu())))
                    selected=torch.zeros_like(logits);selected[torch.topk(logits,k).indices]=1
                else:selected=distribution.sample()
                if training:
                    log_probabilities.append(distribution.log_prob(selected).sum()/math.sqrt(len(logits)))
                    entropies.append(distribution.entropy().mean())
                raw=torch.zeros(env.shape,dtype=torch.bool,device=self.device);raw[legal]=selected.bool()
                action=env.frontier.project_mask(raw.detach().cpu().numpy(),scores,operation,remaining)
                self.projection_seconds+=env.frontier.projection_seconds
                self.generation_attempts=attempt+1
                if action is not None:break
            if training:
                self.tape.append([torch.stack(log_probabilities).sum(),value,torch.stack(entropies).mean(),None]);self.last_attempt_recorded=True
        sync(self.device);self.generation_seconds=time.perf_counter()-stamp
        self.last_rejection=None if action is not None else 'mask_generation_failed'
        return action
    def record(self,reward):
        if self.tape:self.tape[-1][3]=torch.as_tensor(reward,dtype=torch.float32,device=self.device).detach()
    def add_reward(self,reward):
        if self.tape:self.tape[-1][3]+=torch.as_tensor(reward,dtype=torch.float32,device=self.device).detach()
    def bootstrap(self,env):
        if env.terminal:return 0.
        with torch.no_grad():return self.fields(env)[1].detach()
    def finish(self,bootstrap=0.):
        if not self.tape:return dict(updates=self.updates,directMaskUpdates=self.direct_updates,importedUpdates=self.imported_updates,loss=0.,gradientNorm=0.)
        returns=[];total=torch.as_tensor(bootstrap,dtype=torch.float32,device=self.device).detach()
        for _,_,_,reward in reversed(self.tape):total=reward+.99*total;returns.append(total)
        returns=torch.stack(list(reversed(returns)))
        logp,value,entropy,_=zip(*self.tape);value=torch.stack(value);advantage=returns-value
        actor=-(torch.stack(logp)*advantage.detach()).mean();critic=nn.functional.smooth_l1_loss(value,returns)
        loss=actor+.5*critic-.001*torch.stack(entropy).mean()
        self.optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(self.model.parameters(),2.,error_if_nonfinite=True);self.optimizer.step()
        self.updates+=1;self.direct_updates+=1
        result=dict(updates=self.updates,directMaskUpdates=self.direct_updates,importedUpdates=self.imported_updates,
                    loss=float(loss.detach().cpu()),actorLoss=float(actor.detach().cpu()),gradientNorm=float(norm.detach().cpu()))
        self.clear();return result
    def state(self):
        return dict(format='voxelform.policy.v5',actionSpace=ACTION_SPACE,model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),updates=self.updates,directMaskUpdates=self.direct_updates,
                    importedUpdates=self.imported_updates,migration=self.migration)
    def load(self,state):
        version=state.get('format')
        if version not in tuple(f'voxelform.policy.v{i}' for i in range(1,6)):raise ValueError('not a VoxelForm policy')
        with torch.random.fork_rng(devices=[]):torch.manual_seed(123);replacement=Network().to(self.device)
        migration=state.get('migration')
        if version=='voxelform.policy.v5':
            if state.get('actionSpace')!=ACTION_SPACE:raise ValueError('incompatible direct-mask action space')
            replacement.load_state_dict(state['model']);direct=int(state.get('directMaskUpdates',0));imported=int(state.get('importedUpdates',0))
        else:
            weights=replacement.state_dict()
            for key,value in state['model'].items():
                if key.startswith('encoder.') or key.startswith('critic.'):
                    if key=='critic.0.weight' and value.shape[1]==48:
                        expanded=value.new_zeros((32,51));expanded[:,:48]=value;value=expanded
                    weights[key]=value
            replacement.load_state_dict(weights);direct=0;imported=int(state['updates'])
            migration=f'{version}: retained encoder/critic; initialized new direct add/subtract mask head and optimizer'
        optimizer=torch.optim.Adam(replacement.parameters(),lr=1e-3)
        if version=='voxelform.policy.v5':optimizer.load_state_dict(state['optimizer'])
        self.model=replacement;self.optimizer=optimizer;self.updates=int(state['updates'])
        self.direct_updates=direct;self.imported_updates=imported;self.migration=migration;self.clear()
