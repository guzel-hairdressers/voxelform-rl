"""Trainable actor-critic over coherent volume-edit proposals.

Floor images share a CNN; their embeddings aggregate across the building.
One optimizer step follows a rollout. No quality-improvement claim is implied.
"""
import torch
from torch import nn
from torch.distributions import Categorical


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(5,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,24,3,padding=1),nn.SiLU())
        self.actor=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,1))
        self.critic=nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,observation,features,valid,masks):
        encoded=self.encoder(observation);floor=encoded.mean((-1,-2));context=torch.cat((floor.mean(0),floor.amax(0)))
        # Condition each action on the cells it actually edits, not just global
        # averages and coordinates. This lets context distinguish two cut sites.
        weights=masks.float().flatten(1)
        local=(weights@encoded.permute(0,2,3,1).reshape(-1,24))/weights.sum(1,keepdim=True).clamp_min(1)
        logits=self.actor(torch.cat((context.expand(len(features),-1),local,features),dim=1)).flatten()
        return logits.masked_fill(~valid,-1e9),self.critic(context).squeeze()


class Learner:
    def __init__(self,device='cpu',seed=123):
        self.device=torch.device(device)
        with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);self.model=Network().to(device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3);self.updates=0;self.clear()
    def clear(self):self.tape=[];self.previous_quality=None
    def choose(self,env,training=True,greedy=False):
        masks,features,valid=env.candidates()
        with torch.set_grad_enabled(training):
            logits,value=self.model(env.observation(),features,valid,masks);distribution=Categorical(logits=logits)
            action=logits.argmax() if greedy else distribution.sample()
            if training:self.tape.append([distribution.log_prob(action),value,distribution.entropy(),None])
        # One scalar control transfer per edit; field/mask arithmetic stays on device.
        return int(action.detach().cpu()),masks
    def record(self,reward):
        if self.tape:self.tape[-1][3]=reward.detach()
    def finish(self):
        if not self.tape:return dict(updates=self.updates,loss=0.,gradientNorm=0.)
        returns=[];total=torch.zeros((),device=self.device)
        for _,_,_,reward in reversed(self.tape):total=total+reward;returns.append(total)
        returns=torch.stack(list(reversed(returns)))
        logp,value,entropy,_=zip(*self.tape);value=torch.stack(value);advantage=returns-value
        actor=-(torch.stack(logp)*advantage.detach()).mean();critic=nn.functional.smooth_l1_loss(value,returns)
        loss=actor+.5*critic-.001*torch.stack(entropy).mean()
        self.optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(self.model.parameters(),2.,error_if_nonfinite=True);self.optimizer.step();self.updates+=1
        result=dict(updates=self.updates,loss=float(loss.detach().cpu()),actorLoss=float(actor.detach().cpu()),gradientNorm=float(norm.detach().cpu()))
        self.clear();return result
    def state(self):return dict(format='voxelform.policy.v2',model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates)
    def load(self,state):
        version=state.get('format')
        if version not in ('voxelform.policy.v1','voxelform.policy.v2'):raise ValueError('not a VoxelForm policy')
        weights=dict(state['model'])
        if version=='voxelform.policy.v1':
            old=weights['actor.0.weight'];expanded=old.new_zeros((64,84));expanded[:,:48]=old[:,:48];expanded[:,72:]=old[:,48:];weights['actor.0.weight']=expanded
            # The new local-feature columns have no prior optimizer moments.
            self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        self.model.load_state_dict(weights)
        if version=='voxelform.policy.v2':self.optimizer.load_state_dict(state['optimizer'])
        self.updates=state['updates'];self.clear()
