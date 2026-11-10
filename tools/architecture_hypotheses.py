"""Isolated factorial experiment; never imports or replaces the live server policy."""
import argparse,copy,gc,hashlib,itertools,json,platform,resource,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from voxelform.policy import Network,Learner
from voxelform.environment import Environment
from voxelform.context import ContextLibrary
from voxelform.shadows import sync

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=[dict(single=s,facade=f,encoder=e) for e,s,f in itertools.product(('2d','3d','multiscale'),(False,True),(False,True))]
def name(c):return f"{c['encoder']}_{'single' if c['single'] else 'dual'}_{'facade' if c['facade'] else 'implicit'}"

def inputs(env,facade):
    observation=env.observation()
    if facade:
        masks=np.stack((env.frontier.remove_skin,env.frontier.add_skin),axis=1)
        observation=torch.cat((observation,torch.as_tensor(masks,dtype=torch.float32,device=env.device)),dim=1)
    return observation

class ExperimentalNetwork(nn.Module):
    def __init__(self,config,seed=123):
        super().__init__();self.config=config;three=config['encoder']!='2d';conv=nn.Conv3d if three else nn.Conv2d
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);original=Network()
            self.encoder=nn.Sequential(conv(7 if config['facade'] else 5,16,3,padding=1),nn.SiLU(),conv(16,24,3,padding=1),nn.SiLU())
            self.goal_encoder=nn.Linear(4 if config['single'] else 3,24)
            self.mask_head=nn.Sequential(conv(24,24,3,padding=1),nn.SiLU(),conv(24,1 if config['single'] else 2,1))
            self.critic=copy.deepcopy(original.critic)
            if config['encoder']=='multiscale':
                self.down1=nn.Sequential(conv(24,24,3,padding=1),nn.SiLU())
                self.down2=nn.Sequential(conv(24,24,3,padding=1),nn.SiLU())
            with torch.no_grad():
                for a,b in ((self.encoder[0],original.encoder[0]),(self.encoder[2],original.encoder[2]),(self.mask_head[0],original.mask_head[0]),(self.mask_head[2],original.mask_head[2])):
                    w=b.weight
                    if config['single'] and a is self.mask_head[2]:w=w.mean(0,keepdim=True)
                    if three:w=w.unsqueeze(2).repeat(1,1,a.kernel_size[0],1,1)/a.kernel_size[0]
                    a.weight.zero_();a.weight[:,:w.shape[1]].copy_(w)
                    # Zero facade weights initially isolate learning from changed random initialization.
                    a.bias.copy_(b.bias.mean().expand_as(a.bias) if config['single'] and a is self.mask_head[2] else b.bias)
                self.goal_encoder.weight.zero_();self.goal_encoder.bias.zero_()
        self.computed_output_channels=1 if config['single'] else 2
    def forward(self,observation,goal,operation):
        three=self.config['encoder']!='2d'
        x=observation.permute(1,0,2,3)[None] if three else observation
        encoded=self.encoder(x)
        if self.config['encoder']=='multiscale':
            d1=self.down1(F.avg_pool3d(encoded,(1,2,2),ceil_mode=True))
            d2=self.down2(F.avg_pool3d(d1,(1,2,2),ceil_mode=True))
            encoded=encoded+F.interpolate(d1,size=encoded.shape[-3:],mode='trilinear',align_corners=False)+F.interpolate(d2,size=encoded.shape[-3:],mode='trilinear',align_corners=False)
        floor=encoded[0].permute(1,0,2,3).mean((-1,-2)) if three else encoded.mean((-1,-2))
        context=torch.cat((floor.mean(0),floor.amax(0)))
        condition=torch.cat((goal,goal.new_tensor([operation]))) if self.config['single'] else goal
        condition=self.goal_encoder(condition)
        encoded=encoded+(condition[None,:,None,None,None] if three else condition[None,:,None,None])
        out=self.mask_head(encoded);out=out[0] if three else out.permute(1,0,2,3)
        # Compatibility view: only one channel is computed for the conditioned model.
        if self.config['single']:out=out.expand(2,-1,-1,-1)
        return out,self.critic(torch.cat((context,goal))).squeeze()

class ExperimentalLearner(Learner):
    def __init__(self,config,seed=123,device='cpu'):
        super().__init__(device,seed);self.config=config;self.model=ExperimentalNetwork(config,seed).to(device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3);self.input_seconds=0.
    def fields(self,env):
        sync(self.device);stamp=time.perf_counter();observation=inputs(env,self.config['facade']);sync(self.device);self.input_seconds=time.perf_counter()-stamp
        goal=torch.tensor(env.goal_features,dtype=torch.float32,device=self.device)
        return self.model(observation,goal,-1 if env.count>env.target_count else 1)

class Control(Learner):
    def __init__(self,kind):super().__init__();self.kind=kind
    def fields(self,env):
        if self.kind=='uniform':fields=torch.full((2,*env.shape),-3.,device=env.device)
        else:fields=torch.stack((-3.+(1-env.light),-3.+env.light))
        return fields,torch.zeros((),device=env.device)

def prepare(env,operation):
    if operation=='subtract':env.reset()
    else:
        occupied=env.allowed.clone();occupied[min(2,env.nz-1):]=False;env.restore_occupancy(occupied)
    env.configure_far(3.)

def episode(policy,env,operation,seed,cap=192):
    torch.manual_seed(seed);prepare(env,operation);policy.clear();stamp=time.perf_counter();counts=[];kept=[];times=[];attempts=[];failure=None
    while not env.terminal and len(counts)<cap:
        action=policy.choose(env,training=False,greedy=True);times.append(policy.generation_seconds);attempts.append(policy.generation_attempts)
        if action is None:failure=policy.last_rejection;break
        # Independent invariant checks on the actual pre-action facade.
        assert not (action.mask&~env.frontier.eligible(action.operation)).any()
        assert env.frontier.validate_mask(action.mask,action.operation)
        counts.append(action.count);kept.append(action.count/max(action.generated_voxels,1))
        env.apply(action,audit=False)
        if env.edits%8==0:env.evaluate()
    env.evaluate();success=env.count==env.target_count
    return dict(success=success,termination='far_reached' if success else failure or 'action_cap',operation=operation,
                far=env.far,targetFar=env.target_far,quality=float(env.quality),usable=float(env.values['potentialUsableFraction']),
                actions=len(counts),changedVoxels=sum(counts),maskSizes=counts,meanRetained=float(np.mean(kept)) if kept else None,
                meanGenerationMs=1000*float(np.mean(times)) if times else None,attempts=attempts,elapsedSeconds=time.perf_counter()-stamp)

def evaluate(policy,cases,seed):
    return [dict(case=tag,sceneSeed=env.seed,shape=env.shape,**episode(policy,env,op,seed+10000+i)) for i,(tag,env,op) in enumerate(cases)]

def train(policy,envs,seed,budget):
    torch.manual_seed(seed+100);stamp=time.perf_counter();used=0;episodes=[];updates=[];i=0;rewards=0.
    condition_counts=[]
    conditions=[(env,op) for env in envs for op in ('subtract','add')]
    if budget%len(conditions):raise ValueError('Budget must divide equally across plot/operation conditions')
    per_condition=budget//len(conditions)
    for env,op in conditions:
        condition_used=0
        while condition_used<per_condition:
            prepare(env,op);policy.clear();actions=0;rejected=False
            while condition_used<per_condition and not env.terminal and actions<128:
                action=policy.choose(env,training=True);used+=1;condition_used+=1
                if action is None:
                    policy.record(-.1);env.termination=policy.last_rejection;rejected=True;break
                before=env.far_error;done=env.apply(action,audit=False);reward=before-env.far_error+(1. if done else 0.);policy.record(reward);rewards+=reward;actions+=1
                if actions%8==0 or done:
                    before=float(env.quality);env.evaluate();policy.add_reward(.1*(float(env.quality)-before))
                if actions%32==0 and not done:updates.append(policy.finish(policy.bootstrap(env)))
            if policy.tape:updates.append(policy.finish(policy.bootstrap(env)))
            episodes.append(dict(operation=op,plan=env.plan,sceneSeed=env.seed,success=env.count==env.target_count,
                                 actions=actions,rejected=rejected,truncated=not env.terminal,far=env.far))
        condition_counts.append(dict(plan=env.plan,operation=op,attemptedActions=condition_used))
    return dict(attemptedActions=used,conditionCounts=condition_counts,updates=updates,episodes=episodes,rewardSum=rewards,seconds=time.perf_counter()-stamp)

def summary(values):return dict(medianMs=float(np.median(values)),p95Ms=float(np.percentile(values,95)),rawMs=values)
def timed(call,device):
    sync(device);stamp=time.perf_counter();v=call();sync(device);return v,1000*(time.perf_counter()-stamp)

def benchmark(config,env,device,iterations):
    policy=ExperimentalLearner(config,device=device);policy.fields(env);policy.fields(env)
    v={k:[] for k in ('input','forwardIncludingInput','generation','projection','apply')};changed=[]
    for _ in range(iterations):
        with torch.no_grad():_,ms=timed(lambda:policy.fields(env),device)
        v['forwardIncludingInput'].append(ms);v['input'].append(policy.input_seconds*1000)
        action,ms=timed(lambda:policy.choose(env,training=False,greedy=True),device)
        if action is None:raise RuntimeError('benchmark neural generation rejected')
        v['generation'].append(ms);v['projection'].append(policy.projection_seconds*1000);changed.append(action.count)
        _,ms=timed(lambda:env.apply(action,audit=False),device);v['apply'].append(ms)
    fields,value=policy.fields(env);loss=fields.square().mean()+value.square()
    _,backward=timed(lambda:loss.backward(),device)
    assert all(torch.isfinite(p.grad).all() for p in policy.model.parameters() if p.grad is not None)
    return dict(parameters=sum(p.numel() for p in policy.model.parameters()),computedOutputChannels=policy.model.computed_output_channels,
                timings={k:summary(a) for k,a in v.items()},backwardMs=backward,changedVoxels=changed,
                rssHighWaterBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                mpsAllocatedBytes=torch.mps.current_allocated_memory() if device=='mps' else None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,default=512);parser.add_argument('--iterations',type=int,default=10);parser.add_argument('--output',default='results/architecture-hypotheses-2026-11-08.json');parser.add_argument('--phase',choices=('all','learning','benchmark'),default='all');args=parser.parse_args()
    torch.set_num_threads(1);out=Path(args.output);out.parent.mkdir(exist_ok=True,parents=True)
    result=dict(plan='agent_notes/architecture_hypotheses_2026-11-08/PLAN.md',platform=platform.platform(),torch=torch.__version__,cpuThreads=1,
                trainingActionBudget=args.budget,seeds=[123,456,789],variants=VARIANTS,mechanistic={},learning=[],controls=[],benchmarks=[],failures=[],
                hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),*sorted((ROOT/'voxelform').glob('*.py'))]})
    def save():out.write_text(json.dumps(result,indent=2)+'\n')
    save()
    if args.phase in ('all','learning'):
        envs=[Environment((4,12,16),sun_count=4,seed=100+i,plan=p) for i,p in enumerate(('rectangle','triangle','courtyard','branch'))]
        cases=[]
        for i,p in enumerate(('rectangle','triangle','courtyard','branch')):
            env=Environment((4,16,20),sun_count=4,seed=9100+i,plan=p)
            cases.extend([(f'heldout_{p}',env,'subtract'),(f'heldout_{p}',env,'add')])
        real=Environment(context=ContextLibrary().load('m_bcn_0093'),sun_count=8,seed=9200)
        cases.extend([('cached_Barcelona',real,'subtract'),('cached_Barcelona',real,'add')])
        result['scenes']=dict(train=[dict(shape=e.shape,plan=e.plan,seed=e.seed) for e in envs],heldout=[dict(case=t,shape=e.shape,operation=o,sceneSeed=e.seed) for t,e,o in cases])
        for kind in ('uniform','sky_greedy'):
            result['controls'].append(dict(kind=kind,evaluation=evaluate(Control(kind),cases,123)));save()
        for config in VARIANTS:
            for seed in result['seeds']:
                row=dict(variant=name(config),config=config,seed=seed)
                try:
                    p=ExperimentalLearner(config,seed);row['initial']=evaluate(p,cases,seed)
                    row['training']=train(p,envs,seed,args.budget);row['trained']=evaluate(p,cases,seed)
                    checkpoint=ROOT/'results/checkpoints/architecture-hypotheses-balanced-2026-11-08'/f'{name(config)}-{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(config=config,seed=seed,model=p.model.state_dict(),optimizer=p.optimizer.state_dict(),updates=p.updates),checkpoint)
                    row['checkpoint']=str(checkpoint.relative_to(ROOT));print(name(config),seed,'trained',sum(x['success'] for x in row['trained']),'/10',round(row['training']['seconds'],1),'s',flush=True)
                    del p;gc.collect()
                except Exception:row['error']=traceback.format_exc();result['failures'].append(row['error']);print(row['error'],flush=True)
                result['learning'].append(row);save()
        del envs,cases,real;gc.collect()
    if args.phase in ('all','benchmark'):
        devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for scale,shape,suns in [('L',(4,48,64),8),('XL',(8,64,96),12)]:
                env,setup=timed(lambda:Environment(shape,device=device,sun_count=suns,seed=123),device)
                for config in VARIANTS:
                    row=dict(variant=name(config),device=device,scale=scale,shape=shape,sunSamples=suns,contextBuildings=16,setupMs=setup)
                    try:
                        env.reset();row.update(benchmark(config,env,device,args.iterations));print('benchmark',device,scale,name(config),round(row['timings']['generation']['medianMs'],2),'ms',flush=True)
                    except Exception:row['error']=traceback.format_exc();result['failures'].append(row['error']);print(row['error'],flush=True)
                    result['benchmarks'].append(row);save();gc.collect()
                    if device=='mps':torch.mps.empty_cache()
                del env;gc.collect()
    print('Saved',out,flush=True)
if __name__=='__main__':main()
