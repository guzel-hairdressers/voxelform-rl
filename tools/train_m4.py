"""Bounded wall-clock pilot: current reward, quality reward, quality curriculum.

Cached fixed scenes, unchanged GPU environmental evaluations, no browser polling.
One optimizer update per episode/32-step rollout. No claims of convergence.
"""
import argparse,functools,gc,hashlib,json,random,resource,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner
from voxelform.training import step_reward,audit_reward,depth_potential,easy_probability,EnvironmentLearner
from voxelform.shadows import sync
from voxelform.redistribution import choose as choose_exchange,commit as commit_exchange

ROOT=Path(__file__).resolve().parents[1]

def run_episode(p,e,target,seed,mode,training,deadline=None,initial_storeys=None,refine_steps=0):
    torch.manual_seed(seed);p.clear()
    if initial_storeys is None:e.reset()
    else:
        solid=e.allowed.clone();solid[initial_storeys:]=False;e.restore_occupancy(solid)
    if target!=e.target_far:e.configure_far(target)
    start=time.perf_counter();stats={};initial_depth=depth_potential(e);reward_sum=0.;updates=0
    def audit():
        nonlocal reward_sum
        if e.last_audit_edit!=e.edits:
            before=float(e.quality.cpu());e.evaluate();bonus=audit_reward(before,float(e.quality.cpu()),mode)
            if training and p.tape:p.add_reward(bonus)
            reward_sum+=bonus
    def learn():
        nonlocal stats,updates
        if training and p.tape:stats=p.finish(p.bootstrap(e));updates+=1
    interrupted=False
    while not e.terminal:
        if deadline is not None and time.perf_counter()>=deadline:interrupted=True;break
        action=p.choose(e,training)
        if action is None:
            e.termination=p.last_rejection
            penalty=step_reward(e,e.far_error,mode)
            if training and p.tape:
                if p.last_attempt_recorded:p.record(penalty)
                else:p.add_reward(penalty)
            reward_sum+=penalty;break
        before=e.far_error;before_depth=depth_potential(e) if mode=='quality' else 0.
        e.apply(action,audit=False);reward=step_reward(e,before,mode)
        if mode=='quality':reward+=depth_potential(e)-before_depth
        if training:p.record(reward)
        reward_sum+=reward
        if e.edits%8==0 or e.terminal:audit()
        if training and (len(p.tape)>=32 or (e.terminal and not (refine_steps and e.far_satisfied))):learn()
    refinement=[]
    if not interrupted and e.far_satisfied and refine_steps:
        for _ in range(refine_steps):
            if deadline is not None and time.perf_counter()>=deadline:interrupted=True;break
            before=float(e.quality.cpu());before_depth=depth_potential(e)
            exchange=choose_exchange(p,e,max(4,min(256,round(e.target_count*.025))),training)
            if exchange is None:
                if training and p.tape:p.record(-.01)
                break
            commit_exchange(e,exchange)
            reward=audit_reward(before,float(e.quality.cpu()),mode)-.01/refine_steps
            if mode=='quality':reward+=depth_potential(e)-before_depth
            if training:p.record(reward)
            reward_sum+=reward;refinement.append(dict(count=exchange.count,far=e.far,quality=float(e.quality.cpu())))
            if training and len(p.tape)>=32:
                bootstrap=p.fields(e)[1].detach()
                stats=p.finish(bootstrap);updates+=1
    if interrupted:
        # Preserve time-budget fairness: finish the existing on-policy rollout
        # using its nonterminal bootstrap; never label it as a failed episode.
        if training and p.tape:
            stats=p.finish(p.fields(e)[1].detach() if e.far_satisfied and refine_steps else p.bootstrap(e));updates+=1
    else:audit();learn()
    sync(e.device)
    return dict(steps=e.edits,success=e.termination=='far_reached',termination=e.termination,
        interrupted=interrupted,seconds=time.perf_counter()-start,reward=reward_sum,quality=float(e.quality.cpu()),
        environment={k:float(v.detach().cpu()) for k,v in e.values.items()},
        refinement=refinement,
        fieldsCurrent=e.last_audit_edit==e.edits,depthPenalty=-depth_potential(e),depthChange=depth_potential(e)-initial_depth,
        far=e.far,targetFar=e.target_far,remaining=abs(e.count-e.goal_count),updates=updates,learning=stats)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=60);ap.add_argument('--seeds',type=int,nargs='+',default=[123,456]);ap.add_argument('--device',default='mps');ap.add_argument('--tag',default='pilot')
    ap.add_argument('--arms',nargs='+',choices=['current','quality-mixed','quality-curriculum','quality-informed','quality-facade'],default=['current','quality-mixed','quality-curriculum']);ap.add_argument('--resume');ap.add_argument('--far-tolerance',type=float,default=.1)
    ap.add_argument('--ceiling-storeys',type=int);ap.add_argument('--initial-storeys',type=int);ap.add_argument('--refine-steps',type=int,default=0);ap.add_argument('--facade-layers',type=int,default=1);args=ap.parse_args()
    if args.seconds<=0:raise ValueError('seconds must be positive')
    if args.refine_steps<0 or (args.ceiling_storeys is not None and args.ceiling_storeys<1):raise ValueError('Invalid refinement budget or ceiling')
    if args.resume and (len(args.arms)!=1 or len(args.seeds)!=1):raise ValueError('Resume requires one arm and one seed')
    if args.initial_storeys is not None and not (args.ceiling_storeys and 0<args.initial_storeys<=args.ceiling_storeys):raise ValueError('Initial storeys must fit the explicit ceiling')
    torch.set_num_threads(1);c=PatchConfig(allocation='feedback',eligibility='depth',budget_reserve_fraction=.2,minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=True,facade_layers=args.facade_layers)
    lib=ContextLibrary();r=dict(device=args.device,secondsPerArm=args.seconds,seeds=args.seeds,config=asdict(c),arguments=vars(args),setup=[],rows=[],
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'voxelform').glob('*.py'))]})
    path=ROOT/f'results/m4-training-{args.tag}-2026-11-09.json';checkpoints=ROOT/f'results/checkpoints/m4-training-{args.tag}-2026-11-09';checkpoints.mkdir(exist_ok=True,parents=True)
    if path.exists():raise ValueError('Result file already exists; choose a new --tag to preserve it')
    def save():path.write_text(json.dumps(r,indent=2)+'\n')
    def checkpoint(p,name,seed):
        destination=checkpoints/f'{name}-{seed}.pt';temporary=destination.with_suffix('.tmp')
        state=p.state()
        if args.refine_steps:state=dict(format='voxelform.refinement-experiment.v1',policy=state,initialStoreys=args.initial_storeys,ceilingStoreys=args.ceiling_storeys,refineSteps=args.refine_steps)
        torch.save(state,temporary);temporary.replace(destination)
    def make(tag,shape,plan='rectangle',site=None,seed=9301):
        start=time.perf_counter();context=lib.load(site,floors=args.ceiling_storeys or 6) if site else None
        sun_count=12 if shape[0]==8 else 8
        if args.ceiling_storeys:shape=(args.ceiling_storeys,*shape[1:])
        e=Environment(shape,plan=plan,seed=seed,context=context,device=args.device,sun_count=sun_count,
            frontier_factory=functools.partial(PatchFacades,config=c,depth_device=args.device),far_tolerance=args.far_tolerance)
        setup=time.perf_counter()-start;preparation=e.precompute_external()
        r['setup'].append(dict(case=tag,shape=e.shape,setupSeconds=setup,externalPreparation=preparation,contextBuildings=len(context['buildings']) if context else 16));save();print('prepared',tag,round(setup+preparation['seconds'],2),flush=True)
        return e
    pool=[make('train-wide',(4,40,48)),make('train-irregular',(4,32,40),'branch'),make('train-Barcelona',(6,24,32),site='m_bcn_0093')]
    tests=[('heldout-irregular',make('heldout-irregular',(4,36,44),'courtyard',seed=19301)),('heldout-XL',make('heldout-XL',(8,64,96),seed=19302)),('heldout-Barcelona',make('heldout-Barcelona',(6,24,32),site='m_bcn_0094'))]
    for seed in args.seeds:
        # Rotate the order to reduce systematic warmup/thermal ordering bias.
        arms=[('current','current','mixed'),('quality-mixed','quality','mixed'),('quality-curriculum','quality','curriculum'),('quality-informed','quality','mixed'),('quality-facade','quality','mixed')]
        arms=[arm for arm in arms if arm[0] in args.arms]
        if seed%2==0:arms=arms[::-1]
        for name,mode,schedule in arms:
            p=EnvironmentLearner(args.device,seed,c,include_facade_sky=name=='quality-facade') if name in ('quality-informed','quality-facade') else PatchLearner(args.device,seed,c)
            rng=random.Random(seed);torch.manual_seed(seed)
            if args.resume:
                state=torch.load(args.resume,map_location=args.device,weights_only=True)
                if state.get('format')=='voxelform.refinement-experiment.v1':
                    if (state['initialStoreys'],state['ceilingStoreys'],state['refineSteps'])!=(args.initial_storeys,args.ceiling_storeys,args.refine_steps):raise ValueError('Refinement protocol differs from checkpoint')
                    state=state['policy']
                p.load(state)
                if p.config!=c:raise ValueError('Resume checkpoint has different edit settings')
            initial_updates=p.updates
            initial=[dict(case=tag,**run_episode(p,e,3.,seed+80000+i,mode,False,initial_storeys=args.initial_storeys,refine_steps=args.refine_steps)) for i,(tag,e) in enumerate(tests)]
            start=time.perf_counter();deadline=start+args.seconds;history=[];count=0;last_save=start
            while time.perf_counter()<deadline:
                e=pool[count%len(pool)];fraction=(time.perf_counter()-start)/args.seconds
                easy=rng.random()<easy_probability(schedule,fraction)
                starting_far=e.capacity/e.plot_area*(args.initial_storeys/e.nz if args.initial_storeys else 1.)
                target=starting_far*(1-rng.uniform(.1,.25)) if easy else 3.
                history.append(dict(easy=easy,target=target,case=count%len(pool),**run_episode(p,e,target,seed+count+100000,mode,True,deadline,args.initial_storeys,args.refine_steps)));count+=1
                if time.perf_counter()-last_save>=60:checkpoint(p,name,seed);last_save=time.perf_counter()
            seconds=time.perf_counter()-start
            final=[dict(case=tag,**run_episode(p,e,3.,seed+80000+i,mode,False,initial_storeys=args.initial_storeys,refine_steps=args.refine_steps)) for i,(tag,e) in enumerate(tests)]
            checkpoint(p,name,seed)
            r['rows'].append(dict(name=name,mode=mode,schedule=schedule,seed=seed,initial=initial,final=final,training=history,
                actualTrainingSeconds=seconds,actions=sum(x['steps'] for x in history),episodes=sum(not x['interrupted'] for x in history),updates=p.updates,initialUpdates=initial_updates,
                rssHighWaterBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mpsAllocatedBytes=torch.mps.current_allocated_memory() if args.device=='mps' else None))
            save();print(name,seed,'seconds',round(seconds,2),'actions',r['rows'][-1]['actions'],'heldout',sum(x['success'] for x in final),'/3',flush=True)
            del p;gc.collect()
    print(path,flush=True)

if __name__=='__main__':main()
