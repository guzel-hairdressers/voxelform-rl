"""Equal-wall-time phased vs unified action pilot; preserve live training."""
import argparse,functools,hashlib,json,time,gc
from pathlib import Path
import numpy as np
import torch
from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades
from voxelform.training import EnvironmentLearner,depth_potential
from voxelform.unified import UnifiedLearner,commit
from tools.train_m4 import run_episode as phased_episode

ROOT=Path(__file__).resolve().parents[1]
def unified_episode(p,e,target,seed,training,deadline=None,initial_storeys=6):
    torch.manual_seed(seed);p.clear()
    if e.target_far!=target:e.configure_far(target)
    solid=e.allowed.clone();solid[initial_storeys:]=False;e.restore_occupancy(solid)
    e.unified_limit=max(8,e.workload.limit+8);e.unified_actions=0;e.termination=None
    initial=max(1,abs(e.count-e.goal_count));trace=[];updates=0;stats={};interrupted=False
    start=time.perf_counter();reward_sum=0.
    def audit():
        nonlocal reward_sum
        if e.last_audit_edit!=e.edits:
            before=float(e.quality.cpu());e.evaluate();bonus=5*(float(e.quality.cpu())-before)
            if training and p.tape:p.add_reward(bonus)
            reward_sum+=bonus
    for attempt in range(e.unified_limit):
        if deadline is not None and time.perf_counter()>=deadline:interrupted=True;break
        before=abs(e.count-e.goal_count);depth=depth_potential(e);was_far=e.far_satisfied
        move=p.choose(e,training);operation=p.last_operation;e.unified_actions+=1
        if operation=='finish':
            e.termination='far_reached';reward=.1
        elif move is None:reward=-.03
        else:
            commit(e,move)
            reward=.2*(before-abs(e.count-e.goal_count))/initial+depth_potential(e)-depth-.01/e.unified_limit
        if p.amount_feedback and p.last_generated_count:
            e.workload.observe(0 if move is None else int(move.remove.sum()+move.add.sum()),p.last_generated_count)
        if training:p.record(reward)
        reward_sum+=reward
        trace.append(dict(action=operation,accepted=move is not None,far=e.far,wasFarSatisfied=was_far,
            removed=int(move.remove.sum()) if move else 0,added=int(move.add.sum()) if move else 0,
            grewAboveInitial=int(move.add[initial_storeys:].sum()) if move else 0))
        if e.edits%8==0 or e.terminal:audit()
        if e.terminal:break
        if training and len(p.tape)>=32:
            stats=p.finish(p.fields(e)[1].detach());updates+=1
    if not interrupted and not e.terminal:
        e.termination='far_reached' if e.far_satisfied else 'step_budget_exhausted'
        penalty=.1 if e.far_satisfied else -.5*abs(e.count-e.goal_count)/initial
        if training and p.tape:p.add_reward(penalty)
        reward_sum+=penalty
    audit()
    if training and p.tape:stats=p.finish(p.fields(e)[1].detach() if interrupted else 0.);updates+=1
    return dict(success=e.termination=='far_reached',interrupted=interrupted,termination=e.termination,
        far=e.far,steps=e.edits,attempts=e.unified_actions,budget=e.unified_limit,
        quality=float(e.quality.cpu()),environment={k:float(v.cpu()) for k,v in e.values.items()},
        trace=trace,seconds=time.perf_counter()-start,updates=updates,learning=stats,reward=reward_sum,
        highestStorey=int(np.flatnonzero(e.frontier.occupied.any((1,2)))[-1])+1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=90);ap.add_argument('--seeds',type=int,nargs='+',default=[123,456]);ap.add_argument('--device',default='mps')
    ap.add_argument('--tag');ap.add_argument('--arms',nargs='+',choices=['phased','unified'],default=['phased','unified']);ap.add_argument('--net-guided',action='store_true');ap.add_argument('--resume');args=ap.parse_args()
    if args.seconds<=0:raise ValueError('Positive wall-time budget required')
    if args.resume and (args.arms!=['unified'] or len(args.seeds)!=1):raise ValueError('Resume needs one seed and only the unified arm')
    suffix=f'-{args.tag}' if args.tag else ''
    if args.tag and (not args.tag.replace('-','').replace('_','').isalnum()):raise ValueError('Tag must be a simple filename label')
    torch.set_num_threads(1);lib=ContextLibrary();path=ROOT/f'results/unified-training{suffix}-2026-11-10.json'
    if path.exists():raise ValueError('Preserve existing result file')
    c=PatchConfig(mode='combined',allocation='feedback',eligibility='depth',facade_layers=2,minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=True,budget_reserve_fraction=.2)
    result=dict(arguments=vars(args),rows=[],setup=[],note='Live user MPS training may run concurrently; active-time budgets are matched, hardware throughput is not isolated.',hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'voxelform').glob('*.py'))]})
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    def make(tag,shape,plan='rectangle',site=None):
        start=time.perf_counter();context=lib.load(site,floors=10,grid_alignment='auto_fit') if site else None
        e=Environment(shape,plan=plan,context=context,device=args.device,sun_count=12 if shape[-1]==96 else 8,far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=args.device))
        preparation=e.precompute_external();result['setup'].append(dict(case=tag,shape=e.shape,buildings=len(context['buildings']) if context else 16,seconds=time.perf_counter()-start,external=preparation));save();print('prepared',tag,flush=True);return e
    pool=[make('train-wide',(10,40,48)),make('train-Barcelona',(10,24,32),site='m_bcn_0093')]
    tests=[('heldout-XL',make('heldout-XL',(10,64,96))),('heldout-Barcelona',make('heldout-Barcelona',(10,24,32),site='m_bcn_0094'))]
    checkpoints=ROOT/f'results/checkpoints/unified{suffix}-2026-11-10';checkpoints.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        for arm in (['phased','unified'] if seed%2 else ['unified','phased']):
            if arm not in args.arms:continue
            p=UnifiedLearner(args.device,seed,c,amount_feedback=args.net_guided,net_guided=args.net_guided) if arm=='unified' else EnvironmentLearner(args.device,seed,c)
            if args.resume:
                saved=torch.load(args.resume,map_location=args.device,weights_only=True)
                if saved.get('format')=='voxelform.unified-study.v1':saved=saved['policy']
                p.load(saved)
                changed=(p.net_guided!=args.net_guided or p.amount_feedback!=args.net_guided or not p.repair_remove)
                p.net_guided=args.net_guided;p.amount_feedback=args.net_guided;p.repair_remove=True
                if changed:p.optimizer=torch.optim.Adam(p.model.parameters(),lr=1e-3)
            def run(e,target,s,training,deadline=None):
                if arm=='unified':return unified_episode(p,e,target,s,training,deadline)
                return phased_episode(p,e,target,s,'quality',training,deadline,6,4)
            initial=[dict(case=tag,**run(e,3.,seed+80000+i,False)) for i,(tag,e) in enumerate(tests)]
            start=time.perf_counter();deadline=start+args.seconds;history=[];i=0
            while time.perf_counter()<deadline:
                history.append(run(pool[i%len(pool)],3. if i%3 else 5.,seed+100000+i,True,deadline));i+=1
            actual=time.perf_counter()-start
            final=[dict(case=tag,**run(e,3.,seed+80000+i,False)) for i,(tag,e) in enumerate(tests)]
            state=p.state()
            torch.save(dict(format='voxelform.unified-study.v1',protocol=arm,initialStoreys=6,ceilingStoreys=10,policy=state),checkpoints/f'{arm}-{seed}.pt')
            result['rows'].append(dict(arm=arm,seed=seed,initial=initial,final=final,training=history,actualTrainingSeconds=actual,updates=p.updates));save();print(arm,seed,'heldout',sum(x['success'] for x in final),'/2','updates',p.updates,flush=True)
            del p;gc.collect()
    print(path,flush=True)
if __name__=='__main__':main()
