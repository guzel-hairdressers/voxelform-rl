"""Matched workload/count ablations. Synthetic learning is not architectural validation."""
import argparse,functools,hashlib,json,time,gc
from pathlib import Path
import numpy as np
import torch
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner,components
from voxelform.workload import action_reward
from voxelform.context import ContextLibrary
from tools.architecture_hypotheses import prepare

ROOT=Path(__file__).resolve().parents[1]
_ENVS={}

def make(config,shape=(4,16,20),plan='rectangle',device='cpu',context=None,reuse=False):
    factory=functools.partial(PatchFacades,config=config,depth_device=device if device!='cpu' else None)
    key=(tuple(shape),plan,device,None if context is None else context['site_id'])
    if reuse and key in _ENVS:
        e=_ENVS[key];e.frontier_factory=factory;e.reset();return e
    e=Environment(shape,plan=plan,context=context,sun_count=12 if shape[0]==8 else 8,seed=9301,device=device,frontier_factory=factory)
    if reuse:_ENVS[key]=e
    return e

def episode(p,e,op,seed,training=False,cap=600):
    torch.manual_seed(seed);prepare(e,op);p.clear();start=time.perf_counter();masks=[];reason=None
    while not e.terminal and e.edits<cap:
        a=p.choose(e,training=training)
        if a is None:
            e.termination=p.last_rejection
            if training and p.tape:p.record(-2.)
            break
        assert e.frontier.validate_mask(a.mask,a.operation)
        before=e.far_error;done=e.apply(a,audit=False)
        masks.append(dict(count=a.count,generated=a.generated_voxels,patches=components(a.mask)['components'],generationMs=p.generation_seconds*1000))
        if training:p.record(action_reward(e,before))
        if e.edits%8==0 or done:e.evaluate()
        if training and (done or len(p.tape)>=32):p.finish(p.bootstrap(e))
    if training and p.tape:p.finish(p.bootstrap(e))
    e.evaluate()
    return dict(operation=op,success=e.termination=='far_reached',termination=e.termination or 'experiment_cap',steps=e.edits,
        budget=e.workload.state() if e.workload else None,required=abs(e.start_count-e.target_count),remaining=abs(e.count-e.target_count),
        quality=float(e.quality.cpu()),depthPenalty=float(e.depth_badness.sum().cpu())/e.capacity,
        masks=masks,seconds=time.perf_counter()-start)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--device',default='cpu');ap.add_argument('--train-actions',type=int,default=768);args=ap.parse_args()
    torch.set_num_threads(1)
    out=ROOT/'results'/('workload-pilot-2026-11-09.json' if args.pilot else 'workload-budget-2026-11-09.json')
    configs={name:PatchConfig(mode='combined',eligibility='depth',minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,
        batched=True,vertical_interior=True,maximum_edit_voxels=256 if name=='baseline' else 0,allocation=name if name in ('quota','learned') else 'probability')
        for name in ('baseline','uncapped','quota','learned')}
    cases=[('S',(4,16,20),'rectangle',None),('irregular',(4,20,28),'branch',None),('wide',(4,40,48),'rectangle',None),
        ('L',(4,48,64),'rectangle',None),('XL',(8,64,96),'rectangle',None),('Barcelona',None,'rectangle',ContextLibrary().load('m_bcn_0093'))]
    r=dict(configs={k:vars(v) for k,v in configs.items()},device=args.device,trainingActions=args.train_actions,rows=[],hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'voxelform').glob('*.py'))]})
    if args.resume and out.exists():
        current=r['hashes'];r=json.loads(out.read_text());r.setdefault('resumptions',[]).append(dict(startingRows=len(r['rows']),hashes=current,reason='Reuse immutable scene/shadow preparation; preserve existing measurements and complete remaining seeds'))
    def save():out.write_text(json.dumps(r,indent=2)+'\n')
    for seed in ((123,) if args.pilot else (123,456,789)):
        for name,c in configs.items():
            p=PatchLearner(args.device,seed,c)
            for stage in (('initial',) if args.pilot else ('initial','trained')):
                if sum(x['seed']==seed and x['name']==name and x['stage']==stage for x in r['rows'])==len(cases)*2:continue
                if stage=='trained':
                    root=ROOT/'results/checkpoints/workload-budget-2026-11-09';root.mkdir(exist_ok=True,parents=True);checkpoint=root/f'{name}-{seed}.pt'
                    if args.resume and checkpoint.exists():p.load(torch.load(checkpoint,map_location=args.device,weights_only=True))
                    else:
                        es=[make(c,shape=(4,12,16),plan=plan,device=args.device,reuse=True) for plan in ('rectangle','triangle','branch')]
                        actions=0;rollouts=[]
                        while actions<args.train_actions:
                            e=es[len(rollouts)%len(es)];row=episode(p,e,'subtract' if len(rollouts)%2==0 else 'add',seed+90000+len(rollouts),True)
                            actions+=max(1,row['steps']);rollouts.append(row)
                        r.setdefault('training',[]).append(dict(seed=seed,name=name,actions=actions,updates=p.updates,episodes=rollouts))
                        torch.save(p.state(),checkpoint);save()
                for tag,shape,plan,context in cases:
                    if all(any(x['seed']==seed and x['name']==name and x['stage']==stage and x['case']==tag and x['operation']==op for x in r['rows']) for op in ('subtract','add')):continue
                    e=make(c,shape=shape or (6,24,32),plan=plan,context=context,device=args.device,reuse=True)
                    for op in ('subtract','add'):
                        if any(x['seed']==seed and x['name']==name and x['stage']==stage and x['case']==tag and x['operation']==op for x in r['rows']):continue
                        row=dict(seed=seed,name=name,stage=stage,case=tag,shape=e.shape,**episode(p,e,op,seed+50000))
                        r['rows'].append(row);save();print(name,seed,stage,tag,op,row['steps'],row['termination'],round(row['quality'],3),flush=True)
                    del e;gc.collect()
    print(out,flush=True)

if __name__=='__main__':main()
