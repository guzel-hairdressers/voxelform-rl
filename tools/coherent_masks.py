"""Controlled direct-mask coherence experiment; preserves historical policies/results."""
import argparse,functools,gc,hashlib,json,math,platform,resource,time,traceback
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner,gaussian_field,components,MODES
from voxelform.environment import Environment
from voxelform.context import ContextLibrary
from voxelform.shadows import sync
from tools.architecture_hypotheses import train,timed,summary,prepare
ROOT=Path(__file__).resolve().parents[1]

def make_env(config,**kwargs):return Environment(frontier_factory=functools.partial(PatchFacades,config=config),**kwargs)
def structure(mask):
    result=components(mask);p=np.pad(mask,1);boundary=0
    for axis in range(3):boundary+=np.abs(np.diff(p.astype(np.int8),axis=axis)).sum()
    result['boundaryFacesPerVoxel']=float(boundary/max(mask.sum(),1));result['voxels']=int(mask.sum());return result

def episode(policy,env,operation,seed,cap=192):
    torch.manual_seed(seed);prepare(env,operation);policy.clear();start=time.perf_counter();masks=[];attempts=[];failed=None
    while not env.terminal and len(masks)<cap:
        action=policy.choose(env,False,True);attempts.append(policy.generation_attempts)
        if action is None:failed=policy.last_rejection;break
        assert env.frontier.validate_mask(action.mask,action.operation)
        row=dict(raw=policy.raw_structure,applied=structure(action.mask),generatedVoxels=action.generated_voxels,
                 interiorVoxels=int((action.mask&~env.frontier.remove_skin).sum()) if action.operation==-1 else 0,
                 generationMs=policy.generation_seconds*1000,projectionMs=policy.projection_seconds*1000)
        env.apply(action,audit=False);masks.append(row)
        if env.edits%8==0:env.evaluate()
    env.evaluate();success=env.count==env.target_count
    return dict(operation=operation,success=success,termination='far_reached' if success else failed or 'action_cap',
                actions=len(masks),far=env.far,targetFar=env.target_far,targetCells=env.target_count,
                quality=float(env.quality),depthPenalty=env.metrics()['depthPenalty'],masks=masks,attempts=attempts,seconds=time.perf_counter()-start)

def evaluate(p,cases,seed):return [dict(case=tag,shape=e.shape,**episode(p,e,op,seed+10000+i)) for i,(tag,e,op) in enumerate(cases)]
def probes(device='cpu'):
    result=[]
    for shape in [(4,48,64),(8,64,96)]:
        c=PatchConfig();e=make_env(c,shape=shape,sun_count=8 if shape[0]==4 else 12,device=device)
        w=torch.as_tensor(e.frontier.remove_weights,device=device);legal=w>0;k=round(int(legal.sum())*.05)
        for mode in MODES:
            samples=[]
            for seed in range(30):
                torch.manual_seed(seed)
                if mode in ('independent','smooth'):field=torch.randn(shape,device=device)
                else:
                    coarse=tuple(math.ceil(n/s) for n,s in zip(shape,c.stride));field=F.interpolate(torch.randn((1,1,*coarse),device=device),size=shape,mode='trilinear',align_corners=False)[0,0]
                    if mode=='combined':field=gaussian_field(field,c.sigma)
                ids=field.masked_fill(~legal,-torch.inf).flatten().topk(k).indices;mask=torch.zeros(shape,dtype=torch.bool,device=device);mask.flatten()[ids]=1
                samples.append(structure(mask.cpu().numpy()))
            result.append(dict(device=device,shape=shape,mode=mode,exactEditBudget=k,samples=samples,
                note='Constant-mean, fixed-count correlation probe. Independent/smoothed-constant-mean Bernoulli conditioned on count is uniform-k; rank white noise gives that exact distribution. Not a learned-quality test.'))
    return result

def benchmark(c,e,device,iterations):
    p=PatchLearner(device,config=c);p.fields(e);p.fields(e);times={k:[] for k in ['generation','projection','apply','forward']};masks=[]
    for _ in range(iterations):
        with torch.no_grad():_,ms=timed(lambda:p.fields(e),device)
        times['forward'].append(ms);a,ms=timed(lambda:p.choose(e,False,True),device)
        if a is None:raise RuntimeError('Generated mask rejected')
        times['generation'].append(ms);times['projection'].append(p.projection_seconds*1000);masks.append(structure(a.mask))
        _,ms=timed(lambda:e.apply(a,audit=False),device);times['apply'].append(ms)
    for _ in range(4):
        a=p.choose(e,True)
        if a is None:p.record(-.1);break
        before=e.far_error;e.apply(a,audit=False);p.record(before-e.far_error)
    stats,ms=timed(lambda:p.finish(p.bootstrap(e)),device)
    return dict(timings={k:summary(v) for k,v in times.items()},learningMs=ms,learning=stats,masks=masks,
                parameters=sum(v.numel() for v in p.model.parameters()),rssHighWaterBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                mpsAllocatedBytes=torch.mps.current_allocated_memory() if device=='mps' else None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,default=256);parser.add_argument('--iterations',type=int,default=10);args=parser.parse_args();torch.set_num_threads(1)
    path=ROOT/'results/coherent-masks-2026-11-09.json';configs=[PatchConfig(mode=m,eligibility=e) for e in ('facade','depth') for m in MODES]
    result=dict(seeds=[123,456,789],budget=args.budget,platform=platform.platform(),torch=torch.__version__,cpuThreads=1,
                plan='agent_notes/coherent_masks_2026-11-09/PLAN.md',fixedCountProbes=probes(),learning=[],benchmark=[],failures=[],
                sourceHashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),*sorted((ROOT/'voxelform').glob('*.py'))]})
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    library=ContextLibrary();context=library.load('m_bcn_0093')
    for c in configs:
        envs=[make_env(c,shape=(4,12,16),sun_count=4,seed=100+i,plan=plan) for i,plan in enumerate(('rectangle','triangle','courtyard','branch'))]
        cases=[]
        for i,plan in enumerate(('rectangle','triangle','courtyard','branch')):
            e=make_env(c,shape=(4,16,20),sun_count=4,seed=9100+i,plan=plan);cases.extend([(plan,e,'subtract'),(plan,e,'add')])
        real=make_env(c,context=context,sun_count=8,seed=9200);cases.extend([('Barcelona',real,'subtract'),('Barcelona',real,'add')])
        for seed in result['seeds']:
            row=dict(mode=c.mode,eligibility=c.eligibility,seed=seed)
            try:
                p=PatchLearner(config=c,seed=seed);row['initial']=evaluate(p,cases,seed);row['training']=train(p,envs,seed,args.budget);row['trained']=evaluate(p,cases,seed)
                checkpoint=ROOT/'results/checkpoints/coherent-masks-2026-11-09'/f'{c.eligibility}-{c.mode}-{seed}.pt';checkpoint.parent.mkdir(exist_ok=True,parents=True);torch.save(p.state(),checkpoint);row['checkpoint']=str(checkpoint.relative_to(ROOT))
                print(c.eligibility,c.mode,seed,sum(v['success'] for v in row['trained']),'/10',flush=True);del p;gc.collect()
            except Exception:row['error']=traceback.format_exc();result['failures'].append(row['error']);print(row['error'],flush=True)
            result['learning'].append(row);save()
        del envs,cases,real;gc.collect()
    for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
        for shape,suns in [((4,48,64),8),((8,64,96),12)]:
            for c in configs:
                row=dict(device=device,shape=shape,mode=c.mode,eligibility=c.eligibility,sunSamples=suns,contextBuildings=16)
                try:
                    e,ms=timed(lambda:make_env(c,shape=shape,sun_count=suns,device=device,seed=123),device);row['setupMs']=ms
                    row.update(benchmark(c,e,device,args.iterations));print('benchmark',device,shape,c.eligibility,c.mode,round(row['timings']['generation']['medianMs'],2),flush=True);del e;gc.collect()
                except Exception:row['error']=traceback.format_exc();result['failures'].append(row['error']);print(row['error'],flush=True)
                result['benchmark'].append(row);save()
                if device=='mps':torch.mps.empty_cache()
    print('Saved',path,flush=True)
if __name__=='__main__':main()
