"""Reproducible method/scale tests, not a convergence or design-quality claim."""
import argparse,hashlib,json,platform,time,traceback
from pathlib import Path
import numpy as np
import torch
from voxelform.environment import Environment,context_boxes
from voxelform.shadows import (sun_path,sync,DirectShadows,BoundaryShadows,DepthShadows,dda_paths)
from voxelform.fields import receivers

def timed(fn,device):
    sync(device);start=time.perf_counter();result=fn();sync(device)
    return result,(time.perf_counter()-start)*1000

def percentile(values):return dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),raw=values)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',default='results/benchmark-cpu.json');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(123);rng=np.random.default_rng(123);device=args.device
    root=Path(__file__).resolve().parents[1]
    result=dict(seed=123,device=device,torch=torch.__version__,platform=platform.platform(),cpuThreads=1,
        note='Synthetic method/scale tests, not real-site generalization or learning improvement',
        sourceHashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'voxelform').glob('*.py'))},cases=[],failures=[])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    def save():out.write_text(json.dumps(result,indent=2))
    for label,nx,ny,nz,samples,neighbors in [('L',64,48,4,8,10),('XL',96,64,8,12,20)]:
        case=dict(scale=label,shape=[nz,ny,nx],sunSamples=samples,neighbors=neighbors,static=[]);result['cases'].append(case)
        try:
            suns=sun_path(samples,grid_yaw=23);boxes=context_boxes(nx,ny,neighbors,123,nz*3);bounds=((0,0,0),(nx,ny,nz*3))
            points=rng.uniform(bounds[0],bounds[1],(8192,3)).astype(np.float32);p=torch.tensor(points,device=device)
            direct=DirectShadows(boxes,suns,device);reference,reference_ms=timed(lambda:direct.query(p),device)
            case['directReferenceMs']=reference_ms
            for name,cls in [('boundary',BoundaryShadows),('depth',DepthShadows)]:
                for pitch in (1.,.5):
                    method,setup_ms=timed(lambda:cls(boxes,suns,bounds,pitch,device),device)
                    times=[]
                    for _ in range(5):pred,ms=timed(lambda:method.query(p),device);times.append(ms)
                    wrong=pred!=reference
                    entry=dict(method=name,pitch=pitch,setupMs=setup_ms,queryMs=percentile(times),queryCount=len(p),mapBytes=method.bytes,
                        mismatchFraction=float(wrong.float().mean().cpu()),falseClearFraction=float((pred&~reference).float().mean().cpu()),
                        falseShadowFraction=float((~pred&reference).float().mean().cpu()))
                    if name=='boundary':entry.update(faceUnion=sorted(method.face_union),intruderCount=method.intruder_count)
                    case['static'].append(entry);save();print(label,name,pitch,entry,flush=True)
            # End-to-end edits share the same dynamic self-shadow and proxy fields.
            # This scene has 16 neighbours; static tests above have 10/20 explicitly.
            env,setup_ms=timed(lambda:Environment((nz,ny,nx),seed=123,device=device,sun_count=samples,plan='branch'),device)
            case['environmentSetupMs']=setup_ms;case['environmentNeighbors']=len(env.boxes)
            edits=[]
            for step in range(8):
                # Identical proposal values on CPU and GPU; device RNGs differ.
                params=rng.random((24,7),dtype=np.float32);kind=rng.integers(0,5,24,dtype=np.int64)
                (masks,features,valid),candidate_ms=timed(lambda:env.candidates(params=params,kind=kind),device)
                index=int(valid[1:-1].nonzero()[0].detach().cpu())+1
                _,ms=timed(lambda:env.apply(index,masks),device)
                edits.append(dict(step=step,editMs=ms,candidateMs=candidate_ms,**env.metrics()))
            case['edits']=edits
            ids,rp,n,cells,faces=env.receiver_data
            inds=np.linspace(0,len(rp)-1,min(512,len(rp)),dtype=int);points=rp[inds].cpu().numpy()+n[inds].cpu().numpy()*1e-4
            exact=[];flat=np.append(env.occupied.cpu().numpy().flatten(),False)
            for s in env.suns_np:
                paths,_=dda_paths(points,s,env.shape);exact.append(~flat[paths].any(1))
            exact=torch.tensor(np.array(exact),device=device);pred=env.dynamic.query(rp[inds],n[inds]);front=(env.suns@n[inds].T)>1e-6
            case['selfShadowValidation']=dict(receiverCount=len(inds),frontFacingComparisons=int(front.sum().cpu()),
                mismatchFraction=float((pred[front]!=exact[front]).float().mean().cpu()),
                falseClearFraction=float((pred[front]&~exact[front]).float().mean().cpu()),
                falseShadowFraction=float((~pred[front]&exact[front]).float().mean().cpu()))
            from voxelform.policy import Learner
            policy=Learner(device);timings=[]
            for _ in range(4):
                (a,m),ms=timed(lambda:policy.choose(env),device);timings.append(ms);q=env.quality.detach().clone();env.apply(a,m);policy.record(env.quality-q-.002)
            stats,ms=timed(policy.finish,device);case['policy']=dict(actionMs=percentile(timings),backpropMs=ms,**stats)
            print(label,'edits median ms',np.median([e['editMs'] for e in edits]),'self shadow',case['selfShadowValidation'],flush=True)
        except Exception:
            result['failures'].append(dict(scale=label,traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
        save()
    print('Saved',out,flush=True)

if __name__=='__main__':main()
