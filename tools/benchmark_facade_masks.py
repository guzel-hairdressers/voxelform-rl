"""L/XL facade-mask latency, split from policy and environmental audit work.

These are synthetic method timings, not a quality/convergence benchmark or a
claim of a fully GPU-resident pipeline. Geometry/topology runs in NumPy/SciPy C
kernels on CPU; tensors/policy are measured on actual available CPU/MPS/CUDA.
"""
import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import scipy
import torch

from voxelform.actions import ACTION_SPACE
from voxelform.environment import Environment
from voxelform.policy import Learner
from voxelform.shadows import sync


def timed(call,device):
    sync(device);stamp=time.perf_counter();value=call();sync(device)
    return value,1000*(time.perf_counter()-stamp)


def summary(values):
    return dict(medianMs=float(np.median(values)),p95Ms=float(np.percentile(values,95)),rawMs=values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=15)
    parser.add_argument('--output',default='results/direct-neural-masks-2026-11-07.json');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(123)
    devices=['cpu']
    if torch.backends.mps.is_available():devices.append('mps')
    if torch.cuda.is_available():devices.append('cuda')
    root=Path(__file__).resolve().parents[1]
    result=dict(actionSpace=ACTION_SPACE,seed=123,iterations=args.iterations,platform=platform.platform(),
                torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,cpuThreads=1,
                geometryDevice='cpu_numpy_scipy_native',devices=devices,cases=[],failures=[],
                note='Synthetic L/XL method latency, no learning/generalization claim. Browser preview left open; no rendering is included in timed functions.',
                sourceHashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'voxelform').glob('*.py'))})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    def save():output.write_text(json.dumps(result,indent=2)+'\n')
    for device in devices:
        for scale,shape,sun_count in [('L',(4,48,64),8),('XL',(8,64,96),12)]:
            case=dict(scale=scale,device=device,shape=shape,sunSamples=sun_count,contextBuildings=16)
            try:
                env,case['setupMs']=timed(lambda:Environment(shape,device=device,sun_count=sun_count,seed=123),device)
                policy=Learner(device)
                with torch.no_grad():
                    _,case['modelWarmupMs']=timed(lambda:policy.fields(env),device)
                values={k:[] for k in ('facade','neuralMaskTotal','modelForward','constraintProjection','apply','environment')}
                changed=[];generated=[]
                for i in range(args.iterations):
                    _,ms=timed(env.frontier.refresh,device);values['facade'].append(ms)
                    action,ms=timed(lambda:policy.choose(env,training=False,greedy=True),device)
                    if action is None:raise RuntimeError('Neural mask generation failed before FAR target')
                    values['neuralMaskTotal'].append(ms);values['modelForward'].append(policy.model_seconds*1000)
                    values['constraintProjection'].append(policy.projection_seconds*1000)
                    changed.append(action.count);generated.append(action.generated_voxels)
                    _,ms=timed(lambda:env.apply(action,audit=False),device);values['apply'].append(ms)
                    if i%5==4:
                        _,ms=timed(env.evaluate,device);values['environment'].append(ms)
                case.update(timings={k:summary(v) for k,v in values.items() if v},
                            changedVoxels=changed,generatedVoxels=generated,maskBytes=int(action.mask.nbytes),
                            staticMapBytes=int(env.static.bytes),dynamicMapBytes=int(env.dynamic.bytes))
                for _ in range(4):
                    action=policy.choose(env,training=True)
                    if action is None:raise RuntimeError('Neural training mask rejected')
                    before=env.far_error;env.apply(action,audit=False);policy.record(before-env.far_error)
                stats,learning_ms=timed(lambda:policy.finish(policy.bootstrap(env)),device)
                case['learningMs']=learning_ms;case['learningStats']=stats;case['learningRolloutMasks']=4
                case['processHighWaterRssBytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if platform.system()=='Darwin' else 1024)
                if device=='mps':case['mpsAllocatedBytes']=torch.mps.current_allocated_memory();case['mpsDriverAllocatedBytes']=torch.mps.driver_allocated_memory()
                print(device,scale,{k:round(v['medianMs'],3) for k,v in case['timings'].items()},flush=True)
                del action,env,policy;gc.collect()
                if device=='mps':torch.mps.empty_cache()
            except Exception:
                case['error']=traceback.format_exc();result['failures'].append(dict(device=device,scale=scale,error=case['error']))
                print(case['error'],flush=True)
            result['cases'].append(case);save()
    print('Saved',output,flush=True)


if __name__=='__main__':main()
