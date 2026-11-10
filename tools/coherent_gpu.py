"""Final CPU/MPS pipeline profiles: exact 2D depth, fixed-budget vectorized retries."""
import json,functools,time,gc,hashlib,traceback,platform
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from voxelform.fields import facade_depth
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner
from voxelform.environment import Environment
from voxelform.shadows import sync
from tools.architecture_hypotheses import timed,summary
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(1);result=dict(platform=platform.platform(),torch=torch.__version__,cpuThreads=1,cases=[],failures=[],note='20 warmed draws per fixed full-envelope state. CPU uses SciPy depth; MPS uses exact tensor same-floor depth. Every correlated draw tested with the same cap/column/void rules. Host connectivity remains native NumPy/SciPy. Cold costs and failed draws retained.',sourceHashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['voxelform/patch_policy.py','voxelform/fields.py','voxelform/environment.py']});out=ROOT/'results/coherent-gpu-adaptive-2026-11-09.json'
    for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
        for shape,suns in [((4,48,64),8),((8,64,96),12)]:
            for batched in (False,True):
                row=dict(device=device,shape=shape,sunSamples=suns,contextBuildings=16,batched=batched)
                cfg=PatchConfig(mode='combined',eligibility='depth',minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=batched,maximum_edit_voxels=256,vertical_interior=True)
                try:
                    factory=functools.partial(PatchFacades,config=cfg,depth_device=device if device!='cpu' else None)
                    # A CPU-depth call is an error on the GPU path, not a hidden fallback.
                    guard=patch('voxelform.patch_policy.facade_depth',side_effect=AssertionError('CPU depth fallback')) if device=='mps' else patch('voxelform.patch_policy.facade_depth',wraps=facade_depth)
                    with guard:
                        e,setup=timed(lambda:Environment(shape,sun_count=suns,seed=123,device=device,frontier_factory=factory),device);row['setupMs']=setup;p=PatchLearner(device,config=cfg)
                        _,cold=timed(lambda:p.choose(e,False),device);row['coldGenerationMs']=cold
                        for _ in range(2):p.choose(e,False)
                        times=[];attempts=[];phases=[];failed=0;last=None
                        for i in range(20):
                            torch.manual_seed(1000+i);a,ms=timed(lambda:p.choose(e,False),device);times.append(ms);attempts.append(p.generation_attempts)
                            phases.append(dict(forwardMs=p.model_seconds*1000,samplingMs=getattr(p,'sampling_seconds',0)*1000,transferMs=getattr(p,'transfer_seconds',0)*1000,geometryMs=p.projection_seconds*1000))
                            if a is None:failed+=1
                            else:last=a
                        row.update(generation=summary(times),attempts=attempts,phaseSamples=phases,failedDraws=failed)
                        if last:
                            _,ms=timed(lambda:e.apply(last,audit=False),device);row['applyDepthMs']=ms
                            ref=facade_depth(e.frontier.occupied,e.frontier.exterior_air);row['depthMaxErrorMetres']=float(np.abs(e.depth.cpu().numpy()-ref).max());assert row['depthMaxErrorMetres']<1e-5
                        for _ in range(4):
                            a=p.choose(e,True)
                            if a is None:p.record(-.1);break
                            before=e.far_error;e.apply(a,audit=False);p.record(before-e.far_error)
                        stats,ms=timed(lambda:p.finish(p.bootstrap(e)),device);row['backwardUpdateMs']=ms;row['learning']=stats
                        if device=='mps':row['mpsAllocatedBytes']=torch.mps.current_allocated_memory();row['mpsDriverBytes']=torch.mps.driver_allocated_memory()
                        print(device,shape,batched,'warm',round(row['generation']['medianMs'],2),'cold',round(cold,1),'attempts',round(np.mean(attempts),1),'depth error',row.get('depthMaxErrorMetres'),flush=True)
                except Exception:row['error']=traceback.format_exc();result['failures'].append(row['error']);print(row['error'],flush=True)
                result['cases'].append(row);out.write_text(json.dumps(result,indent=2)+'\n');gc.collect()
                if device=='mps':torch.mps.empty_cache()
if __name__=='__main__':main()
