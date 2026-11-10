"""Interleaved head comparison on identical frozen L/XL inputs; no geometry edits."""
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from tools.architecture_hypotheses import ExperimentalNetwork,inputs,timed,ROOT,summary
from voxelform.environment import Environment

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=30);args=parser.parse_args();torch.set_num_threads(1)
    path=ROOT/'results/architecture-paired-head-latency-2026-11-08.json';r=dict(iterations=args.iterations,seed=321,torch=torch.__version__,cases=[],failures=[],sourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),note='Paired randomized order, fixed full-envelope observation, no inputs/topology/rendering timed; CPU one thread, synchronized actual devices.')
    devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
    for device in devices:
        for scale,shape,suns in [('L',(4,48,64),8),('XL',(8,64,96),12)]:
            env=Environment(shape,device=device,sun_count=suns,seed=123)
            for encoder in ('2d','3d','multiscale'):
                for facade in (False,True):
                    row=dict(device=device,scale=scale,encoder=encoder,facade=facade,shape=shape)
                    try:
                        obs=inputs(env,facade);g=torch.tensor(env.goal_features,device=device)
                        models=[ExperimentalNetwork(dict(single=s,facade=facade,encoder=encoder)).to(device) for s in (False,True)]
                        samples=[[],[]];rng=np.random.default_rng(321)
                        with torch.no_grad():
                            for _ in range(5):
                                for m in models:m(obs,g,-1)
                            for _ in range(args.iterations):
                                for i in rng.permutation(2):
                                    _,ms=timed(lambda:models[i](obs,g,-1),device);samples[i].append(ms)
                        a,b=map(np.median,samples);row.update(dual=summary(samples[0]),single=summary(samples[1]),singleSpeedupPercent=float(100*(a-b)/a))
                        print(device,scale,encoder,facade,'single speedup %',round(row['singleSpeedupPercent'],2),flush=True)
                        del models,obs,g;gc.collect()
                        if device=='mps':torch.mps.empty_cache()
                    except Exception:row['error']=traceback.format_exc();r['failures'].append(row['error']);print(row['error'],flush=True)
                    r['cases'].append(row);path.write_text(json.dumps(r,indent=2)+'\n')
            del env;gc.collect()
if __name__=='__main__':main()
