"""Posthoc hypothesis: pack operation with FAR before GPU forward, avoiding a host transfer."""
import gc,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tools.architecture_hypotheses import ROOT,Environment,ExperimentalNetwork,inputs,timed,summary

def forward_packed(m,observation,condition):
    # Same mathematical network/weights as single-head forward; only input packing differs.
    goal=condition[:3];three=m.config['encoder']!='2d'
    x=observation.permute(1,0,2,3)[None] if three else observation
    encoded=m.encoder(x)
    if m.config['encoder']=='multiscale':
        d1=m.down1(F.avg_pool3d(encoded,(1,2,2),ceil_mode=True));d2=m.down2(F.avg_pool3d(d1,(1,2,2),ceil_mode=True))
        encoded=encoded+F.interpolate(d1,size=encoded.shape[-3:],mode='trilinear',align_corners=False)+F.interpolate(d2,size=encoded.shape[-3:],mode='trilinear',align_corners=False)
    floor=encoded[0].permute(1,0,2,3).mean((-1,-2)) if three else encoded.mean((-1,-2));context=torch.cat((floor.mean(0),floor.amax(0)))
    condition=m.goal_encoder(condition);encoded=encoded+(condition[None,:,None,None,None] if three else condition[None,:,None,None])
    out=m.mask_head(encoded);out=out[0] if three else out.permute(1,0,2,3)
    return out.expand(2,-1,-1,-1),m.critic(torch.cat((context,goal))).squeeze()

def main():
    torch.set_num_threads(1);r=dict(iterations=30,facade=True,seed=321,sourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),note='Posthoc follow-up to MPS slowdown: three interleaved forwards on frozen inputs, same single-head weights for original/packed path. Packing cost excluded, so this isolates forward overhead, not an end-to-end speedup.',cases=[],failures=[])
    out=ROOT/'results/architecture-packed-condition-latency-2026-11-08.json'
    for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
        for scale,shape,suns in [('L',(4,48,64),8),('XL',(8,64,96),12)]:
            env=Environment(shape,device=device,sun_count=suns,seed=123);obs=inputs(env,True);g=torch.tensor(env.goal_features,device=device);packed=torch.tensor([*env.goal_features,-1],device=device)
            for enc in ('2d','3d','multiscale'):
                row=dict(device=device,scale=scale,encoder=enc)
                try:
                    dual=ExperimentalNetwork(dict(single=False,facade=True,encoder=enc)).to(device);single=ExperimentalNetwork(dict(single=True,facade=True,encoder=enc)).to(device)
                    calls=[lambda:dual(obs,g,-1),lambda:single(obs,g,-1),lambda:forward_packed(single,obs,packed)]
                    values=[[],[],[]];rng=np.random.default_rng(321)
                    with torch.no_grad():
                        for a,b in zip(calls[1](),calls[2]()):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
                        for _ in range(5):
                            for call in calls:call()
                        for _ in range(30):
                            for i in rng.permutation(3):_,ms=timed(calls[i],device);values[i].append(ms)
                    a,b,c=map(np.median,values);row.update(dual=summary(values[0]),singleOriginal=summary(values[1]),singlePacked=summary(values[2]),packedVsDualSpeedupPercent=float(100*(a-c)/a),packedVsOriginalSingleSpeedupPercent=float(100*(b-c)/b),equivalentSingleOutputs=True)
                    print(device,scale,enc,'packed vs dual %',round(row['packedVsDualSpeedupPercent'],2),'vs original single %',round(row['packedVsOriginalSingleSpeedupPercent'],2),flush=True)
                    del dual,single,calls;gc.collect()
                    if device=='mps':torch.mps.empty_cache()
                except Exception:row['error']=traceback.format_exc();r['failures'].append(row['error']);print(row['error'],flush=True)
                r['cases'].append(row);out.write_text(json.dumps(r,indent=2)+'\n')
            del env,obs,g,packed;gc.collect()
if __name__=='__main__':main()
