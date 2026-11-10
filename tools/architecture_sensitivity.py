"""Post-training feature-use diagnostics, not an outcome or causality score."""
import json
from pathlib import Path
import torch
from tools.architecture_hypotheses import ROOT,ExperimentalLearner,Environment,inputs,prepare

def main():
    torch.set_num_threads(1);raw=json.loads((ROOT/'results/architecture-hypotheses-2026-11-08.json').read_text());rows=[]
    env=Environment((4,16,20),seed=9100,sun_count=4)
    for r in raw['learning']:
        if 'error' in r:continue
        p=ExperimentalLearner(r['config'],r['seed']);p.model.load_state_dict(torch.load(ROOT/r['checkpoint'],weights_only=True)['model'])
        for operation in ('subtract','add'):
            prepare(env,operation);op=-1 if operation=='subtract' else 1
            obs=inputs(env,r['config']['facade']);goal=torch.tensor(env.goal_features);legal=torch.tensor(env.frontier.eligible(op))
            with torch.no_grad():
                out,_=p.model(obs,goal,op);prob=out[0 if op==-1 else 1].sigmoid()
                row=dict(variant=r['variant'],seed=r['seed'],operation=operation)
                if r['config']['facade']:
                    zero=obs.clone();zero[:,5:]=0;other,_=p.model(zero,goal,op)
                    row['zeroFacadeMeanProbabilityChange']=float((prob-other[0 if op==-1 else 1].sigmoid()).abs()[legal].mean())
                if r['config']['single']:
                    other,_=p.model(obs,goal,-op)
                    row['flipOperationMeanProbabilityChange']=float((prob-other[0].sigmoid()).abs()[legal].mean())
            rows.append(row)
    path=ROOT/'results/architecture-feature-sensitivity-2026-11-08.json';path.write_text(json.dumps(dict(note='Frozen weights; same held-out rectangle, illegal alternate inputs only for feature-use diagnostics. This is not performance evidence.',cases=rows),indent=2)+'\n');print('Saved',len(rows),'feature-use diagnostics')
if __name__=='__main__':main()
