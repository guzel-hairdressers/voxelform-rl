"""Posthoc replication of heuristic controls across the same three paired seeds."""
import json
import torch
from tools.architecture_hypotheses import ROOT,Control,Environment,ContextLibrary,evaluate

def main():
    torch.set_num_threads(1);raw=json.loads((ROOT/'results/architecture-hypotheses-2026-11-08.json').read_text());rows=[dict(seed=123,**c) for c in raw['controls']]
    cases=[]
    for i,p in enumerate(('rectangle','triangle','courtyard','branch')):
        env=Environment((4,16,20),sun_count=4,seed=9100+i,plan=p);cases.extend([(f'heldout_{p}',env,'subtract'),(f'heldout_{p}',env,'add')])
    real=Environment(context=ContextLibrary().load('m_bcn_0093'),sun_count=8,seed=9200);cases.extend([('cached_Barcelona',real,'subtract'),('cached_Barcelona',real,'add')])
    for seed in (456,789):
        for kind in ('uniform','sky_greedy'):
            rows.append(dict(kind=kind,seed=seed,evaluation=evaluate(Control(kind),cases,seed)));print(kind,seed,flush=True)
    (ROOT/'results/architecture-controls-replicated-2026-11-08.json').write_text(json.dumps(dict(note='Control replication added after observing pilot/main results, preserving original seed123 control and adding seeds456/789 with identical evaluation protocol.',controls=rows),indent=2)+'\n')
if __name__=='__main__':main()
