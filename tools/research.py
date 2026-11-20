"""Headless, auditable train/evaluate entry point for the organic RL task."""
from __future__ import annotations
import argparse
from dataclasses import fields
import functools
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from types import SimpleNamespace
from voxelform.context import ContextLibrary
from voxelform.cores import solve_cores
from voxelform.environment import Environment
from voxelform.experiment import (ExperimentConfig,archive_source,read_checkpoint,restore_rng,rng_state,runtime_manifest,
    save_checkpoint,seed_everything,source_manifest,write_json)
from voxelform.objectives import organic_objective,organic_step_reward,organic_terminal_reward
from voxelform.patch_policy import PatchConfig,PatchFacades
from voxelform.shadows import device_name,sync
from voxelform.unified import UnifiedLearner,commit


class Scenes:
    def __init__(self,config):
        self.config=config;self.device=device_name(config.device);self.envs={};self.manifest={};self.setup=[]
        self.patch=PatchConfig(mode='combined',allocation='feedback',eligibility='depth',facade_layers=2,
            minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,
            vertical_interior=False,budget_reserve_fraction=.2)

    def get(self,site):
        if site in self.envs:return self.envs[site]
        t=time.perf_counter();c=self.config;context=None;plan='rectangle'
        scene_seed=int.from_bytes(hashlib.sha256(site.encode()).digest()[:4],'little')%(2**31)
        if site.startswith('synthetic:'):
            plan=site.split(':',1)[1];source_hash=hashlib.sha256(f'{site}:{scene_seed}:{c.synthetic_width}:{c.synthetic_length}'.encode()).hexdigest()
        else:
            library=ContextLibrary();record=next((r for r in library.catalog() if r['site_id']==site),None)
            if record is None:raise ValueError(f'Cached scene unavailable: {site}; see docs/DATA_CARD.md')
            source_hash=hashlib.sha256(library._scene_path(record).read_bytes()).hexdigest()
            context=library.load(site,floors=c.ceiling_storeys,grid_alignment='auto_fit')
        env=Environment((c.ceiling_storeys,c.synthetic_width,c.synthetic_length),seed=scene_seed,plan=plan,context=context,
            device=self.device,sun_count=8,target_far=c.target_far,far_tolerance=c.far_tolerance,
            frontier_factory=functools.partial(PatchFacades,config=self.patch,depth_device=self.device),
            fast_self_shadows=True,fast_diffuse=True,cores_enabled=c.cores_enabled,
            min_core_spacing=c.min_core_spacing,max_core_dist=c.max_core_dist)
        env.precompute_external();self.envs[site]=env
        self.manifest[site]=dict(sourceSHA256=source_hash,sceneSeed=scene_seed,shape=list(env.shape),parcelArea=env.plot_area,
            allowedSHA256=hashlib.sha256(np.packbits(env.allowed.cpu().numpy()).tobytes()).hexdigest(),
            contextBuildings=len(context['buildings']) if context else 16,gridAngle=(context or {}).get('gridAngle',0.),
            center=[context.get('lat'),context.get('lon')] if context else None,contextRadius=(context or {}).get('radius',100.))
        self.setup.append(dict(site=site,seconds=time.perf_counter()-t))
        return env

    def prepare(self):
        groups=(self.config.train_sites,self.config.validation_sites,self.config.test_sites)
        for group in groups:
            for site in group:self.get(site)
        # Reject cross-split context circles that overlap when georeferencing is
        # available. Distinct site identifiers alone do not prevent spatial leakage.
        import math
        for i,group in enumerate(groups):
            for later in groups[i+1:]:
                for left in group:
                    for right in later:
                        a,b=self.manifest[left],self.manifest[right]
                        if not a['center'] or not b['center'] or any(v is None for v in (*a['center'],*b['center'])):continue
                        la,lo=map(math.radians,a['center']);lb,lp=map(math.radians,b['center'])
                        q=math.sin((lb-la)/2)**2+math.cos(la)*math.cos(lb)*math.sin((lp-lo)/2)**2
                        distance=2*6371000*math.asin(min(1.,math.sqrt(q)))
                        if distance<=float(a['contextRadius'])+float(b['contextRadius']):
                            raise ValueError(f'Spatial context leakage between {left} and {right}')


def make_policy(config,device,seed):
    patch=PatchConfig(mode='combined',allocation='feedback',eligibility='depth',facade_layers=2,
        minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=False,budget_reserve_fraction=.2)
    p=UnifiedLearner(device,seed,patch,amount_feedback=True,net_guided=True,organic_3d=True,spatial_aux_weight=config.spatial_aux_weight)
    p.evaluation_temperature=config.evaluation_temperature;p.evaluation_noise_scale=config.evaluation_noise_scale
    return p


def episode(policy,env,config,training):
    solid=env.allowed.clone();solid[config.initial_storeys:]=False;env.restore_occupancy(solid)
    env.organic_3d_growth=True;env.floor_copy_growth=False;env._ln6_cache=None
    env.unified_limit=max(8,env.workload.limit+8);env.unified_actions=0;env.termination=None
    start=time.perf_counter();trace=[];reward_sum=0.;tape_start=len(policy.tape)
    _,diag=organic_objective(env)
    previous_core=float(diag.get('coreScore') or 0.)
    for _ in range(env.unified_limit):
        move=policy.choose(env,training=training);env.unified_actions+=1
        op=policy.last_operation
        if op=='finish':env.termination='far_reached';reward=0.
        elif move is None:reward=-.01
        else:
            commit(env,move,validated=True)
            delta=0.
            if training and config.cores_enabled:
                env.update_cores();m=env.core_metrics
                score=max(0.,(1.-float(m['coreUncoveredFraction']))*(1.-.5*float(m['coreDistBadness'])))
                delta=score-previous_core;previous_core=score
            reward=organic_step_reward(env,delta)
            if env.edits%config.audit_steps==0:env.evaluate()
        if training and policy.last_attempt_recorded:policy.record(reward)
        reward_sum+=reward
        trace.append(dict(attempt=env.unified_actions,operation=op,accepted=move is not None,
            removed=int(move.remove.sum()) if move else 0,added=int(move.add.sum()) if move else 0,far=env.far))
        if env.terminal:break
    if not env.terminal:env.termination='far_reached' if env.far_satisfied else 'step_budget_exhausted'
    env.evaluate();obj,diag=organic_objective(env)
    terminal=organic_terminal_reward(env,obj);reward_sum+=terminal
    learning=None
    if training and len(policy.tape)>tape_start:
        policy.add_reward(terminal);policy.mark_done()
        if len(policy.tape)>=config.rollout_steps:learning=policy.finish(0.)
    reference=None
    if not training:
        # A common fixed referee makes cores-on/off and spacing ablations
        # comparable without showing extra channels to the acting policy.
        res=solve_cores(env.occupied,min_core_spacing=10,max_core_dist=12)
        referee=SimpleNamespace(**env.__dict__)
        referee.cores_enabled=True;referee.max_core_dist=12
        referee.core_mask=res.core_mask;referee.core_distance=res.core_distance
        referee.core_dist_badness=res.core_dist_badness;referee.core_metrics=res.metrics
        score,diagnostics=organic_objective(referee)
        reference=dict(quality=score,diagnostics=diagnostics,minCoreSpacing=10,maxCoreDist=12)
    sync(env.device)
    return dict(reference=reference,success=env.termination=='far_reached',termination=env.termination,far=env.far,farError=env.far_error,
        steps=env.edits,attempts=env.unified_actions,budget=env.unified_limit,seconds=time.perf_counter()-start,
        reward=reward_sum,quality=float(obj),environment={k:float(v.detach().cpu()) for k,v in env.values.items()},
        diagnostics=diag,learning=learning,trace=trace)


def evaluate(policy,scenes,config,split):
    """Common random eval seeds; no optimizer step or perturbation of training RNG."""
    state=rng_state();rows=[]
    try:
        for site in getattr(config,split+'_sites'):
            env=scenes.get(site)
            for seed in config.evaluation_seeds:
                seed_everything(seed,config.strict_determinism)
                rows.append(dict(split=split,site=site,evaluationSeed=seed,**episode(policy,env,config,False)))
    finally:restore_rng(state)
    return rows


def summarize(rows):
    # Include unsuccessful episodes in every reported denominator/mean.
    if not rows:return dict(count=0)
    return dict(count=len(rows),farSuccessRate=statistics.mean(float(r['success']) for r in rows),
        meanQuality=statistics.mean(r['quality'] for r in rows),meanReferenceQuality=statistics.mean(r['reference']['quality'] for r in rows),meanSeconds=statistics.mean(r['seconds'] for r in rows),
        meanUsableGoodArea=statistics.mean(r['diagnostics']['usableGoodArea'] for r in rows),
        meanDirectSun=statistics.mean(r['environment']['directSunProxy'] for r in rows),
        meanDiffuseSky=statistics.mean(r['environment']['diffuseFacadeProxy'] for r in rows))


def aggregate(summaries):
    out={}
    for split in ('validation','test'):
        rows=[s[split] for s in summaries if split in s]
        if not rows:continue
        stats={}
        for key in ('farSuccessRate','meanQuality','meanReferenceQuality','meanUsableGoodArea','meanDirectSun','meanDiffuseSky'):
            vals=[r[key] for r in rows];ci=None
            if len(vals)>=3:
                rng=np.random.default_rng(2026);boots=rng.choice(vals,size=(5000,len(vals))).mean(1)
                ci=np.quantile(boots,[.025,.975]).tolist()
            stats[key]=dict(mean=statistics.mean(vals),stdAcrossModelSeeds=statistics.stdev(vals) if len(vals)>1 else None,
                bootstrap95AcrossModelSeeds=ci)
        out[split]=dict(modelSeeds=len(rows),metrics=stats)
    return out


def train(args):
    config=ExperimentConfig.read(args.config);torch.set_num_threads(config.threads)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    seed_everything(config.seeds[0],config.strict_determinism)
    manifest=runtime_manifest();manifest['configHash']=config.fingerprint
    manifest['resumeFrom']=str(Path(args.resume).resolve()) if args.resume else None
    write_json(out/'config.json',config.to_dict());write_json(out/'manifest.json',manifest);archive_source(out/'source.tar.gz',manifest)
    scenes=Scenes(config);scenes.prepare()
    write_json(out/'scenes.json',scenes.manifest);write_json(out/'setup.json',scenes.setup)
    summaries=[];saved=read_checkpoint(args.resume,config) if args.resume else None
    if saved and saved['scenes']!=scenes.manifest:raise ValueError('Scene content changed since checkpoint')
    for seed in ([saved['seed']] if saved else config.seeds):
        seed_everything(seed,config.strict_determinism);policy=make_policy(config,scenes.device,seed)
        folder=out/f'seed-{seed}';folder.mkdir();start=0
        if saved:policy.load(saved['policy']);start=int(saved['nextEpisode']);restore_rng(saved['rng'])
        else:
            baseline=evaluate(policy,scenes,config,'validation');write_json(folder/'untrained-validation.json',baseline)
        t=time.perf_counter();completed=start
        with (folder/'train.jsonl').open('x') as ledger:
            for i in range(start,config.episodes):
                site=config.train_sites[i%len(config.train_sites)]
                row=dict(episode=i+1,modelSeed=seed,site=site,**episode(policy,scenes.get(site),config,True))
                ledger.write(json.dumps(row,allow_nan=False)+'\n');ledger.flush();completed=i+1
                stopping=args.stop_after is not None and completed-start>=args.stop_after
                if completed%config.checkpoint_interval==0 or completed==config.episodes or stopping:
                    if policy.tape:policy.finish(0.)
                    if source_manifest()!=manifest['source']:raise RuntimeError('Source changed during training; preserve this run and start a new experiment')
                    save_checkpoint(folder/'latest.pt',policy,config,seed,completed,scenes.manifest)
                if stopping:break
        summary=dict(modelSeed=seed,completedEpisodes=completed,trainingSeconds=time.perf_counter()-t,updates=policy.updates)
        if completed==config.episodes:
            random_policy=make_policy(config,scenes.device,seed)
            baseline=evaluate(random_policy,scenes,config,'test');write_json(folder/'untrained-test.json',baseline)
            summary['untrainedTest']=summarize(baseline)
            for split in ('validation','test'):
                rows=evaluate(policy,scenes,config,split);write_json(folder/f'{split}.json',rows);summary[split]=summarize(rows)
        summaries.append(summary);write_json(folder/'summary.json',summary)
        print(json.dumps(summary),flush=True)
    write_json(out/'summary.json',dict(seeds=summaries,aggregate=aggregate(summaries),
        note='All failures retained. Confidence intervals resample model seeds, not correlated episodes. Small runs are pipeline checks, not convergence evidence.'))


def evaluate_checkpoint(args):
    saved=read_checkpoint(args.checkpoint)
    config=ExperimentConfig(**{k:v for k,v in saved['config'].items() if k in {f.name for f in fields(ExperimentConfig)}})
    torch.set_num_threads(config.threads);seed_everything(saved['seed'],config.strict_determinism)
    scenes=Scenes(config);scenes.prepare()
    if saved['scenes']!=scenes.manifest:raise ValueError('Evaluation scene content changed')
    policy=make_policy(config,scenes.device,saved['seed']);policy.load(saved['policy'])
    path=Path(args.output)
    if path.exists():raise ValueError('Refuse to overwrite existing evidence')
    rows=evaluate(policy,scenes,config,args.split);write_json(path,dict(summary=summarize(rows),episodes=rows,checkpointSHA256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()))
    print(path.resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--config',required=True);t.add_argument('--output',required=True)
    t.add_argument('--resume',help='Resume one model seed into a NEW run directory at an episode boundary')
    t.add_argument('--stop-after',type=int,help='Cleanly stop after N additional episodes, saving a resumable checkpoint')
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint',required=True);e.add_argument('--split',choices=['validation','test'],default='test');e.add_argument('--output',required=True)
    args=parser.parse_args()
    if getattr(args,'stop_after',None) is not None and args.stop_after<1:parser.error('--stop-after must be positive')
    if args.command=='train':train(args)
    else:evaluate_checkpoint(args)
if __name__=='__main__':main()
