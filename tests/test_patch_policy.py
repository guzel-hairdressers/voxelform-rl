import copy,functools,unittest
import numpy as np
import torch
from voxelform.environment import Environment
from voxelform.actions import exterior_air
from voxelform.policy import Learner
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner,gaussian_field,components

torch.set_num_threads(1)
def env_for(c,shape=(4,16,20)):
    return Environment(shape,sun_count=4,frontier_factory=functools.partial(PatchFacades,config=c))
class CoherentMasks(unittest.TestCase):
    def test_gaussian_preserves_constant_and_supports_gradients(self):
        x=torch.ones((4,12,16),requires_grad=True);y=gaussian_field(x,(2,2,2))
        torch.testing.assert_close(y,x);y.square().mean().backward();self.assertTrue(torch.isfinite(x.grad).all())
    def test_depth_union_exact_values_and_protects_shallow_interior(self):
        c=PatchConfig(eligibility='depth');f=PatchFacades(np.ones((4,41,41),bool),np.ones((4,41,41),bool),config=c)
        self.assertEqual(f.remove_weights[1,20,0],1);self.assertEqual(f.remove_weights[1,20,3],0)
        self.assertAlmostEqual(float(f.remove_weights[1,20,12]),(12.5-8)/8);self.assertEqual(f.remove_weights[1,20,20],1)
    def test_open_atrium_is_legal_and_sealed_pockets_are_rejected(self):
        c=PatchConfig(eligibility='depth');f=PatchFacades(np.ones((4,41,41),bool),np.ones((4,41,41),bool),config=c)
        closed=np.zeros(f.shape,bool);closed[1:3,18:23,18:23]=1
        self.assertFalse(f.validate_mask(closed,-1));self.assertIsNone(f.project_mask(closed,np.ones(f.shape),-1,1000))
        opened=np.zeros(f.shape,bool);opened[:,18:23,18:23]=1
        self.assertTrue(f.validate_mask(opened,-1));a=f.project_mask(opened,np.ones(f.shape),-1,1000)
        self.assertIsNotNone(a);self.assertFalse(a.facade_only);f.apply(a);self.assertTrue(f.exterior_air[:,20,20].all())
    def test_coarse_neural_field_can_generate_an_open_interior_well(self):
        from torch import nn
        class Oracle(nn.Module):
            def __init__(self):super().__init__();self.patch_log_std=nn.Parameter(torch.zeros(2))
            def forward(self,observation,goal):
                out=torch.full((2,4,41,41),-9.);out[0,:,14:27,14:27]=9.
                return out,torch.tensor(0.)
        c=PatchConfig(mode='combined',eligibility='depth');e=env_for(c,(4,41,41));p=PatchLearner(config=c);p.model=Oracle()
        a=p.choose(e,False,True);self.assertIsNotNone(a);self.assertGreater(int((a.mask&~e.frontier.remove_skin).sum()),0)
        e.apply(a,audit=False);self.assertTrue(e.frontier.exterior_air[a.mask].all())
    def test_interior_opening_count_and_minimum_footprint_are_enforced(self):
        c=PatchConfig(eligibility='depth',maximum_voids=1,minimum_void_area=4)
        o=np.ones((4,51,51),bool);f=PatchFacades(o,o,config=c)
        two=np.zeros_like(o);two[:,24:27,16:19]=1;two[:,24:27,31:34]=1
        self.assertFalse(f.validate_mask(two,-1));a=f.project_mask(two,np.ones(f.shape),-1,1000)
        self.assertIsNotNone(a);self.assertEqual(a.count,36);f.apply(a)
        self.assertTrue(all(len(ids)<=1 for _,ids,_ in f.old_voids))
        tiny=np.zeros_like(o);tiny[:,25,25]=1
        fresh=PatchFacades(o,o,config=c);self.assertIsNone(fresh.project_mask(tiny,np.ones(f.shape),-1,1000))
    def test_growth_cannot_split_an_existing_courtyard_into_two(self):
        c=PatchConfig(eligibility='depth',maximum_voids=1,minimum_void_area=4)
        o=np.ones((4,41,41),bool);o[:,17:24,17:24]=0;f=PatchFacades(o,np.ones_like(o),config=c)
        strip=np.zeros_like(o);strip[:,17:24,20]=1
        # Only facade-adjacent growth is allowed; here ends are adjacent,
        # so check the void constraint directly on a complete future strip.
        self.assertFalse(f._void_filter(strip,1,check_only=True))
    def test_courtyard_can_merge_with_outside_by_later_facade_steps(self):
        c=PatchConfig(eligibility='depth',maximum_voids=1,minimum_void_area=4)
        o=np.ones((4,41,41),bool);o[:,17:24,17:24]=0;f=PatchFacades(o,np.ones_like(o),config=c)
        self.assertTrue(all(len(ids)==1 for _,ids,_ in f.old_voids))
        for x in range(17):
            cut=np.zeros_like(o);cut[:,17:24,x]=1
            self.assertTrue(f.eligible(-1)[cut].all())
            a=f.project_mask(cut,np.ones(f.shape),-1,1000);self.assertIsNotNone(a);f.apply(a)
        self.assertTrue(all(len(ids)==0 for _,ids,_ in f.old_voids))
        self.assertTrue(f.exterior_air[:,20,:24].all())
    def test_addition_cannot_shrink_a_valid_opening_below_minimum_area(self):
        cfg=PatchConfig(eligibility='depth',maximum_voids=1,minimum_void_area=4)
        o=np.ones((4,41,41),bool);o[:,20:22,20:22]=0;f=PatchFacades(o,np.ones_like(o),config=cfg)
        column=np.zeros_like(o);column[:,20,20]=1
        self.assertFalse(f.validate_mask(column,1));self.assertIsNone(f.project_mask(column,np.ones(f.shape),1,100))
        fill=np.zeros_like(o);fill[:,20:22,20:22]=1
        self.assertTrue(f.validate_mask(fill,1))
    def test_patch_limits_only_prune_generated_bits(self):
        c=PatchConfig(minimum_patch=4,maximum_patches=1);f=PatchFacades(np.ones((4,16,20),bool),np.ones((4,16,20),bool),config=c)
        raw=np.zeros(f.shape,bool);raw[:,2:6,0]=1;raw[0,12,0]=1;raw[0,0,18]=1
        a=f.project_mask(raw,np.ones(f.shape),-1,100)
        self.assertIsNotNone(a);self.assertFalse((a.mask&~raw).any());self.assertEqual(components(a.mask)['components'],1);self.assertEqual(a.count,16)
    def test_gaussian_latent_score_has_correct_mean_gradient(self):
        p=PatchLearner(config=PatchConfig(mode='combined'));p._operation=-1
        field=torch.full((4,12,16),-3.,requires_grad=True);weights=torch.ones_like(field)
        torch.manual_seed(7);_,logp,_,_=p.draw(field,weights,100,True,False);logp.backward()
        self.assertGreater(float(field.grad.abs().sum()),0);self.assertGreater(float(p.model.patch_log_std.grad.abs().sum()),0)
        # Dist.sample must be detached: differentiating log_prob(rsample())
        # would cancel this score-function mean gradient.
    def test_all_modes_backprop_and_inference_does_not_change_weights(self):
        for mode in ('independent','smooth','correlated','combined'):
            c=PatchConfig(mode=mode);e=env_for(c);p=PatchLearner(config=c)
            a=p.choose(e,True);self.assertIsNotNone(a,mode);e.apply(a,audit=False);p.record(.2)
            stats=p.finish(p.bootstrap(e));self.assertGreater(stats['gradientNorm'],0)
            old=copy.deepcopy(p.model.state_dict());a=p.choose(e,False,True)
            for k,v in old.items():torch.testing.assert_close(p.model.state_dict()[k],v,rtol=0,atol=0)
    def test_checkpoint_roundtrip_and_old_policy_migration_are_transactional(self):
        c=PatchConfig(mode='combined',eligibility='depth');p=PatchLearner(config=c);old=Learner();p.load(old.state())
        self.assertEqual(p.config,c);self.assertEqual(p.direct_updates,0)
        for k,v in old.model.state_dict().items():torch.testing.assert_close(p.model.state_dict()[k],v,rtol=0,atol=0)
        saved=copy.deepcopy(p.state());q=PatchLearner();q.load(saved);self.assertEqual(q.config,c)
        saved['model']['encoder.0.weight']=torch.ones(2)
        with self.assertRaises(RuntimeError):q.load(saved)
        self.assertEqual(q.config,c)
    def test_large_patches_have_less_fragmentation_than_independent_draws(self):
        weights=torch.zeros((4,32,32));weights[:,:,0]=1;weights[:,:,-1]=1;weights[:,0]=1;weights[:,-1]=1
        metrics={}
        for mode in ('independent','combined'):
            p=PatchLearner(config=PatchConfig(mode=mode));p._operation=-1;counts=[]
            for seed in range(12):
                torch.manual_seed(seed);a,*_=p.draw(torch.full_like(weights,-3),weights,1000,False,False);counts.append(components(a.numpy())['components'])
            metrics[mode]=np.mean(counts)
        self.assertLess(metrics['combined'],metrics['independent']/2)
    def test_session_mode_and_checkpoint_switch_preserve_mass(self):
        from voxelform.session import Session
        e=Environment((4,16,20),sun_count=4);s=Session(env=e,autostart=False);old=e.occupied.clone()
        s.configure_masks('combined',True);self.assertIsInstance(s.policy,PatchLearner)
        torch.testing.assert_close(e.occupied,old);self.assertTrue(s.state()['settings']['deepInterior'])
        same=s.policy;s.configure_masks('combined',True);self.assertIs(s.policy,same)
        s.start(training=False,single=True);s.advance_one();self.assertEqual(e.edits,1)
        self.assertEqual(s.edit_trace[-1]['maskSource'],'neural_combined')
        mass=e.occupied.clone();saved=PatchLearner(config=PatchConfig(mode='correlated')).state();s.load_policy(saved)
        self.assertEqual(s.env.frontier.config,s.policy.config);torch.testing.assert_close(e.occupied,mass)
    def test_autosave_restores_weights_mass_settings_and_optimizer(self):
        import tempfile,json
        from pathlib import Path
        from voxelform.session import Session
        class Library:
            def catalog(self):return [{'site_id':'unused'}]
        with tempfile.TemporaryDirectory() as directory:
            e=Environment((4,16,20),sun_count=4);s=Session(env=e,autostart=False);s.persistence_root=Path(directory)
            s.configure_masks('combined',True);s.start(training=True,single=True);s.advance_one();s.learn_rollout();s.pause()
            self.assertIsNone(s.persistence_error)
            manifest=json.loads((Path(directory)/'latest.json').read_text());self.assertTrue((Path(directory)/manifest['policy']).is_file())
            q=Session(library=Library(),autostart=False,device='cpu');q.persistence_root=Path(directory);q.build()
            torch.testing.assert_close(q.env.occupied,s.env.occupied);self.assertEqual(q.policy.config,s.policy.config)
            self.assertEqual(q.policy.updates,s.policy.updates);self.assertEqual(q.env.edits,s.env.edits)
            for k,v in s.policy.model.state_dict().items():torch.testing.assert_close(q.policy.model.state_dict()[k],v,rtol=0,atol=0)
            self.assertTrue(q.policy.optimizer.state_dict()['state'])
    def test_batched_gaussian_matches_individual_gaussian(self):
        x=torch.randn(3,4,12,16);a=gaussian_field(x,(2,2,2));b=torch.stack([gaussian_field(v,(2,2,2)) for v in x])
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    def test_batched_latents_use_consumed_prefix_and_preserve_gradient(self):
        from unittest.mock import patch
        from torch.distributions import Normal
        def fixed(distribution,sample_shape=torch.Size()):
            base=distribution.loc.detach()+torch.sin(torch.arange(distribution.loc.numel()).reshape(distribution.loc.shape))*.3
            return base.expand(*sample_shape,*base.shape).clone()
        cfg=PatchConfig(mode='combined',batched=True,maximum_edit_voxels=32)
        e=env_for(cfg);p=PatchLearner(config=cfg)
        with patch.object(Normal,'sample',fixed):a=p.choose(e,True)
        self.assertIsNotNone(a);self.assertLessEqual(a.count,32);one=p.tape[-1][0].detach().clone()
        p.clear();original=e.frontier.project_mask;calls=[0]
        def delayed(*args):
            calls[0]+=1
            if calls[0]<3:return None
            return original(*args)
        with patch.object(Normal,'sample',fixed),patch.object(e.frontier,'project_mask',delayed):a=p.choose(e,True)
        self.assertIsNotNone(a);self.assertEqual(p.generation_attempts,3);torch.testing.assert_close(p.tape[-1][0],3*one)
        e.apply(a,audit=False);p.record(.2);stats=p.finish(p.bootstrap(e));self.assertGreater(stats['gradientNorm'],0)
    def test_batch_and_sequential_decoder_agree_for_identical_latents(self):
        from unittest.mock import patch
        from torch.distributions import Normal
        def fixed(distribution,sample_shape=torch.Size()):
            base=distribution.loc.detach()+torch.sin(torch.arange(distribution.loc.numel()).reshape(distribution.loc.shape))*.3
            return base.expand(*sample_shape,*base.shape).clone()
        a_cfg=PatchConfig(mode='combined',maximum_edit_voxels=32);b_cfg=PatchConfig(mode='combined',maximum_edit_voxels=32,batched=True)
        a_e,b_e=env_for(a_cfg),env_for(b_cfg);a_p,b_p=PatchLearner(config=a_cfg),PatchLearner(config=b_cfg)
        with patch.object(Normal,'sample',fixed):a=a_p.choose(a_e,True);b=b_p.choose(b_e,True)
        self.assertIsNotNone(a);self.assertIsNotNone(b);np.testing.assert_array_equal(a.mask,b.mask)
        torch.testing.assert_close(a_p.tape[-1][0],b_p.tape[-1][0],rtol=1e-5,atol=1e-6)
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_actual_mps_optimizer_can_resume_from_checkpoint(self):
        import io
        cfg=PatchConfig(batched=True,maximum_edit_voxels=16,minimum_patch=4)
        e=Environment((4,8,10),device='mps',sun_count=4,frontier_factory=functools.partial(PatchFacades,config=cfg,depth_device='mps'))
        p=PatchLearner('mps',config=cfg);a=p.choose(e,True);self.assertIsNotNone(a);e.apply(a,audit=False);p.record(.2);p.finish(p.bootstrap(e))
        buffer=io.BytesIO();torch.save(p.state(),buffer);buffer.seek(0)
        q=PatchLearner('mps');q.load(torch.load(buffer,map_location='mps',weights_only=True))
        a=q.choose(e,True);self.assertIsNotNone(a);e.apply(a,audit=False);q.record(.2);stats=q.finish(q.bootstrap(e));self.assertEqual(stats['updates'],2)
    def test_frozen_mask_and_far_residual_are_preserved(self):
        c=PatchConfig(mode='combined');e=env_for(c);e.configure_far((e.count-1)/e.plot_area);p=PatchLearner(config=c)
        a=p.choose(e,False,True);self.assertIsNotNone(a);self.assertEqual(a.count,1);e.apply(a,audit=False);self.assertTrue(e.terminal)
        with self.assertRaises(ValueError):e.frontier.apply(a)
if __name__=='__main__':unittest.main()
