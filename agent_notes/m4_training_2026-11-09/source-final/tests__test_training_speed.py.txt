import functools,unittest
import numpy as np
import torch
from unittest.mock import patch
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades
from voxelform.fields import receivers

torch.set_num_threads(1)
class ExternalPreparation(unittest.TestCase):
    def check_device(self,device):
        c=PatchConfig(eligibility='depth')
        e=Environment((4,12,16),sun_count=4,device=device,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=device))
        first=e.precompute_external(chunk=73)
        self.assertGreater(first['newReceivers'],0);self.assertEqual(e.precompute_external()['newReceivers'],0)
        # Multiple roofs, recesses and an open well expose previously internal
        # faces. Compare cached fixed visibility against fresh actual ray queries.
        for seed in range(3):
            occupied=e.allowed.clone();occupied[:,3:6,4:7]=False;occupied[-1,7:10,7:13]=False
            occupied[-2:,1+seed,9:13]=False;e.restore_occupancy(occupied)
            ids,points,n,cells,faces=receivers(e.occupied)
            expected=e.static.query(points);cached,fresh=e.static_receivers.query(ids,points)
            self.assertEqual(fresh,0);self.assertTrue(torch.equal(cached,expected))
            quality=e.quality.detach().clone();light=e.light.clone();visible=e.visible.clone()
            e.static_receivers.known.zero_();e.evaluate()
            torch.testing.assert_close(e.quality,quality,rtol=0,atol=0)
            torch.testing.assert_close(e.light,light,rtol=0,atol=0)
            self.assertTrue(torch.equal(visible,e.visible));e.precompute_external()
    def test_cpu_exact_fields_after_geometry_changes(self):self.check_device('cpu')
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_gpu_exact_fields_after_geometry_changes(self):self.check_device('mps')
    def test_irregular_allowed_area_and_ground_undersides(self):
        e=Environment((3,8,10),plan='triangle',sun_count=4);r=e.precompute_external()
        volume=e.occupied.numel();known=e.static_receivers.known.reshape(6,*e.shape)
        self.assertFalse(bool(known[5,0].any()));self.assertFalse(bool((known&~e.allowed[None]).any()))
        self.assertEqual(r['knownReceivers'],6*e.capacity-int(e.allowed[0].sum()))
        for chunk in (0,1.5,True):
            with self.assertRaises(ValueError):e.precompute_external(chunk)
    def test_far_only_changes_reuse_current_fields_but_audit_stale_geometry(self):
        e=Environment((4,12,16),sun_count=4);light=e.light.clone();depth=e.depth.clone()
        with patch.object(e,'evaluate',wraps=e.evaluate) as evaluate:
            e.configure_far(2.5);evaluate.assert_not_called()
        torch.testing.assert_close(e.light,light,rtol=0,atol=0);torch.testing.assert_close(e.depth,depth,rtol=0,atol=0)
        self.assertEqual(e.depth_edit,e.edits);self.assertEqual(e.last_audit_edit,e.edits)
        e.last_audit_edit=-1
        with patch.object(e,'evaluate',wraps=e.evaluate) as evaluate:
            e.configure_far(3.);evaluate.assert_called_once()

class TrainingRewards(unittest.TestCase):
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_informed_facade_actor_runs_backward_on_gpu(self):
        from voxelform.training import EnvironmentLearner
        c=PatchConfig(allocation='feedback',batched=True)
        e=Environment((4,12,16),device='mps',sun_count=4,frontier_factory=functools.partial(PatchFacades,config=c,depth_device='mps'))
        p=EnvironmentLearner('mps',config=c);a=p.choose(e,True);self.assertIsNotNone(a)
        e.apply(a,audit=False);p.record(.3);self.assertGreater(p.finish(p.bootstrap(e))['gradientNorm'],0)
    def test_informed_inputs_preserve_initial_weights_and_train_extra_channels(self):
        import copy
        from voxelform.training import EnvironmentLearner
        from voxelform.patch_policy import PatchLearner
        c=PatchConfig(allocation='feedback',batched=True)
        e=Environment((4,12,16),sun_count=4,frontier_factory=functools.partial(PatchFacades,config=c))
        e.precompute_external();p=EnvironmentLearner(config=c);base=PatchLearner(config=c)
        observation=p.informed_observation(e);self.assertEqual(observation.shape[1],9)
        torch.testing.assert_close(p.fields(e)[0],base.fields(e)[0],rtol=1e-5,atol=1e-6)
        a=p.choose(e,True);self.assertIsNotNone(a);e.apply(a,audit=False);p.record(.3);p.finish(p.bootstrap(e))
        self.assertGreater(float(p.model.encoder[0].weight[:,5:].detach().abs().sum()),0)
        q=EnvironmentLearner();q.load(copy.deepcopy(p.state()))
        for k,v in p.model.state_dict().items():torch.testing.assert_close(q.model.state_dict()[k],v,rtol=0,atol=0)
        self.assertIsNotNone(q.choose(e,False))
    def test_no_air_padding_can_hide_external_exposure_from_solid_voxel_inputs(self):
        from voxelform.policy import Learner
        e=Environment((4,12,16),sun_count=4);e.noise.zero_()  # Isolate the light channel in a fully solid grid.
        e.sky_boundary.fill_(1);e.static_receivers.known.fill_(True);e.static_receivers.values.fill_(True);e.evaluate()
        observation=e.observation().clone();quality=float(e.quality)
        p=Learner();before=p.fields(e)[0].detach().clone()
        e.sky_boundary.zero_();e.static_receivers.values.zero_();e.evaluate()
        self.assertNotEqual(float(e.quality),quality)
        torch.testing.assert_close(e.observation(),observation,rtol=0,atol=0)
        torch.testing.assert_close(p.fields(e)[0],before,rtol=0,atol=0)
    def test_quality_mode_has_continuous_failure_penalty(self):
        from voxelform.training import step_reward,audit_reward,easy_probability
        c=PatchConfig(allocation='feedback');e=Environment((4,12,16),sun_count=4,frontier_factory=functools.partial(PatchFacades,config=c))
        e.count=e.target_count+1;e.termination='step_budget_exhausted'
        self.assertGreater(step_reward(e,e.far_error,'quality'),-.1)
        self.assertLess(step_reward(e,e.far_error,'current'),-1)
        self.assertAlmostEqual(audit_reward(.2,.3,'quality'),.5)
        self.assertAlmostEqual(audit_reward(.2,.3,'current'),.01)
        self.assertAlmostEqual(easy_probability('curriculum',0),.8);self.assertAlmostEqual(easy_probability('curriculum',1),.2)
        self.assertAlmostEqual(easy_probability('mixed',.3),.5)

if __name__=='__main__':unittest.main()
