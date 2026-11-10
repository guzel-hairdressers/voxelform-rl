import functools,unittest
import torch
import numpy as np
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner
from voxelform.redistribution import Exchange,validate,commit

torch.set_num_threads(1)
class FarBand(unittest.TestCase):
    def make(self):
        c=PatchConfig(allocation='feedback',batched=True)
        return Environment((6,10,10),sun_count=4,far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c)),PatchLearner(config=c)
    def test_bounds_and_workload_use_nearest_acceptable_count(self):
        e,p=self.make();self.assertEqual(e.acceptable_counts,(290,310));self.assertEqual(e.goal_count,310)
        self.assertEqual(e.workload.required,290);self.assertAlmostEqual(e.far_error,2.9)
        for count in (290,300,310):
            e.count=count;self.assertTrue(e.far_satisfied);self.assertEqual(e.far_error,0)
        for count in (289,311):e.count=count;self.assertFalse(e.far_satisfied)
    def test_success_after_one_cell_without_chasing_centre(self):
        e,p=self.make();solid=e.allowed.clone();solid.flatten()[311:]=False;e.restore_occupancy(solid)
        action=p.choose(e,False);self.assertEqual(action.count,1);e.apply(action,audit=False)
        self.assertEqual(e.count,310);self.assertEqual(e.termination,'far_reached')
        self.assertEqual(e.metrics()['remainingVoxels'],0)
    def test_restore_inside_band_is_immediately_successful(self):
        e,p=self.make();solid=e.allowed.clone();solid[3:]=False;e.restore_occupancy(solid)
        self.assertTrue(e.terminal);self.assertEqual(e.workload.limit,0)
    def test_unequal_exchange_allowed_only_within_band(self):
        e,p=self.make();solid=e.allowed.clone();solid[3:]=False;e.restore_occupancy(solid)
        r=np.zeros(e.shape,bool);a=np.zeros(e.shape,bool);r[0:2,0,0]=True;a[3,3:5,3:5]=True
        move=Exchange(r,a,e.frontier.version);self.assertTrue(validate(e,move));commit(e,move)
        self.assertEqual(e.count,302);self.assertTrue(e.far_satisfied)
        e.configure_far(3.,0.);self.assertFalse(validate(e,Exchange(r,a,e.frontier.version)))
    def test_validation_and_exact_mode(self):
        e,p=self.make()
        for value in (-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):e.configure_far(3.,value)
        e.configure_far(3.,0);self.assertEqual(e.acceptable_counts,(300,300))
        e.configure_far(5.95,.1);self.assertEqual(e.acceptable_counts[1],600)
    def test_policy_configuration_preserves_success_inside_band(self):
        from voxelform.session import Session
        e,p=self.make();solid=e.allowed.clone();solid.flatten()[305:]=False;e.restore_occupancy(solid)
        s=Session(env=e,autostart=False);s.configure_masks('combined',True,allocation='feedback')
        self.assertEqual(e.termination,'far_reached');self.assertEqual(e.workload.required,0)
        self.assertEqual(s.state()['settings']['farTolerance'],.1)

if __name__=='__main__':unittest.main()
