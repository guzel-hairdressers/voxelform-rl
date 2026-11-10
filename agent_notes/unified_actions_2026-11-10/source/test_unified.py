import functools,unittest
import numpy as np
import torch
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades
from voxelform.unified import UnifiedLearner,OPERATIONS,commit,valid_move

torch.set_num_threads(1)
def fixture(device='cpu'):
    c=PatchConfig(mode='combined',allocation='feedback',facade_layers=2,minimum_patch=4,maximum_patches=4)
    e=Environment((8,12,16),sun_count=4,device=device,far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=device))
    solid=e.allowed.clone();solid[4:]=False;e.restore_occupancy(solid);e.termination=None;e.unified_limit=12
    return e,UnifiedLearner(device,123,c)

class Unified(unittest.TestCase):
    def test_growth_is_available_first_while_above_target_far(self):
        e,p=fixture();self.assertGreater(e.far,e.target_far)
        self.assertEqual(p.legal_operations(e).tolist(),[True,True,True,False])
        move=p.choose(e,False,force_operation='grow');self.assertIsNotNone(move)
        self.assertTrue(move.add[4].any());self.assertFalse(move.remove.any());before=e.count;commit(e,move)
        self.assertGreater(e.count,before);self.assertEqual(e.edits,1)
        with self.assertRaises(ValueError):commit(e,move)
    def test_remove_and_exchange_available_without_a_phase(self):
        for operation in ['remove','exchange']:
            e,p=fixture();move=p.choose(e,False,force_operation=operation);self.assertIsNotNone(move)
            self.assertTrue(valid_move(e,move));before=e.count;commit(e,move)
            if operation=='exchange':self.assertEqual(e.count,before)
            else:self.assertLess(e.count,before)
    def test_far_acceptance_does_not_force_termination(self):
        e,p=fixture();solid=e.allowed.clone();solid[3:]=False;e.restore_occupancy(solid);e.termination=None
        self.assertEqual(p.legal_operations(e).tolist(),[True,True,True,True])
        move=p.choose(e,False,force_operation='exchange');self.assertIsNotNone(move);commit(e,move)
        self.assertFalse(e.terminal);self.assertTrue(e.far_satisfied)
        self.assertIsNone(p.choose(e,False,force_operation='finish'));self.assertEqual(p.last_operation,'finish')
    def test_policy_and_mask_heads_receive_gradients_and_checkpoint_roundtrip(self):
        e,p=fixture();torch.manual_seed(7);operations=[]
        for _ in range(10):
            move=p.choose(e,True);operations.append(p.last_operation)
            if move is not None:commit(e,move)
            p.record(.2+e.far_error*.01)
        result=p.finish(0.);self.assertGreater(result['gradientNorm'],0)
        self.assertGreater(float(p.model.operation_head.weight.grad.abs().sum()),0)
        self.assertTrue(bool((p.model.mask_head[-1].weight.grad.flatten(1).abs().sum(1)>0).all()))
        q=UnifiedLearner(config=p.config);q.load(p.state())
        for key,value in p.model.state_dict().items():torch.testing.assert_close(value,q.model.state_dict()[key])
        with self.assertRaises(ValueError):q.load({'format':'voxelform.patch.v1'})
        self.assertIn('grow',operations);self.assertIn('remove',operations)
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_native_mps_operation_and_mask_backward(self):
        e,p=fixture('mps');torch.manual_seed(13)
        move=p.choose(e,True);self.assertIsNotNone(move);commit(e,move);p.record(.2)
        self.assertGreater(p.finish(0.)['gradientNorm'],0)

if __name__=='__main__':unittest.main()
