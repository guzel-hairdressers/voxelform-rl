import unittest
import numpy as np
import torch
from tests.test_unified import fixture
from voxelform.unified import commit,valid_move,copy_plates,supported_removal,project
from voxelform.redistribution import Exchange

class FloorCopy(unittest.TestCase):
    def policy(self,device='cpu'):
        e,p=fixture(device);p.floor_copy=True;p.net_guided=True;p.amount_feedback=True
        return e,p
    def test_complete_plate_can_grow_on_first_edit_and_is_linked_to_source(self):
        e,p=self.policy();before=e.frontier.occupied.copy()
        move=p.choose(e,False,force_operation='grow');self.assertIsNotNone(move)
        np.testing.assert_array_equal(move.add[4],before[3]);self.assertFalse(move.add[:4].any())
        commit(e,move)
        self.assertTrue((e.copy_labels[3][before[3]]>0).all())
        np.testing.assert_array_equal(e.copy_labels[3],e.copy_labels[4])
        # A later top-only selection is pruned; selecting both linked cells survives.
        r=np.zeros(e.shape,bool);r[4,0,:4]=True
        self.assertFalse(supported_removal(e,r).any())
        r[3,0,:4]=True;np.testing.assert_array_equal(supported_removal(e,r),r)
    def test_courtyard_is_preserved_by_copy(self):
        e,p=self.policy();solid=e.frontier.occupied.copy();solid[:4,4:8,6:10]=False
        e.restore_occupancy(solid);e.termination=None
        move=p.choose(e,False,force_operation='grow');self.assertIsNotNone(move)
        np.testing.assert_array_equal(move.add[4],solid[3]);self.assertFalse(move.add[:,4:8,6:10].any())
        commit(e,move);self.assertFalse(e.frontier.occupied[:,4:8,6:10].any())
    def test_auto_exchange_grows_and_reduces_far_without_random_roof_bits(self):
        e,p=self.policy();torch.manual_seed(123)
        with torch.no_grad():p.model.operation_head.bias[2]=20
        before=e.far_error;move=p.choose(e,False);self.assertIsNotNone(move)
        self.assertTrue(move.remove.any() and move.add.any())
        np.testing.assert_array_equal(move.add[4],e.frontier.occupied[3]&~move.remove[3])
        self.assertFalse((move.remove&~e.frontier.eligible(-1)).any())
        commit(e,move);self.assertLess(e.far_error,before)
        self.assertFalse((e.frontier.occupied[1:]&~e.frontier.occupied[:-1]).any())
    def test_no_floating_upper_cells_or_fragmented_copy(self):
        e,p=self.policy();p.choose(e,False,force_operation='grow') # selects new semantics
        r=np.zeros(e.shape,bool);a=r.copy();r[0,0,:4]=True
        self.assertFalse(valid_move(e,Exchange(r,a,e.frontier.version)))
        r[:]=False;a[4,0,:4]=True
        self.assertFalse(valid_move(e,Exchange(r,a,e.frontier.version)))
        e.frontier.occupied[:]=True;e.frontier.refresh()
        self.assertEqual(copy_plates(e),[])
    def test_checkpoint_preserves_new_decoder_and_old_defaults(self):
        e,p=self.policy();q=type(p)();q.load(p.state());self.assertTrue(q.floor_copy and q.net_guided)
        old=p.state();del old['floorCopy'];q.load(old);self.assertFalse(q.floor_copy)
    def test_linked_cut_remains_subset_of_neural_selection(self):
        e,p=self.policy();move=p.choose(e,False,force_operation='grow');commit(e,move)
        masks=np.zeros((2,*e.shape),bool);masks[0,3:5,0,:8]=True
        result=project(e,masks,np.ones_like(masks,float),'remove',True,False)
        self.assertIsNotNone(result);self.assertFalse((result.remove&~masks[0]).any())
        np.testing.assert_array_equal(result.remove[3],result.remove[4]);commit(e,result)
    def test_copy_exchange_rebalances_the_extra_loss_from_cutting_its_source(self):
        e,p=self.policy();e.configure_far(4.5);e.termination=None;e.floor_copy_growth=True
        masks=np.zeros((2,*e.shape),bool);masks[0]=e.frontier.eligible(-1);masks[1]=copy_plates(e)[0]
        scores=np.broadcast_to(np.arange(e.nz)[None,:,None,None],masks.shape).astype(float)
        move=project(e,masks,scores,'exchange',True,True);self.assertIsNotNone(move)
        self.assertFalse((move.remove&~masks[0]).any())
        np.testing.assert_array_equal(move.add[4],e.frontier.occupied[3]&~move.remove[3])
        before=e.count;commit(e,move);self.assertGreater(e.count,before);self.assertTrue(e.far_satisfied)
    def test_small_edit_cap_never_fragments_a_floor_to_make_it_fit(self):
        from dataclasses import replace
        e,p=self.policy();p.config=replace(p.config,maximum_edit_voxels=16);e.frontier.config=p.config
        self.assertEqual(copy_plates(e),[])
        self.assertFalse(p.legal_operations(e)[2])
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_native_mps_copy_then_linked_backward(self):
        e,p=self.policy('mps');torch.manual_seed(123)
        # First on-policy exchange, followed by another pass through linked scores.
        with torch.no_grad():p.model.operation_head.bias[2]=20
        m=p.choose(e,True);self.assertIsNotNone(m);commit(e,m);p.record(.2)
        m=p.choose(e,True)
        if m:commit(e,m)
        p.record(.1);self.assertGreater(p.finish(0.)['gradientNorm'],0)

if __name__=='__main__':unittest.main()
