import functools,unittest
import numpy as np
import torch
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades,PatchLearner
from voxelform.redistribution import Exchange,validate,commit,duplicate_top,choose,project_pair

torch.set_num_threads(1)
def make(layers=1,device='cpu'):
    c=PatchConfig(facade_layers=layers,maximum_patches=4)
    e=Environment((8,16,20),sun_count=4,device=device,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=device))
    return e,PatchLearner(device,config=c)

class Redistribution(unittest.TestCase):
    def test_layers_are_horizontal_frozen_and_do_not_include_roofs(self):
        for n in (1,2,3):
            e,p=make(n);mask=e.frontier.eligible(-1)
            self.assertTrue(mask[:,0:n,8].all());self.assertFalse(mask[:,n,8].any())
            self.assertFalse(mask[-1,7,8]);self.assertEqual(float(e.frontier.weights_tensor(-1,'cpu')[0,n-1,8]),1.)
        for n in (0,4,True):
            with self.assertRaises(ValueError):PatchConfig(facade_layers=n)
    def test_pair_can_split_mass_and_add_height_at_identical_far(self):
        e,p=make(3);solid=np.zeros(e.shape,bool);solid[:4,2:14,2:18]=True;e.restore_occupancy(solid)
        remove=np.zeros(e.shape,bool);remove[:4,2:14,9:11]=True
        # Interior cut isn't within the frozen 3-layer facade band everywhere;
        # use a narrow bridge between two initially connected wings instead.
        solid[:,:,9:11]=False;solid[:2,2:4,9:11]=True;e.restore_occupancy(solid)
        remove=np.zeros(e.shape,bool);remove[:2,2:4,9:11]=True
        add=np.zeros(e.shape,bool);add[4,2:4,2:6]=True
        move=Exchange(remove,add,e.frontier.version);before=e.count
        self.assertTrue(validate(e,move));self.assertFalse(validate(e,move,maximum_buildings=1));commit(e,move)
        self.assertEqual(e.count,before);self.assertEqual(int(e.occupied.sum()),before);self.assertEqual(e.frontier.components,2)
        with self.assertRaises(ValueError):commit(e,move)
    def test_top_storey_duplication_uses_only_next_layer(self):
        e,p=make();solid=np.zeros(e.shape,bool);solid[:3,2:8,2:8]=True;e.restore_occupancy(solid)
        add=duplicate_top(e);self.assertEqual(int(add.sum()),36);self.assertTrue(add[3,2:8,2:8].all());self.assertFalse(add[4:].any())
    def test_projection_only_discards_selected_bits_and_rebalances(self):
        e,p=make(3);solid=e.allowed.cpu().numpy().copy();solid[3:]=False;e.restore_occupancy(solid)
        masks=np.zeros((2,*e.shape),bool);masks[0,:3,0:2,2:10]=True;masks[1,3,2:6,2:6]=True
        scores=np.broadcast_to(np.arange(np.prod(e.shape)).reshape(e.shape),(2,*e.shape)).copy()
        move=project_pair(e,masks,scores);self.assertIsNotNone(move)
        self.assertFalse((move.remove&~masks[0]).any());self.assertFalse((move.add&~masks[1]).any())
        self.assertEqual(move.remove.sum(),move.add.sum())
    def test_actual_network_generates_balanced_pair_and_both_heads_train(self):
        e,p=make(3);solid=e.allowed.cpu().numpy().copy();solid[3:]=False;e.restore_occupancy(solid)
        torch.manual_seed(123);move=choose(p,e,24,True);self.assertIsNotNone(move)
        commit(e,move);p.record(.4);p.finish(0.)
        gradient=p.model.mask_head[-1].weight.grad
        self.assertTrue(bool((gradient.flatten(1).abs().sum(1)>0).all()))
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_gpu_exchange_backward(self):
        e,p=make(2,'mps');solid=e.allowed.clone();solid[3:]=False;e.restore_occupancy(solid)
        move=choose(p,e,24,True);self.assertIsNotNone(move);commit(e,move);p.record(.2);self.assertGreater(p.finish(0.)['gradientNorm'],0)

if __name__=='__main__':unittest.main()
