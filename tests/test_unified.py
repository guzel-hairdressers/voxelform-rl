import functools,unittest
import numpy as np
import torch
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades
from voxelform.redistribution import Exchange
from voxelform.shadows import FastVoxelShadows,dda_paths
from voxelform.fields import receivers
from voxelform.context import prepare_scene
from tests.test_context import fixture as ctx_fixture
from voxelform.unified import UnifiedLearner,Organic3DNetwork,OPERATIONS,clean_floor_tensor,clean_floor_numpy,long_narrow_mask,commit,valid_move,project,project_organic_3d

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
    def test_safe_pruning_can_split_grounded_wings_without_inventing_edits(self):
        e,p=fixture();solid=np.zeros(e.shape,bool)
        solid[:4,1:11,1:7]=True;solid[:4,1:11,9:15]=True;solid[:2,2:4,7:9]=True;e.restore_occupancy(solid)
        masks=np.zeros((2,*e.shape),bool);masks[0,:2,2:4,7:9]=True
        scores=np.ones_like(masks,dtype=float)
        move=project(e,masks,scores,'remove',repair_remove=True);self.assertIsNotNone(move)
        self.assertFalse((move.remove&~masks[0]).any());self.assertFalse(move.add.any())
        self.assertEqual(e.frontier.components,1);commit(e,move);self.assertEqual(e.frontier.components,2)
    def test_net_area_guidance_still_grows_roof_on_the_first_above_far_edit(self):
        e,p=fixture();p.net_guided=True;p.amount_feedback=True
        self.assertEqual(p.legal_operations(e).tolist(),[True,False,True,False])
        move=p.choose(e,False,force_operation='exchange');self.assertIsNotNone(move)
        self.assertTrue(move.add[4].any());self.assertGreaterEqual(int(move.remove.sum()),int(move.add.sum()))
        before=e.far_error;commit(e,move);self.assertLessEqual(e.far_error,before)
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
        e,p=fixture('mps');p.net_guided=True;p.amount_feedback=True
        with torch.no_grad():p.model.operation_head.bias[2]=10.
        torch.manual_seed(13)
        move=p.choose(e,True);self.assertIsNotNone(move);commit(e,move);p.record(.2)
        self.assertGreater(p.finish(0.)['gradientNorm'],0)

    def test_long_narrow_mask_and_sliver_detection(self):
        occ=torch.zeros((1,20,24),dtype=torch.bool)
        occ[0,4:16,2:14]=True  # 12x12 room
        self.assertEqual(int(long_narrow_mask(occ,width=6,stub=2).sum()),0)
        occ2=occ.clone();occ2[0,8:12,14:24]=True  # 4x10 narrow strip attached to room
        ln=long_narrow_mask(occ2,width=6,stub=2)
        self.assertFalse(bool(ln[0,4:16,2:14].any()))
        self.assertFalse(bool(ln[0,8:12,14:16].any()))  # 2m stub exempted
        self.assertEqual(int(ln[0,8:12,16:24].sum()),32)
        strip2=torch.zeros((1,12,12),dtype=torch.bool);strip2[0,2:4,2:10]=True
        self.assertFalse(bool(clean_floor_tensor(strip2,min_width=3).any()))
        rng=np.random.default_rng(42)
        rand_occ=rng.random((4,20,24))>0.25
        for mw in (2,3,4,6):
            np.testing.assert_array_equal(clean_floor_numpy(rand_occ,min_width=mw),clean_floor_tensor(torch.from_numpy(rand_occ),min_width=mw).numpy())

    def test_organic_3d_network_shapes_and_gradients(self):
        import math
        net=Organic3DNetwork(12)
        goal=torch.zeros(5)
        for Z,H,W in ((6,16,18),(5,13,17)):
            obs=torch.randn((Z,12,H,W))
            coarse,cval,cop=net.forward_coarse(obs,goal)
            self.assertEqual(tuple(coarse.shape),(2,math.ceil(Z/2),math.ceil(H/4),math.ceil(W/4)))
            self.assertEqual(cval.ndim,0)
            self.assertEqual(tuple(cop.shape),(4,))
            fields,val,op=net(obs,goal)
            self.assertEqual(tuple(fields.shape),(2,Z,H,W))
            self.assertEqual(val.ndim,0)
            self.assertEqual(tuple(op.shape),(4,))
        loss=fields.square().mean()+val.square()+op.square().mean()
        loss.backward()
        for name,param in net.named_parameters():
            if name=='patch_log_std':continue
            self.assertIsNotNone(param.grad,name)
            self.assertTrue(bool(torch.isfinite(param.grad).all()),name)

    def test_project_organic_3d_always_passes_valid_move(self):
        c=PatchConfig(mode='combined',allocation='feedback',eligibility='depth',facade_layers=2,minimum_patch=4,maximum_patches=4)
        e=Environment((6,20,22),sun_count=4,device='cpu',far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device='cpu'))
        solid=e.allowed.clone();solid[4:]=False
        e.restore_occupancy(solid);e.organic_3d_growth=True
        rng=np.random.default_rng(7)
        from voxelform.redistribution import growth_mask
        w_rem=e.frontier.eligible(-1);w_add=growth_mask(e)
        checked=0
        for op in ('remove','grow','exchange'):
            for _ in range(3):
                masks=np.zeros((2,*e.shape),bool)
                if op in ('remove','exchange'):masks[0]=w_rem&(rng.random(e.shape)<.35)
                if op in ('grow','exchange'):masks[1]=w_add&(rng.random(e.shape)<.35)
                scores=rng.random(masks.shape)
                mv=project_organic_3d(e,masks,scores,op,True)
                if mv is not None:
                    self.assertTrue(valid_move(e,mv))
                    checked+=1
        self.assertGreater(checked,0)

    def test_dense_spatial_rl_and_checkpoint_arch(self):
        devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
        c=PatchConfig(mode='combined',allocation='feedback',eligibility='depth',facade_layers=2,minimum_patch=4,maximum_patches=4)
        for dev in devices:
            e=Environment((6,16,18),sun_count=4,device=dev,far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device=dev))
            e.precompute_external();e.organic_3d_growth=True
            p=UnifiedLearner(dev,123,c,amount_feedback=True,net_guided=True,organic_3d=True)
            self.assertEqual(p.observation(e).shape[1],14)
            for _ in range(6):
                e.reset();solid_ep=e.allowed.clone();solid_ep[4:]=False;e.restore_occupancy(solid_ep);e.termination=None;e.unified_actions=0
                for _ in range(3):
                    mv=p.choose(e,True)
                    if mv is not None:commit(e,mv,validated=True)
                    p.record(.1)
                    if e.terminal:break
                p.mark_done()
            res=p.finish(0.)
            self.assertGreater(res['gradientNorm'],0)
            self.assertIsNotNone(p.model.patch_log_std.grad)
            self.assertGreater(float(p.model.patch_log_std.grad.abs().sum()),0)
            saved=p.state()
            self.assertEqual(saved.get('organic3dArch'),'coarse_2d1d_v4')
            self.assertEqual(saved.get('organic3dChannels'),14)
            q=UnifiedLearner(dev,config=c);q.load(saved)
            bad=dict(saved);bad['organic3dArch']='legacy_v1'
            with self.assertRaises(ValueError):q.load(bad)

    def test_organic_3d_permits_cantilevers_rejects_slivers_and_trains_across_episodes(self):
        c=PatchConfig(mode='combined',allocation='feedback',facade_layers=2,minimum_patch=4,maximum_patches=4)
        e=Environment((8,20,22),sun_count=4,device='cpu',far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device='cpu'))
        solid=torch.zeros(e.shape,dtype=torch.bool)
        solid[:5,4:16,4:12]=True
        solid[5,4:12,4:12]=True
        e.restore_occupancy(solid);e.termination=None;e.unified_limit=12;e.organic_3d_growth=True
        # Cantilever growth at z=5 (x=12) over empty z=4 must be accepted
        a=np.zeros(e.shape,bool);a[5,4:10,12]=True
        move=Exchange(np.zeros(e.shape,bool),a,e.frontier.version)
        self.assertTrue(valid_move(e,move))
        changed=(e.frontier.occupied&~move.remove)|move.add
        self.assertTrue(bool((changed[5]&~changed[4]).any()))
        # Leaving a 2-voxel outer strip at y=4:6 by cutting y=6:8 must be rejected
        r_bad=np.zeros(e.shape,bool);r_bad[:5,6:8,4:12]=True
        self.assertFalse(valid_move(e,Exchange(r_bad,np.zeros(e.shape,bool),e.frontier.version)))
        # Learner runs multi-episode GAE with mark_done() and checkpoint roundtrip
        p=UnifiedLearner('cpu',123,c,amount_feedback=True,net_guided=True,organic_3d=True)
        for ep in range(2):
            e.reset();solid_ep=e.allowed.clone();solid_ep[4:]=False;e.restore_occupancy(solid_ep);e.termination=None;e.unified_actions=0
            for _ in range(4):
                mv=p.choose(e,True)
                if mv is not None:commit(e,mv,validated=True)
                p.record(.1)
                if e.terminal:break
            p.mark_done()
        res=p.finish(0.)
        self.assertGreater(res['gradientNorm'],0)
        self.assertIsNotNone(p.model.patch_log_std.grad)
        q=UnifiedLearner('cpu',config=c)
        q.load(p.state())
        self.assertTrue(q.organic_3d and not q.floor_copy)
        for k,v in p.model.state_dict().items():torch.testing.assert_close(v,q.model.state_dict()[k])

    def test_forward_coarse_batch_and_tiled_vertical_window(self):
        torch.manual_seed(17)
        net=Organic3DNetwork(12)
        obs_b=torch.randn((3,18,12,16,20))
        goal_b=torch.randn((3,5))
        c_full,v_full,o_full=net.forward_coarse_batch(obs_b,goal_b,window_floors=64)
        c_tile,v_tile,o_tile=net.forward_coarse_batch(obs_b,goal_b,window_floors=8)
        torch.testing.assert_close(c_tile,c_full,atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(v_tile,v_full,atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(o_tile,o_full,atol=1e-4,rtol=1e-4)
        c0,v0,o0=net.forward_coarse(obs_b[1],goal_b[1],window_floors=8)
        torch.testing.assert_close(c0,c_tile[1],atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(v0,v_tile[1],atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(o0,o_tile[1],atol=1e-4,rtol=1e-4)

    def test_fast_voxel_shadows_and_precompute_above_max_z(self):
        from voxelform.shadows import sun_path
        suns=sun_path(6)
        for nz in (10,24):
            occ=torch.zeros((nz,20,24),dtype=torch.bool)
            occ[:6,2:10,2:10]=True;occ[:9,11:18,12:20]=True
            dyn=FastVoxelShadows(occ.shape,suns)
            dyn.update(occ)
            _,p,n,_,faces=receivers(occ)
            vis=dyn.query(p,normals=n,faces=faces)
            front=(torch.as_tensor(suns,dtype=torch.float32)@n.T)>1e-6
            sr=front.nonzero(as_tuple=False)
            o_np=(p[sr[:,1]]+0.02*n[sr[:,1]]).numpy()
            d_np=suns[sr[:,0]]
            ids_np,_=dda_paths(o_np,d_np,occ.shape)
            exact=~np.append(occ.numpy().ravel(),False)[ids_np].any(axis=1)
            agree=float((vis[sr[:,0],sr[:,1]].numpy()==exact).mean())
            self.assertGreaterEqual(agree,0.985)
        scene=prepare_scene(ctx_fixture(),dict(site_id='test'),floors=10)
        slow=Environment(context=scene,sun_count=4,fast_self_shadows=False)
        fast=Environment(context=scene,sun_count=4,fast_self_shadows=True)
        r_s=slow.precompute_external();r_f=fast.precompute_external()
        self.assertEqual(r_s['knownReceivers'],r_f['knownReceivers'])
        self.assertTrue(torch.equal(slow.static_receivers.known,fast.static_receivers.known))
        self.assertTrue(torch.equal(slow.static_receivers.values,fast.static_receivers.values))

    def test_project_organic_3d_preserves_upper_cuts_when_ground_stem_severed(self):
        c=PatchConfig(mode='combined',allocation='feedback',facade_layers=2,minimum_patch=4,maximum_patches=4)
        e=Environment((6,16,16),sun_count=4,device='cpu',far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device='cpu'))
        solid=torch.zeros(e.shape,dtype=torch.bool)
        solid[0:2,6:10,6:10]=True
        solid[2:5,4:12,4:12]=True
        e.restore_occupancy(solid);e.termination=None;e.unified_limit=12;e.organic_3d_growth=True
        for z_neck in (0,1):
            masks=np.zeros((2,*e.shape),bool)
            masks[0,z_neck,6:10,6:10]=True
            masks[0,4,4:8,10:12]=True
            scores=masks.astype(np.float32)
            move=project_organic_3d(e,masks,scores,'remove',net_guided=False)
            self.assertIsNotNone(move)
            self.assertTrue(move.remove[4,4:8,10:12].all())
            self.assertFalse(move.remove[z_neck].any())
            self.assertTrue(valid_move(e,move))

if __name__=='__main__':unittest.main()
