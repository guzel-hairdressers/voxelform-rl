import functools,tempfile,unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from fastapi.testclient import TestClient
from tests.test_context import fixture as ctx_fixture
from tests.test_unified import fixture
import unified_server as server
from voxelform.context import prepare_scene
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig,PatchFacades
from voxelform.unified import UnifiedLearner

class LivePreview(unittest.TestCase):
    def session(self):
        s=server.UnifiedSession(device='cpu',autostart=False);s.env,s.policy=fixture();s.config=s.policy.config
        s.policy.floor_copy=True;s.policy.net_guided=True;s.ready=True;s.env.unified_actions=0
        return s
    def test_rejected_draw_consumes_budget_without_ending_immediately(self):
        s=self.session();s.env.unified_limit=2
        s.policy.choose=Mock(return_value=None);s.policy.last_operation='exchange';s.policy.last_rejection='mask_generation_failed'
        s.single=True;s.advance_one();self.assertEqual(s.env.unified_actions,1);self.assertFalse(s.env.terminal)
        s.single=True;s.advance_one();self.assertEqual(s.env.unified_actions,2);self.assertEqual(s.env.termination,'step_budget_exhausted')
        self.assertTrue(s.episode_recorded)
    def test_terminal_rollout_uses_zero_bootstrap(self):
        s=self.session();s.training=True;s.env.termination='far_reached'
        s.policy=Mock();s.policy.tape=[object()];s.learn_rollout();s.policy.finish.assert_called_once_with(0.)
        s.policy.fields.assert_not_called()
    def test_far_change_resets_attempt_clock(self):
        s=self.session();s.env.unified_actions=9;s.env.termination='step_budget_exhausted'
        s.configure_far(3.,'medium');self.assertEqual(s.env.unified_actions,0);self.assertIsNone(s.env.termination)
    def test_devox_is_finished_generation_only_and_revision_checked(self):
        s=self.session();client=TestClient(server.app)
        with patch.object(server,'session',s):
            self.assertEqual(client.get('/api/devoxelize_live').status_code,409)
            s.env.termination='far_reached';s.training=True
            self.assertEqual(client.get('/api/devoxelize_live').status_code,409)
            s.training=False
            self.assertEqual(client.get('/api/devoxelize_live?revision=99').status_code,409)
            self.assertEqual(client.get('/api/devoxelize_live?method=fake').status_code,400)
            response=client.get('/api/devoxelize_live?method=extruded_slabs&revision=0')
            self.assertEqual(response.status_code,200);self.assertGreater(response.json()['stats']['triangles'],0)
    def test_grow_probe_can_be_used_above_far_and_bad_probe_does_not_kill_worker(self):
        s=self.session();s.start(False,True,'grow');s.advance_one();self.assertEqual(s.env.edits,1)
        with self.assertRaises(ValueError):s.start(False,True,'unknown')
        self.assertIsNone(s.error)
    def test_manual_policy_export_import_and_bad_input(self):
        s=self.session();client=TestClient(server.app)
        with patch.object(server,'session',s):
            saved=client.get('/api/policy');self.assertEqual(saved.status_code,200)
            self.assertEqual(client.post('/api/policy',content=saved.content).status_code,200)
            self.assertTrue(s.policy.floor_copy and s.policy.net_guided)
            self.assertEqual(client.post('/api/policy',content=b'bad checkpoint').status_code,400)
    def test_settings_apply_and_unsupported_modes_are_rejected(self):
        s=self.session();s.configure_masks('correlated',False,2,100,'feedback')
        self.assertEqual(s.policy.config.maximum_voids,2);self.assertEqual(s.env.frontier.config.eligibility,'facade')
        with self.assertRaises(ValueError):s.configure_masks('independent',True,1,0,'feedback')
    def test_training_continues_across_multiple_episodes_including_budget_exhaustion(self):
        s=self.session();s.env.unified_limit=2;s.start(training=True)
        with patch.object(s.policy,'choose',return_value=None):
            s.policy.last_operation='exchange';s.policy.last_rejection='mask_generation_failed'
            s.advance_one();s.advance_one()
        self.assertEqual(s.episodes,1);self.assertEqual(s.env.termination,'step_budget_exhausted')
        self.assertTrue(s.running)
        s.advance_one()
        self.assertEqual(s.episodes,1);self.assertEqual(s.env.unified_actions,1);self.assertTrue(s.running)
        while s.episodes<2:
            s.advance_one()
        self.assertTrue(s.running);self.assertGreaterEqual(s.episodes,2)
        s.pause();self.assertFalse(s.running)
        s.next_episode();s.start(training=False)
        while s.running:
            s.advance_one()
        self.assertFalse(s.running)

    def _organic_session(self):
        c=PatchConfig(mode='combined',allocation='feedback',eligibility='depth',facade_layers=2,minimum_patch=4,maximum_patches=4,maximum_voids=1,minimum_void_area=4,batched=True,vertical_interior=False,budget_reserve_fraction=0.2)
        e=Environment((8,12,16),sun_count=4,device='cpu',far_tolerance=.1,frontier_factory=functools.partial(PatchFacades,config=c,depth_device='cpu'))
        solid=e.allowed.clone();solid[4:]=False;e.restore_occupancy(solid);e.termination=None;e.unified_limit=12;e.precompute_external()
        p=UnifiedLearner('cpu',123,c,amount_feedback=True,net_guided=True,floor_copy=False,organic_3d=True)
        e.organic_3d_growth=True;e.floor_copy_growth=False
        s=server.UnifiedSession(device='cpu',autostart=False)
        s.env,s.policy,s.config=e,p,c
        s.rollout_steps=8;s.audit_steps=3
        s.last_objective,s.last_diag=server.organic_objective(e)
        s.ready=True;s.env.unified_actions=0
        return s

    def test_exact_disk_scene_cache_hit_matches_cold_build(self):
        scene=prepare_scene(ctx_fixture(),dict(site_id='test'),floors=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            cold=Environment(context=scene,sun_count=4,scene_cache_dir=tmpdir)
            self.assertFalse(cold._scene_cache_hit)
            cold.precompute_external()
            warm=Environment(context=scene,sun_count=4,scene_cache_dir=tmpdir)
            self.assertTrue(warm._scene_cache_hit)
            warm.precompute_external()
            torch.testing.assert_close(cold.sky_boundary,warm.sky_boundary)
            self.assertTrue(torch.equal(cold.static_receivers.known,warm.static_receivers.known))
            self.assertTrue(torch.equal(cold.static_receivers.values,warm.static_receivers.values))
            self.assertTrue(torch.equal(cold.visible,warm.visible))
            torch.testing.assert_close(cold.quality,warm.quality)

    def test_organic_3d_session_continuous_training_roundtrip_and_cantilever_devox(self):
        s=self._organic_session()
        client=TestClient(server.app)
        with patch.object(server,'session',s):
            saved=client.get('/api/policy');self.assertEqual(saved.status_code,200)
            self.assertEqual(client.post('/api/policy',content=saved.content).status_code,200)
            self.assertTrue(s.policy.organic_3d and not s.policy.floor_copy)
            self.assertEqual(s.state()['settings']['growthMode'],'organic_3d')
            s.start(training=True)
            steps=0
            while s.episodes<3 and steps<200:
                s.advance_one();steps+=1
            self.assertTrue(s.running)
            self.assertGreaterEqual(s.episodes,3)
            self.assertGreater(s.policy.updates,0)
            s.pause()
            # Test devoxelization on a finished cantilevered organic occupancy
            cant=np.zeros(s.env.shape,bool);cant[:3,2:8,2:8]=True;cant[3,2:8,6:12]=True
            s.env.restore_occupancy(cant);s.env.termination='far_reached';s.training=False;s.revision+=1
            resp=client.get(f'/api/devoxelize_live?method=correlated_loft&revision={s.revision}')
            self.assertEqual(resp.status_code,200);self.assertGreater(resp.json()['stats']['triangles'],0)

    def test_state_view_filtering_snapshot_reset_and_dashboard_telemetry(self):
        s=self._organic_session()
        s._capture_initial_snapshot()
        init_occ=s.env.occupied.clone()
        init_light=s.env.light.clone()
        client=TestClient(server.app)
        with patch.object(server,'session',s):
            solid_state=client.get('/api/state?view=solid').json()
            self.assertIn('occupied',solid_state)
            self.assertIn('dashboard',solid_state)
            for k in ('baselineObjective','bestObjective','recentObjective','opCounts'):
                self.assertIn(k,solid_state['dashboard'])
            for heavy in ('light','facadeDepth','depthBadness','noise','usable'):
                self.assertNotIn(heavy,solid_state)
            usable_state=client.get('/api/state?view=usable').json()
            self.assertIn('usable',usable_state)
            for heavy in ('light','facadeDepth','depthBadness','noise'):
                self.assertNotIn(heavy,usable_state)
            full_state=client.get('/api/state').json()
            for heavy in ('light','facadeDepth','depthBadness','noise','usable'):
                self.assertIn(heavy,full_state)
            # Step, then restore via cached initial snapshot twice to ensure immutability
            for _ in range(2):
                s.start(False,True,'exchange');s.advance_one()
                self.assertEqual(s.env.edits,1)
                self.assertFalse(torch.equal(s.env.occupied,init_occ))
                s.next_episode()
                self.assertEqual(s.env.edits,0)
                self.assertTrue(torch.equal(s.env.occupied,init_occ))
                torch.testing.assert_close(s.env.light,init_light)

    def test_reset_weights_and_envelope_controls(self):
        s=self._organic_session()
        client=TestClient(server.app)
        with patch.object(server,'session',s):
            s.start(training=True)
            steps=0
            while s.policy.updates<1 and steps<100:
                s.advance_one();steps+=1
            self.assertGreaterEqual(s.policy.updates,1)
            before_w=[v.detach().clone() for v in s.policy.model.state_dict().values()]
            r=client.post('/api/control',json={'command':'reset_weights'})
            self.assertEqual(r.status_code,200)
            self.assertEqual(s.policy.updates,0)
            self.assertEqual(s.episodes,0)
            self.assertEqual(s.recent,[])
            after_w=list(s.policy.model.state_dict().values())
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before_w,after_w)))
            r_bad=client.post('/api/control',json={'command':'envelope','initialStoreys':15,'ceilingStoreys':10})
            self.assertEqual(r_bad.status_code,400)
            r_ok=client.post('/api/control',json={'command':'envelope','initialStoreys':5,'ceilingStoreys':14})
            self.assertEqual(r_ok.status_code,200)
            self.assertEqual(s.env.shape[0],14)
            self.assertEqual(s.initial_storeys,5)
            self.assertEqual(s.ceiling_storeys,14)
            self.assertTrue(bool(s.env.occupied[:5].any()))
            self.assertFalse(bool(s.env.occupied[5:].any()))
            for a,b in zip(after_w,s.policy.model.state_dict().values()):
                torch.testing.assert_close(a,b)

    def test_cores_control_and_core_distance_view(self):
        s=self._organic_session()
        s.configure_cores(enabled=True,min_spacing=10,max_dist=12)
        client=TestClient(server.app)
        with patch.object(server,'session',s):
            st=client.get('/api/state?view=coreDistance').json()
            self.assertIn('cores',st)
            self.assertIn('coreMask',st)
            self.assertIn('coreDistance',st)
            self.assertTrue(st['settings']['coresEnabled'])
            self.assertGreaterEqual(st['metrics']['coreCount'],1)
            self.assertIn('coreRatio',st['metrics'])
            self.assertIsNotNone(st['metrics'].get('coreScore'))
            self.assertIsNotNone(st['metrics'].get('coreCoverage'))
            r_plans=client.get('/core-plans')
            self.assertEqual(r_plans.status_code,200)
            r=client.post('/api/control',json={'command':'cores','coresEnabled':True,'minCoreSpacing':6,'maxCoreDist':8})
            self.assertEqual(r.status_code,200)
            st2=client.get('/api/state?view=coreDistance').json()
            self.assertEqual(st2['settings']['minCoreSpacing'],6)
            self.assertEqual(st2['settings']['maxCoreDist'],8)
            r_off=client.post('/api/control',json={'command':'cores','coresEnabled':False})
            self.assertEqual(r_off.status_code,200)
            st3=client.get('/api/state?view=solid').json()
            self.assertFalse(st3['settings']['coresEnabled'])
            self.assertEqual(st3['metrics']['coreCount'],0)
            self.assertIsNone(st3['metrics'].get('coreScore'))
            self.assertIsNone(st3['metrics'].get('coreCoverage'))



class CorePlanAPI(unittest.TestCase):
    def test_canonical_plan_data_and_svg_endpoints(self):
        client=TestClient(server.app)
        result=client.get('/api/core-plans');self.assertEqual(result.status_code,200)
        plans=result.json()['plans'];self.assertEqual(len(plans),5)
        self.assertEqual(plans[0]['liftCount'],0)
        self.assertEqual((plans[1]['width'],plans[1]['length']),(5,5))
        for p in plans:
            response=client.get('/api/core-plans/'+p['tier']+'.svg')
            self.assertEqual(response.status_code,200)
            self.assertIn('image/svg+xml',response.headers['content-type'])
            if p['stairLayout']=='scissor':
                self.assertIn('A: +1.5 m',response.text)
                self.assertIn('B: +1.5 m',response.text)
                self.assertIn('2 interlocking routes',response.text)
            else:
                self.assertIn('Half landing',response.text)
        self.assertEqual(client.get('/api/core-plans/unknown.svg').status_code,404)

if __name__=='__main__':unittest.main()
