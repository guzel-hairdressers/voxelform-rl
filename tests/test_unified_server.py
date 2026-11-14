import unittest
from unittest.mock import Mock,patch
from fastapi.testclient import TestClient
from tests.test_unified import fixture
import unified_server as server

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

if __name__=='__main__':unittest.main()
