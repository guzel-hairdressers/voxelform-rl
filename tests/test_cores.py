import functools
import unittest

import torch

from voxelform.cores import (
    CORE_TEMPLATES,
    UNREACHABLE_DIST,
    compute_core_distance,
    core_distance_badness,
    core_edge_gap,
    solve_cores,
    template_for_height,
)
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.unified import UnifiedLearner


class CoreTests(unittest.TestCase):
    def test_height_schedule_and_compact_grid_reservations(self):
        areas = [t.area for t in CORE_TEMPLATES]
        self.assertEqual(areas, sorted(areas))
        self.assertEqual(areas, [15, 25, 35, 63, 70])
        self.assertEqual(template_for_height(3).tier, "T1")
        self.assertEqual(template_for_height(8).tier, "T2")
        self.assertEqual(template_for_height(15).tier, "T3")
        self.assertEqual(template_for_height(24).tier, "T4")
        self.assertEqual(template_for_height(40).tier, "T5")
        self.assertIsNone(template_for_height(0))
        self.assertIsNone(template_for_height(61))
        # Metric zones exactly cover the enclosure; grid rounding is explicit.
        for tmpl in CORE_TEMPLATES:
            plan=tmpl.metric_plan
            self.assertGreaterEqual(tmpl.width,plan['metricWidth'])
            self.assertGreaterEqual(tmpl.length,plan['metricLength'])
            self.assertAlmostEqual(sum(z['w']*z['h'] for z in tmpl.zones),plan['metricArea'],places=4)
            self.assertEqual(sum(z['type']=='lift' for z in tmpl.zones),plan['liftCount'])
            self.assertEqual(sum(c['type']=='landing' for c in plan['components']),2*plan['stairCount'])
            self.assertEqual(sum(c['type']=='flight' for c in plan['components']),2*plan['stairCount'])
            for i,a in enumerate(tmpl.zones):
                for b in tmpl.zones[i+1:]:
                    overlap=max(0,min(a['x']+a['w'],b['x']+b['w'])-max(a['x'],b['x']))*max(0,min(a['y']+a['h'],b['y']+b['h'])-max(a['y'],b['y']))
                    self.assertLess(overlap,1e-8)
        self.assertEqual(CORE_TEMPLATES[0].metric_plan['liftCount'],0)
        self.assertLess(CORE_TEMPLATES[1].area,36)

    def test_24_storey_core_uses_active_scissor_reservation(self):
        occ=torch.zeros((24,16,16),dtype=torch.bool)
        occ[:,1:15,1:15]=True
        result=solve_cores(occ,min_core_spacing=10,max_core_dist=12)
        self.assertEqual(len(result.cores),1)
        core=result.cores[0]
        self.assertEqual((core.tier,core.floors,core.footprint_area),('T4',24,63))
        self.assertEqual(sorted((core.size_x,core.size_y)),[7,9])
        self.assertEqual(int(result.core_mask.sum()),24*63)
        self.assertEqual(template_for_height(24).metric_plan['stairLayout'],'scissor')
        self.assertEqual(core.to_dict()['liftCount'],4)
        self.assertEqual(core.to_dict()['layoutVersion'],'compact-core.v4')

    def test_active_environment_and_actor_use_four_lift_scissor_core(self):
        cfg=PatchConfig(eligibility='depth',facade_layers=2)
        env=Environment((24,16,16),sun_count=2,device='cpu',cores_enabled=True,
            frontier_factory=functools.partial(PatchFacades,config=cfg,depth_device='cpu'))
        env.organic_3d_growth=True
        learner=UnifiedLearner('cpu',123,cfg,organic_3d=True,net_guided=True)
        self.assertTrue(env.cores)
        self.assertTrue(all(c.tier=='T4' and c.to_dict()['liftCount']==4 for c in env.cores))
        obs=learner.observation(env)
        self.assertEqual(obs.shape[1],14)
        torch.testing.assert_close(obs[:,12],env.core_mask[:obs.shape[0]].float())
        torch.testing.assert_close(obs[:,13],env.core_dist_badness[:obs.shape[0]])
        self.assertEqual(env.metrics()['coreVoxels'],len(env.cores)*24*63)

    def test_tall_tower_cannot_cheat_with_low_rise_core(self):
        # 12F tower only 5x6 fits T2 (6-10F) but NOT T3 (7x5 for 11-18F), so T1/T2 must NOT place inside it
        narrow = torch.zeros((14, 12, 14), dtype=torch.bool)
        narrow[:12, 2:7, 2:8] = True
        res_narrow = solve_cores(narrow, min_core_spacing=10, max_core_dist=12)
        self.assertEqual(len(res_narrow.cores), 0)
        self.assertAlmostEqual(res_narrow.metrics["coreUnreachableFraction"], 1.0, places=5)

        # 12F tower 12x12 fits a T3 (7x5x12F) core cleanly with surrounding floorplate attachment
        tower = torch.zeros((14, 16, 16), dtype=torch.bool)
        tower[:12, 2:14, 2:14] = True
        res_tower = solve_cores(tower, min_core_spacing=10, max_core_dist=12)
        self.assertEqual(len(res_tower.cores), 1)
        c = res_tower.cores[0]
        self.assertEqual(c.tier, "T3")
        self.assertEqual(sorted((c.size_y, c.size_x)), [5, 7])
        self.assertEqual(c.floors, 12)
        self.assertAlmostEqual(res_tower.metrics["coreUncoveredFraction"], 0.0, places=5)

    def test_grounded_shaft_cantilever_reach_and_attachment(self):
        # Floating overhang without z=0 cannot hold a core; core sits on grounded wing and serves bridge
        occ = torch.zeros((6, 12, 20), dtype=torch.bool)
        occ[:4, 1:9, 1:9] = True  # Grounded main wing (8x8 fits T1 5x6)
        occ[2:4, 3:7, 9:17] = True  # Cantilevered bridge at z=2..3 (empty z=0..1)
        res = solve_cores(occ, min_core_spacing=10, max_core_dist=12)
        self.assertEqual(len(res.cores), 1)
        c = res.cores[0]
        self.assertEqual(c.tier, "T1")
        self.assertLessEqual(c.x0 + c.size_x, 9)
        self.assertTrue(bool(occ[0, c.y0 : c.y0 + c.size_y, c.x0 : c.x0 + c.size_x].all()))
        # Floors 2..3 reach out onto the cantilever bridge (finite distance)
        self.assertTrue(bool((res.core_distance[2:4, 3:7, 9:17] < 20.0).all()))

        # Isolated 3x5x4 block with zero surrounding perimeter fails attachment check
        isolated = torch.zeros((6, 10, 10), dtype=torch.bool)
        isolated[:4, 2:7, 2:5] = True
        self.assertEqual(len(solve_cores(isolated).cores), 0)

    def test_min_core_spacing_and_multiple_cores_on_wide_slab(self):
        # 10F bar 12x44 with max_core_dist=12 needs 2 T2 (6x8) cores spaced >= 10 m apart
        slab = torch.zeros((10, 16, 48), dtype=torch.bool)
        slab[:10, 2:14, 2:46] = True
        res = solve_cores(slab, min_core_spacing=10, max_core_dist=12)
        self.assertGreaterEqual(len(res.cores), 2)
        for c in res.cores:
            self.assertEqual(c.tier, "T2")
        for i in range(len(res.cores)):
            for j in range(i + 1, len(res.cores)):
                self.assertGreaterEqual(core_edge_gap(res.cores[i], res.cores[j]), 10.0 - 1e-6)
        self.assertLess(res.metrics["coreUncoveredFraction"], 0.10)

        # Very strict spacing allows only 1 core
        res_strict = solve_cores(slab, min_core_spacing=34, max_core_dist=12)
        self.assertEqual(len(res_strict.cores), 1)

    def test_in_floor_walk_around_courtyard_and_badness_monotonicity(self):
        # U-shape around empty courtyard; distance must walk around the U rather than jumping air
        occ = torch.zeros((3, 14, 16), dtype=torch.bool)
        occ[:2, 2:12, 2:14] = True
        occ[:2, 5:12, 6:10] = False  # Courtyard open toward y >= 5
        core_mask = torch.zeros_like(occ)
        core_mask[:1, 6:8, 3:5] = True  # Only z=0 has core, z=1 has no core
        dist = compute_core_distance(occ, core_mask)
        # Other tip of U at (z=0, y=11, x=13) cannot cut across empty air (y=5..11, x=6..9)
        tip_dist = float(dist[0, 11, 13].item())
        self.assertGreater(tip_dist, 14.0)
        self.assertLess(tip_dist, 30.0)
        # Unserved upper floor z=1 must be marked UNREACHABLE_DIST
        self.assertTrue(bool((dist[1][occ[1]] >= UNREACHABLE_DIST - 1.0).all()))
        bad = core_distance_badness(dist, occ, max_core_dist=12, core_mask=core_mask)
        self.assertEqual(float(bad[~occ].abs().max().item()), 0.0)
        self.assertEqual(float(bad[core_mask].abs().max().item()), 0.0)
        self.assertAlmostEqual(float(bad[1][occ[1]].min().item()), 1.0, places=5)
        self.assertLess(float(bad[0, 5, 5].item()), float(bad[0, 11, 13].item()))

    def test_environment_and_14ch_observation_integration(self):
        cfg = PatchConfig(mode="combined", allocation="feedback", eligibility="depth", facade_layers=2, minimum_patch=4, maximum_patches=4)
        env_off = Environment((6, 16, 18), sun_count=4, device="cpu", far_tolerance=0.1, frontier_factory=functools.partial(PatchFacades, config=cfg, depth_device="cpu"), cores_enabled=False)
        env_off.precompute_external()
        env_off.organic_3d_growth = True
        learner = UnifiedLearner("cpu", 123, cfg, amount_feedback=True, net_guided=True, organic_3d=True)
        obs_off = learner.observation(env_off)
        self.assertEqual(obs_off.shape[1], 14)
        self.assertEqual(float(obs_off[:, 12:14].abs().sum().item()), 0.0)
        self.assertEqual(env_off.metrics()["coreCount"], 0)

        env_on = Environment((6, 16, 24), sun_count=4, device="cpu", far_tolerance=0.1, frontier_factory=functools.partial(PatchFacades, config=cfg, depth_device="cpu"), cores_enabled=True, min_core_spacing=10, max_core_dist=12)
        env_on.precompute_external()
        env_on.organic_3d_growth = True
        m_on = env_on.metrics()
        self.assertTrue(m_on["coresEnabled"])
        self.assertGreaterEqual(m_on["coreCount"], 1)
        obs_on = learner.observation(env_on)
        self.assertEqual(obs_on.shape[1], 14)
        self.assertGreater(float(obs_on[:, 12].sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
