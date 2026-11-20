import functools
import unittest

import torch

from unified_server import organic_objective
from voxelform.coarse_to_fine import CoarseEnv, CoarseToFineLearner, run_cascade_episode, run_episode
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.unified import clean_floor_tensor, min_room_width


class CoarseToFineTests(unittest.TestCase):
    def _make_env(self, shape=(10, 24, 24), plan='rectangle'):
        cfg = PatchConfig(
            mode='combined',
            allocation='feedback',
            eligibility='depth',
            facade_layers=2,
            minimum_patch=4,
            maximum_patches=4,
            maximum_voids=1,
            minimum_void_area=4,
            batched=True,
            vertical_interior=False,
            budget_reserve_fraction=0.2,
        )
        env = Environment(
            shape=shape,
            sun_count=4,
            device='cpu',
            plan=plan,
            target_far=3.0,
            far_tolerance=0.1,
            frontier_factory=functools.partial(PatchFacades, config=cfg, depth_device='cpu'),
        )
        env.precompute_external()
        solid = env.allowed.clone()
        solid[6:] = False
        env.restore_occupancy(solid)
        return env, cfg

    def test_coarse_volume_equivalence_and_upsample_invariants(self):
        for plan, shape, bz in (('triangle', (10, 22, 22), 1), ('rectangle', (12, 24, 24), 2)):
            env, _cfg = self._make_env(shape=shape, plan=plan)
            cenv = CoarseEnv(env, block=(bz, 2, 2))
            cenv.reset(initial_storeys=6)
            Z, Y, X = env.shape
            exp = cenv.occupied.repeat_interleave(bz, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)[:Z, :Y, :X] & env.allowed
            self.assertEqual(cenv.count_fine, int(exp.sum()))
            cenv.upsample_to_fine()
            self.assertFalse(bool((env.occupied & ~env.allowed).any()))
            self.assertTrue(bool(env.occupied[0].any()))
            mw = min_room_width(env)
            if mw > 1:
                slim = ((env.occupied & ~clean_floor_tensor(env.occupied, mw)) & clean_floor_tensor(env.allowed, mw)).sum()
                self.assertEqual(int(slim), 0)

    def test_two_stage_episode_and_shared_gradients_for_both_blocks(self):
        for bz, shape in ((1, (10, 24, 24)), (2, (12, 24, 24))):
            env, cfg = self._make_env(shape=shape, plan='rectangle')
            cenv = CoarseEnv(env, block=(bz, 2, 2))
            learner = CoarseToFineLearner(device='cpu', seed=123, block=(bz, 2, 2), fine_config=cfg)
            rows = []
            for _ in range(2):
                rows.append(run_episode(env, cenv, learner, 6, organic_objective, training=True, rollout_steps=999))
            self.assertTrue(all(r['coarseSteps'] >= 1 for r in rows), rows)
            res = learner.finish(0.0)
            self.assertGreater(res['gradientNorm'], 0.0)
            self.assertIsNotNone(learner.model.coarse_stem[0].weight.grad)
            self.assertGreater(float(learner.model.coarse_stem[0].weight.grad.abs().max()), 0.0)
            if any(r['fineSteps'] > 0 for r in rows):
                self.assertIsNotNone(learner.model.stem[0].weight.grad)
                self.assertGreater(float(learner.model.stem[0].weight.grad.abs().max()), 0.0)
            self.assertIsNotNone(learner.model.ctx_xy1.weight.grad)
            self.assertGreater(float(learner.model.ctx_xy1.weight.grad.abs().max()), 0.0)

    def test_three_and_four_stage_cascades_train_cleanly(self):
        env, cfg = self._make_env(shape=(12, 32, 32), plan='rectangle')
        c8 = CoarseEnv(env, block=(3, 8, 8))
        c4 = CoarseEnv(env, block=(1, 4, 4))
        c2 = CoarseEnv(env, block=(1, 2, 2))
        learner = CoarseToFineLearner(device='cpu', seed=123, block=(1, 2, 2), fine_config=cfg)
        r3 = run_cascade_episode(env, [c4, c2], learner, 8, organic_objective, training=True, rollout_steps=999)
        r4 = run_cascade_episode(env, [c8, c4, c2], learner, 8, organic_objective, training=True, rollout_steps=999)
        self.assertEqual(len(r3['stageSteps']), 3)
        self.assertEqual(len(r4['stageSteps']), 4)
        res = learner.finish(0.0)
        self.assertGreater(res['gradientNorm'], 0.0)
        self.assertIsNotNone(learner.model.macro_stem[0].weight.grad)
        self.assertIsNotNone(learner.model.coarse_stem[0].weight.grad)


if __name__ == '__main__':
    unittest.main()
