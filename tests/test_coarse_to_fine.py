import copy
import functools
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from unified_server import organic_objective
from voxelform.coarse_to_fine import (
    CoarseEnv,
    CoarseToFineLearner,
    run_cascade_episode,
    run_episode,
    tier_cascade_blocks,
)
from voxelform.decoder_diagnostics import DecoderDiagnostics
from voxelform.environment import Environment
from voxelform.objectives import organic_step_reward, organic_terminal_reward
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.unified import clean_floor_tensor, min_room_width


class CoarseToFineTests(unittest.TestCase):
    def _make_env(self, shape=(10, 24, 24), plan="rectangle", cores_enabled=False):
        cfg = PatchConfig(
            mode="combined",
            allocation="feedback",
            eligibility="depth",
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
            device="cpu",
            plan=plan,
            target_far=3.0,
            far_tolerance=0.1,
            frontier_factory=functools.partial(
                PatchFacades, config=cfg, depth_device="cpu"
            ),
            cores_enabled=cores_enabled,
        )
        env.precompute_external()
        solid = env.allowed.clone()
        solid[6:] = False
        env.restore_occupancy(solid)
        return env, cfg

    def test_opt_in_decoder_measurements_preserve_the_rollout(self):
        env, cfg = self._make_env()
        cenv = CoarseEnv(env, block=(1, 2, 2))
        learner = CoarseToFineLearner(device="cpu", seed=123, fine_config=cfg)

        torch.manual_seed(42)
        expected = run_cascade_episode(
            env, [cenv], learner, 6, organic_objective, training=False
        )
        occupancy = env.occupied.clone()

        torch.manual_seed(42)
        probe = DecoderDiagnostics()
        learner.decoder_diagnostics = probe
        actual = run_cascade_episode(
            env, [cenv], learner, 6, organic_objective, training=False
        )
        self.assertEqual(actual["trace"], expected["trace"])
        torch.testing.assert_close(env.occupied, occupancy)
        accepted = [row for row in probe.attempts if row.accepted_digest is not None]
        self.assertEqual(len(accepted), actual["steps"])
        for row in probe.attempts:
            self.assertLessEqual(row.kept_voxels, row.proposed_voxels)
            self.assertLessEqual(row.proposed_voxels, row.requested_voxels)

    def test_imitation_update_is_independent_of_episode_reward(self):
        env, cfg = self._make_env()
        teacher_trained = CoarseToFineLearner(
            seed=123, fine_config=cfg, training_objective="imitation"
        )
        differently_rewarded = CoarseToFineLearner(
            seed=123, fine_config=cfg, training_objective="imitation"
        )
        torch.manual_seed(42)
        run_cascade_episode(
            env,
            [CoarseEnv(env, block=(1, 2, 2))],
            teacher_trained,
            6,
            organic_objective,
            training=True,
            rollout_steps=1000,
        )
        self.assertTrue(
            any(row["teacher_mean"] is not None for row in teacher_trained.tape)
        )
        differently_rewarded.tape = copy.deepcopy(teacher_trained.tape)
        for row in differently_rewarded.tape:
            row["reward"] = torch.tensor(-100.0)
        self.assertGreater(teacher_trained.finish()["gradientNorm"], 0.0)
        differently_rewarded.finish()
        torch.testing.assert_close(
            teacher_trained.model.state_dict(),
            differently_rewarded.model.state_dict(),
            rtol=0,
            atol=0,
        )

    def test_adaptive_retry_shrinks_after_the_first_rejection(self):
        env, cfg = self._make_env()
        cenv = CoarseEnv(env, block=(1, 2, 2))
        cenv.reset(6)
        budgets = {}
        for schedule in ("legacy", "adaptive"):
            learner = CoarseToFineLearner(
                seed=123, fine_config=cfg, decoder_retry_schedule=schedule
            )
            learner.decoder_diagnostics = DecoderDiagnostics()
            torch.manual_seed(42)
            with patch("voxelform.coarse_to_fine.project_coarse_3d", return_value=None):
                learner.choose_stage(cenv, is_coarse=True, training=False, retries=2)
            budgets[schedule] = [
                row.requested_voxels for row in learner.decoder_diagnostics.attempts
            ]
        self.assertEqual(budgets["legacy"][0], budgets["legacy"][1])
        self.assertEqual(budgets["adaptive"][0], budgets["legacy"][0])
        self.assertLess(budgets["adaptive"][1], budgets["adaptive"][0])

    def test_coarse_volume_equivalence_and_upsample_invariants(self):
        for plan, shape, bz in (
            ("triangle", (10, 22, 22), 1),
            ("rectangle", (12, 24, 24), 2),
        ):
            env, _cfg = self._make_env(shape=shape, plan=plan)
            cenv = CoarseEnv(env, block=(bz, 2, 2))
            cenv.reset(initial_storeys=6)
            Z, Y, X = env.shape
            exp = (
                cenv.occupied.repeat_interleave(bz, 0)
                .repeat_interleave(2, 1)
                .repeat_interleave(2, 2)[:Z, :Y, :X]
                & env.allowed
            )
            self.assertEqual(cenv.count_fine, int(exp.sum()))
            cenv.upsample_to_fine()
            self.assertFalse(bool((env.occupied & ~env.allowed).any()))
            self.assertTrue(bool(env.occupied[0].any()))
            mw = min_room_width(env)
            if mw > 1:
                slim = (
                    (env.occupied & ~clean_floor_tensor(env.occupied, mw))
                    & clean_floor_tensor(env.allowed, mw)
                ).sum()
                self.assertEqual(int(slim), 0)

    def test_two_stage_episode_and_shared_gradients_for_both_blocks(self):
        for bz, shape in ((1, (10, 24, 24)), (2, (12, 24, 24))):
            env, cfg = self._make_env(shape=shape, plan="rectangle")
            cenv = CoarseEnv(env, block=(bz, 2, 2))
            learner = CoarseToFineLearner(
                device="cpu", seed=123, block=(bz, 2, 2), fine_config=cfg
            )
            rows = []
            for _ in range(2):
                rows.append(
                    run_episode(
                        env,
                        cenv,
                        learner,
                        6,
                        organic_objective,
                        training=True,
                        rollout_steps=999,
                    )
                )
            self.assertTrue(all(r["coarseSteps"] >= 1 for r in rows), rows)
            res = learner.finish(0.0)
            self.assertGreater(res["gradientNorm"], 0.0)
            self.assertIsNotNone(learner.model.coarse_stem[0].weight.grad)
            self.assertGreater(
                float(learner.model.coarse_stem[0].weight.grad.abs().max()), 0.0
            )
            if any(r["fineSteps"] > 0 for r in rows):
                self.assertIsNotNone(learner.model.stem[0].weight.grad)
                self.assertGreater(
                    float(learner.model.stem[0].weight.grad.abs().max()), 0.0
                )
            self.assertIsNotNone(learner.model.ctx_xy1.weight.grad)
            self.assertGreater(
                float(learner.model.ctx_xy1.weight.grad.abs().max()), 0.0
            )

    def test_three_and_four_stage_cascades_train_cleanly(self):
        env, cfg = self._make_env(shape=(12, 32, 32), plan="rectangle")
        c8 = CoarseEnv(env, block=(3, 8, 8))
        c4 = CoarseEnv(env, block=(1, 4, 4))
        c2 = CoarseEnv(env, block=(1, 2, 2))
        learner = CoarseToFineLearner(
            device="cpu", seed=123, block=(1, 2, 2), fine_config=cfg
        )
        r3 = run_cascade_episode(
            env,
            [c4, c2],
            learner,
            8,
            organic_objective,
            training=True,
            rollout_steps=999,
        )
        r4 = run_cascade_episode(
            env,
            [c8, c4, c2],
            learner,
            8,
            organic_objective,
            training=True,
            rollout_steps=999,
        )
        self.assertEqual(len(r3["stageSteps"]), 3)
        self.assertEqual(len(r4["stageSteps"]), 4)
        res = learner.finish(0.0)
        self.assertGreater(res["gradientNorm"], 0.0)
        self.assertIsNotNone(learner.model.macro_stem[0].weight.grad)
        self.assertIsNotNone(learner.model.coarse_stem[0].weight.grad)

    def test_tier_cascade_blocks_thresholds(self):
        self.assertEqual(tier_cascade_blocks(SimpleNamespace(plot_area=1500.0)), ())
        self.assertEqual(tier_cascade_blocks(SimpleNamespace(plot_area=2499.9)), ())
        self.assertEqual(
            tier_cascade_blocks(SimpleNamespace(plot_area=2500.0)), ((1, 2, 2),)
        )
        self.assertEqual(
            tier_cascade_blocks(SimpleNamespace(plot_area=3799.9)), ((1, 2, 2),)
        )
        self.assertEqual(
            tier_cascade_blocks(SimpleNamespace(plot_area=3800.0)),
            ((1, 4, 4), (1, 2, 2)),
        )
        self.assertEqual(
            tier_cascade_blocks(SimpleNamespace(plot_area=5800.0)),
            ((1, 4, 4), (1, 2, 2)),
        )

    def test_multiscale_14ch_cores_empty_and_coarse_with_checkpoint(self):
        env, cfg = self._make_env(
            shape=(12, 32, 32), plan="rectangle", cores_enabled=True
        )
        c2 = CoarseEnv(env, block=(1, 2, 2))
        c2.reset(6)
        learner = CoarseToFineLearner(
            device="cpu", seed=123, block=(1, 2, 2), fine_config=cfg, input_channels=14
        )
        self.assertEqual(c2.observation(include_cores=True).shape[1], 14)
        self.assertEqual(learner._fine_obs(env).shape[1], 14)
        r_direct = run_cascade_episode(
            env,
            [],
            learner,
            6,
            organic_objective,
            training=True,
            rollout_steps=999,
            step_reward_fn=organic_step_reward,
            terminal_reward_fn=organic_terminal_reward,
        )
        self.assertEqual(len(r_direct["stageSteps"]), 1)
        self.assertTrue(
            all(
                key in r_direct
                for key in ("steps", "attempts", "budget", "reward", "diagnostics")
            )
        )
        r_two = run_cascade_episode(
            env,
            [c2],
            learner,
            6,
            organic_objective,
            training=True,
            rollout_steps=999,
            step_reward_fn=organic_step_reward,
            terminal_reward_fn=organic_terminal_reward,
        )
        self.assertEqual(len(r_two["stageSteps"]), 2)
        self.assertGreater(len(learner.tape), 0)
        res = learner.finish(0.0)
        self.assertTrue(math.isfinite(res["gradientNorm"]))
        self.assertGreater(res["gradientNorm"], 0.0)
        saved = learner.state()
        self.assertEqual(saved["format"], "voxelform.multiscale.v1")
        self.assertEqual(saved["inputChannels"], 14)
        restored = CoarseToFineLearner(device="cpu", seed=456, fine_config=cfg)
        restored.load(saved)
        self.assertEqual(restored.input_channels, 14)
        self.assertEqual(restored.updates, learner.updates)
        for k, v in learner.model.state_dict().items():
            self.assertTrue(torch.equal(v, restored.model.state_dict()[k]))

    def test_greedy_heuristic_mode_runs_cascade_without_neural_forward(self):
        env, cfg = self._make_env(
            shape=(12, 32, 32), plan="rectangle", cores_enabled=True
        )
        c2 = CoarseEnv(env, block=(1, 2, 2))
        learner = CoarseToFineLearner(
            device="cpu",
            seed=123,
            block=(1, 2, 2),
            fine_config=cfg,
            input_channels=14,
            evaluation_noise_scale=0.0,
            heuristic_only=True,
        )

        def _forbid(*_args, **_kwargs):
            raise AssertionError("heuristic_only must not call neural network")

        learner.model.forward_stage_batch = _forbid
        res = run_cascade_episode(
            env, [c2], learner, 6, organic_objective, training=False
        )
        self.assertEqual(len(res["stageSteps"]), 2)
        self.assertGreater(res["steps"], 0)


if __name__ == "__main__":
    unittest.main()
