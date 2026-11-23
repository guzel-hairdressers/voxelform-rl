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
from voxelform.unified import UnifiedLearner, clean_floor_tensor, min_room_width


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

    def test_tier_cascade_blocks_macro_and_xl8_schedules(self):
        for area, expected_macro4, expected_xl8 in (
            (2000.0, (), ()),
            (3000.0, ((1, 4, 4), (1, 2, 2)), ((1, 2, 2),)),
            (3900.0, ((1, 4, 4), (1, 2, 2)), ((1, 4, 4), (1, 2, 2))),
            (
                4500.0,
                ((1, 8, 8), (1, 4, 4), (1, 2, 2)),
                ((1, 8, 8), (1, 4, 4), (1, 2, 2)),
            ),
        ):
            env = SimpleNamespace(plot_area=area)
            self.assertEqual(tier_cascade_blocks(env, "tier_macro4"), expected_macro4)
            self.assertEqual(
                tier_cascade_blocks(env, "tier_adaptive_xl8"), expected_xl8
            )
            self.assertEqual(
                tier_cascade_blocks(env, "tier_meso2"),
                () if area < 2500.0 else ((1, 2, 2),),
            )

    def test_dense_potential_and_on_policy_teacher_distillation(self):
        env, cfg = self._make_env(shape=(12, 32, 32), plan="rectangle")
        env.plot_area = 3200.0
        c2 = CoarseEnv(env, block=(1, 2, 2))
        c2.plot_area = 3200.0
        teacher = CoarseToFineLearner(device="cpu", seed=99, fine_config=cfg)
        student = CoarseToFineLearner(
            device="cpu",
            seed=123,
            fine_config=cfg,
            spatial_aux_weight=1.0,
            training_objective="on_policy_distill",
        )
        student.set_teachers({"L_XL": copy.deepcopy(teacher.model)})
        torch.manual_seed(7)
        res = run_cascade_episode(
            env,
            [c2],
            student,
            6,
            organic_objective,
            training=True,
            rollout_steps=999,
            reward_profile="dense_potential",
        )
        self.assertEqual(len(res["stageSteps"]), 2)
        self.assertGreater(len(student.tape), 0)
        self.assertTrue(
            any(row.get("teacher_logits") is not None for row in student.tape)
        )
        self.assertTrue(
            any(row.get("teacher_mean") is not None for row in student.tape)
        )
        # Verify dense potential step rewards are populated and non-constant.
        step_rews = [
            float(row["reward"])
            for row in student.tape[:-1]
            if row.get("reward") is not None
        ]
        self.assertGreater(len(step_rews), 0)
        self.assertGreater(max(step_rews) - min(step_rews), 0.0)
        stats = student.finish(0.0)
        self.assertTrue(math.isfinite(stats["loss"]))
        self.assertGreater(stats["gradientNorm"], 0.0)

    def test_learned_cap_varies_edit_size_and_allows_early_coarse_finish(self):
        env, cfg = self._make_env(shape=(12, 32, 32), plan="rectangle")
        cenv = CoarseEnv(env, block=(1, 2, 2))
        cenv.reset(8)
        learner = CoarseToFineLearner(
            device="cpu",
            seed=123,
            fine_config=cfg,
            decoder_retry_schedule="learned_cap",
            evaluation_noise_scale=0.0,
        )
        learner.decoder_diagnostics = DecoderDiagnostics()
        base_logits = torch.tensor([10.0, -10.0, -10.0, -10.0])

        def _fake_forward_neg(obs_b, *_args, **_kwargs):
            b, _z, _c, h, w = obs_b.shape
            mean = torch.full((b, 2, obs_b.shape[1], h, w), -5.0)
            val = torch.zeros(b)
            logits = base_logits.unsqueeze(0).expand(b, -1).clone()
            return mean, val, logits

        learner.model.forward_stage_batch = _fake_forward_neg
        learner.choose_stage(cenv, is_coarse=True, training=False, retries=1)
        low_req = learner.decoder_diagnostics.attempts[-1].requested_voxels

        def _fake_forward_pos(obs_b, *_args, **_kwargs):
            b, _z, _c, h, w = obs_b.shape
            mean = torch.full((b, 2, obs_b.shape[1], h, w), 5.0)
            val = torch.zeros(b)
            logits = base_logits.unsqueeze(0).expand(b, -1).clone()
            return mean, val, logits

        learner.model.forward_stage_batch = _fake_forward_pos
        learner.choose_stage(cenv, is_coarse=True, training=False, retries=1)
        high_req = learner.decoder_diagnostics.attempts[-1].requested_voxels
        self.assertLess(low_req, high_req)

        # Verify early coarse "finish" is legal after 2 edits once near target FAR.
        cenv.reset(4)
        cenv.count_fine = int(round(3.6 * cenv.plot_area))
        cenv.edits = 2
        self.assertFalse(cenv.near_far)
        finish_logits = torch.tensor([-10.0, -10.0, -10.0, 10.0])

        def _fake_forward_finish(obs_b, *_args, **_kwargs):
            b, _z, _c, h, w = obs_b.shape
            mean = torch.zeros((b, 2, obs_b.shape[1], h, w))
            val = torch.zeros(b)
            logits = finish_logits.unsqueeze(0).expand(b, -1).clone()
            return mean, val, logits

        learner.model.forward_stage_batch = _fake_forward_finish
        for is_train in (True, False):
            mv = learner.choose_stage(cenv, is_coarse=True, training=is_train)
            self.assertIsNone(mv)
            self.assertEqual(learner.last_operation, "finish")

        # Verify "exchange" is gated when |far - target_far| > 0.75, and allowed within 0.75.
        exchange_logits = torch.tensor([-5.0, -5.0, 10.0, -5.0])

        def _fake_forward_exchange(obs_b, *_args, **_kwargs):
            b, _z, _c, h, w = obs_b.shape
            mean = torch.zeros((b, 2, obs_b.shape[1], h, w))
            val = torch.zeros(b)
            logits = exchange_logits.unsqueeze(0).expand(b, -1).clone()
            return mean, val, logits

        learner.model.forward_stage_batch = _fake_forward_exchange
        cenv.reset(8)
        learner.choose_stage(cenv, is_coarse=True, training=False, retries=1)
        self.assertEqual(learner.last_operation, "remove")

        cenv.reset(4)
        cenv.count_fine = int(round(3.6 * cenv.plot_area))
        learner.choose_stage(cenv, is_coarse=True, training=False, retries=1)
        self.assertEqual(learner.last_operation, "exchange")

    def test_evaluation_temperature_scaling_and_deterministic_argmax(self):
        env, cfg = self._make_env(shape=(8, 16, 16))
        env.unified_limit = 12
        env.unified_actions = 0
        env.organic_3d_growth = True
        env.floor_copy_growth = False
        env._ln6_cache = None
        learner = CoarseToFineLearner("cpu", 17, fine_config=cfg, input_channels=12)
        flat = UnifiedLearner("cpu", 17, config=cfg, organic_3d=True, net_guided=True)
        fixed_logits = torch.tensor([[3.0, 1.0, 0.5, -4.0]])

        def fake_c2f(obs_b, goal_b, is_coarse=False, stride=(1, 1, 1), scale=1):
            b = obs_b.shape[0]
            h, w = max(1, obs_b.shape[-2] // 4), max(1, obs_b.shape[-1] // 4)
            z = max(1, obs_b.shape[1] // 2)
            return (
                torch.zeros((b, 2, z, h, w)),
                torch.zeros((b,)),
                fixed_logits.expand(b, -1),
            )

        with patch.object(learner.model, "forward_stage_batch", side_effect=fake_c2f):
            learner.evaluation_temperature = 0.0
            for s in range(5):
                torch.manual_seed(100 + s)
                learner.choose_stage(env, is_coarse=False, training=False)
                self.assertEqual(learner.last_operation, "remove")

            seen_probs = {}
            from torch.distributions import Categorical as RealCat

            for temp in (0.5, 2.0):
                captured = {}

                def capture_cat(*, logits):
                    captured["logits"] = logits.detach().clone()
                    return RealCat(logits=logits)

                learner.evaluation_temperature = temp
                with patch(
                    "voxelform.coarse_to_fine.Categorical", side_effect=capture_cat
                ):
                    learner.choose_stage(env, is_coarse=False, training=False)
                probs = torch.softmax(captured["logits"], dim=-1)
                seen_probs[temp] = float(probs[0])
            self.assertGreater(seen_probs[0.5], seen_probs[2.0])

        def fake_flat(obs, goal, stride=(2, 4, 4)):
            z = max(1, math.ceil(obs.shape[0] / stride[0]))
            h = max(1, math.ceil(obs.shape[-2] / stride[1]))
            w = max(1, math.ceil(obs.shape[-1] / stride[2]))
            return (
                torch.zeros((2, z, h, w)),
                torch.zeros(()),
                fixed_logits[0],
            )

        with patch.object(flat.model, "forward_coarse", side_effect=fake_flat):
            flat.evaluation_temperature = 0.0
            for s in range(5):
                torch.manual_seed(200 + s)
                flat.choose(env, training=False)
                self.assertEqual(flat.last_operation, "remove")

    def test_fair_greedy_learned_cap_uses_quota_and_early_handoff(self):
        env, cfg = self._make_env(shape=(10, 48, 48))
        env.unified_limit = 12
        env.unified_actions = 0
        env.organic_3d_growth = True
        env.floor_copy_growth = False
        env._ln6_cache = None
        ce = CoarseEnv(env, (1, 2, 2))
        ce.reset(6)
        # Put coarse FAR within (0.45, 0.70] of target_far=3.0 so near_far=False yet early handoff is legal.
        occ = ce.occupied.clone()
        occ[3:6] = False
        occ[3, :14, :] = True  # roughly 3.0 + 0.58 FAR
        ce.restore_occupancy(occ)
        ce.edits = 2
        self.assertFalse(ce.near_far)
        self.assertLessEqual(abs(ce.far - ce.target_far), 0.70)

        fair_g = CoarseToFineLearner(
            "cpu",
            11,
            fine_config=cfg,
            input_channels=14,
            heuristic_only=True,
            decoder_retry_schedule="learned_cap",
            greedy_learned_cap=True,
        )
        legacy_g = CoarseToFineLearner(
            "cpu",
            11,
            fine_config=cfg,
            input_channels=14,
            heuristic_only=True,
            decoder_retry_schedule="learned_cap",
            greedy_learned_cap=False,
        )
        self.assertIsNone(fair_g.choose_stage(ce, is_coarse=True, training=False))
        self.assertEqual(fair_g.last_operation, "finish")

        legacy_g.choose_stage(ce, is_coarse=True, training=False)
        self.assertNotEqual(legacy_g.last_operation, "finish")

        with patch(
            "voxelform.coarse_to_fine._learned_quota", return_value=2
        ) as mock_quota:
            fair_g.choose_stage(env, is_coarse=False, training=False)
            self.assertGreaterEqual(mock_quota.call_count, 1)

    def test_spatial_aux_regression_retry_mean_entropy_and_shaping_gamma(self):
        env, cfg = self._make_env(shape=(8, 16, 16))
        env.unified_limit = 12
        env.unified_actions = 0
        env.organic_3d_growth = True
        env.floor_copy_growth = False
        env._ln6_cache = None
        learner = CoarseToFineLearner(
            "cpu",
            29,
            fine_config=cfg,
            input_channels=12,
            spatial_aux_weight=1.0,
            spatial_aux_mode="regression",
            retry_logp_norm="mean",
            continuous_entropy_coef=0.1,
        )
        learner.choose_stage(env, is_coarse=False, training=True)
        self.assertEqual(len(learner.tape), 1)
        entry = learner.tape[0]
        if entry["elig_c"] is not None:
            self.assertIsNotNone(entry["des_target"])
            self.assertIsNone(entry["local_adv"])
            with torch.no_grad():
                mean_b, _, _ = learner.model.forward_stage_batch(
                    entry["obs"][None],
                    entry["goal"][None],
                    is_coarse=entry["is_coarse"],
                    stride=entry["stride"],
                    scale=entry["scale"],
                )
            consumed = entry["drawn_channels"]
            loss_exact = learner._masked_spatial_loss(
                mean_b[0, consumed], mean_b[0, consumed].detach(), entry["elig_c"]
            )
            self.assertAlmostEqual(float(loss_exact.item()), 0.0, places=6)

        learner.record(0.5)
        learner.mark_done()
        learner.finish(0.0)
        # Positive continuous_entropy_coef encourages larger patch_log_std (grad < 0 before optimizer step).
        learner2 = CoarseToFineLearner(
            "cpu",
            31,
            fine_config=cfg,
            input_channels=12,
            spatial_aux_weight=0.0,
            retry_logp_norm="mean",
            continuous_entropy_coef=0.5,
        )
        learner2.choose_stage(env, is_coarse=False, training=True)
        obs0 = learner2.tape[0]["obs"]
        goal0 = learner2.tape[0]["goal"]
        draw0 = learner2.tape[0]["latents"][0]
        base_entry = dict(
            obs=obs0,
            goal=goal0,
            is_coarse=False,
            scale=1,
            stride=learner2.fine_config.stride,
            op=torch.tensor(0),
            legal=torch.tensor([True, True, True, False]),
            value=torch.tensor(0.0),
            latent=draw0,
            latents=[draw0],
            drawn_channels=[0],
            consumed=[0],
            elig_c=None,
            local_adv=None,
            des_target=None,
            teacher_mean=None,
            teacher_logits=None,
            heuristic_op=None,
            aux_weight=0.0,
            reward=torch.tensor(1.0),
            done=True,
        )
        learner2.tape = [
            dict(base_entry),
            dict(base_entry, latents=[draw0, draw0, draw0, draw0]),
        ]
        stats = learner2.finish(0.0)
        self.assertTrue(math.isfinite(stats["loss"]))
        self.assertIsNotNone(learner2.model.patch_log_std.grad)
        self.assertLess(float(learner2.model.patch_log_std.grad[0].item()), 0.0)

        env_s, cfg_s = self._make_env(shape=(10, 24, 24))
        pol_s = CoarseToFineLearner(
            "cpu", 33, fine_config=cfg_s, input_channels=12, spatial_aux_weight=0.0
        )
        res = run_cascade_episode(
            env_s,
            (),
            pol_s,
            4,
            organic_objective,
            training=True,
            rollout_steps=32,
            audit_steps=1,
            step_reward_fn=organic_step_reward,
            terminal_reward_fn=organic_terminal_reward,
            reward_profile="dense_potential",
            potential_shaping_gamma=0.99,
        )
        self.assertIn("rawObjective", res)


class BundledTeachersAndServerImportTests(unittest.TestCase):
    def _organic_session(self, shape=(8, 12, 16), initial_storeys=4):
        import unified_server as server
        from voxelform.environment import Environment
        from voxelform.patch_policy import PatchConfig, PatchFacades
        from voxelform.unified import UnifiedLearner

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
            shape,
            sun_count=4,
            device="cpu",
            far_tolerance=0.1,
            frontier_factory=functools.partial(
                PatchFacades, config=cfg, depth_device="cpu"
            ),
        )
        solid = env.allowed.clone()
        solid[initial_storeys:] = False
        env.restore_occupancy(solid)
        env.termination = None
        env.unified_limit = 12
        env.unified_actions = 0
        env.precompute_external()
        env.organic_3d_growth = True
        env.floor_copy_growth = False
        policy = UnifiedLearner(
            "cpu",
            123,
            cfg,
            amount_feedback=True,
            net_guided=True,
            floor_copy=False,
            organic_3d=True,
        )
        session = server.UnifiedSession(
            device="cpu",
            autostart=False,
            initial_storeys=initial_storeys,
            ceiling_storeys=shape[0],
        )
        session.env, session.policy, session.config = env, policy, cfg
        session.last_objective, session.last_diag = organic_objective(env)
        from voxelform.objectives import fine_potential

        session.last_fine_potential = fine_potential(
            env, session.last_objective, env.start_count
        )
        session.ready = True
        return server, session, cfg

    def test_distill_config_loads_bundled_teachers(self):
        from pathlib import Path
        from tools.research import make_policy
        from voxelform.experiment import ExperimentConfig

        cfg = ExperimentConfig.read(Path("configs/research-distill-all-tiers.json"))
        learner = make_policy(cfg, torch.device("cpu"), 4211)
        self.assertIsInstance(learner, CoarseToFineLearner)
        self.assertEqual(set(learner.teachers.keys()), {"XS_M", "L_XL"})
        for net in learner.teachers.values():
            params = list(net.parameters())
            self.assertGreater(len(params), 0)
            self.assertTrue(all(torch.isfinite(p).all() for p in params))

    def test_server_imports_research_multiscale_checkpoint(self):
        import io
        from fastapi.testclient import TestClient

        server, session, cfg = self._organic_session()
        research_learner = CoarseToFineLearner(
            "cpu",
            77,
            fine_config=cfg,
            input_channels=14,
            decoder_retry_schedule="learned_cap",
        )
        state = research_learner.state()
        state["updates"] = 42
        payload_buf = io.BytesIO()
        torch.save({"format": "voxelform.research.v2", "policy": state}, payload_buf)

        with patch.object(server, "session", session):
            client = TestClient(server.app)
            resp = client.post("/api/policy", content=payload_buf.getvalue())
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertTrue(resp.json()["ok"])
            self.assertEqual(session.policy.updates, 42)
            self.assertTrue(session.policy.organic_3d)
            self.assertTrue(session.policy.net_guided)
            self.assertEqual(session.policy.decoder_retry_schedule, "learned_cap")
            session.env.configure_far(session.env.far + 1.5, 0.1)
            self.assertFalse(bool(session.policy.legal_operations(session.env)[2]))
            session.env.configure_far(session.env.far, 0.1)
            self.assertTrue(bool(session.policy.legal_operations(session.env)[2]))
            expected = research_learner.model.state_dict()
            for key, tensor in session.policy.model.state_dict().items():
                target = expected[key]
                if (
                    target.ndim == 2
                    and tensor.ndim == 2
                    and target.shape[1] == tensor.shape[1] + 1
                ):
                    target = target[:, : tensor.shape[1]]
                torch.testing.assert_close(tensor, target)

            bad_state = dict(state, inputChannels=12)
            bad_buf = io.BytesIO()
            torch.save(
                {"format": "voxelform.research.v2", "policy": bad_state}, bad_buf
            )
            bad_resp = client.post("/api/policy", content=bad_buf.getvalue())
            self.assertEqual(bad_resp.status_code, 400)

    def test_server_runs_coarse_to_fine_stage_on_large_parcel(self):
        import io
        from fastapi.testclient import TestClient

        server, session, cfg = self._organic_session(
            shape=(8, 52, 52), initial_storeys=6
        )
        self.assertGreaterEqual(session.env.plot_area, 2500.0)
        research_learner = CoarseToFineLearner(
            "cpu",
            79,
            fine_config=cfg,
            input_channels=14,
            decoder_retry_schedule="learned_cap",
        )
        state = research_learner.state()
        payload_buf = io.BytesIO()
        torch.save({"format": "voxelform.research.v2", "policy": state}, payload_buf)

        with patch.object(server, "session", session):
            client = TestClient(server.app)
            resp = client.post("/api/policy", content=payload_buf.getvalue())
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertIsNotNone(session.c2f_policy)
            self.assertIsNotNone(session.coarse_env)
            self.assertTrue(session.in_coarse_stage)
            self.assertEqual(session.coarse_env.block, (1, 2, 2))
            self.assertTrue(session.state()["coarseStageActive"])

            # Step through the 2x2x1 coarse stage until handoff (or cap steps).
            for _ in range(10):
                if not session.in_coarse_stage:
                    break
                session.single = True
                session.advance_one()
            self.assertGreaterEqual(session.coarse_env.unified_actions, 1)
            coarse_sources = [
                row["maskSource"]
                for row in session.edit_trace
                if row["maskSource"].startswith("coarse_2x2x1_")
            ]
            self.assertTrue(
                bool(coarse_sources) or not session.in_coarse_stage,
                "Expected coarse_2x2x1_ edits or clean handoff to fine stage",
            )
            # Force handoff if still in coarse stage and confirm fine stage executes cleanly.
            if session.in_coarse_stage:
                session.coarse_env.unified_actions = session.coarse_env.unified_limit
            before_fine_actions = session.env.unified_actions
            session.single = True
            session.advance_one()
            self.assertFalse(session.in_coarse_stage)
            self.assertEqual(session.env.unified_actions, before_fine_actions + 1)

    def test_server_audit_environment_applies_fine_potential_shaping(self):
        from voxelform.objectives import fine_potential

        _server, session, _cfg = self._organic_session()
        session.training = True
        session.policy.tape.append({"reward": torch.tensor(0.0)})
        prev_pot = session.last_fine_potential
        # Trigger an occupancy edit so fine_potential shifts on audit.
        occ = session.env.occupied.clone()
        occ[3, :2, :2] = False
        session.env.restore_occupancy(occ)
        session.audit_environment()
        cur_pot = fine_potential(
            session.env, session.last_objective, session.env.start_count
        )
        self.assertAlmostEqual(session.last_fine_potential, cur_pot, places=6)
        self.assertAlmostEqual(
            float(session.policy.tape[-1]["reward"]),
            2.0 * (cur_pot - prev_pot),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
