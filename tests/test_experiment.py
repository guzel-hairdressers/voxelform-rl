import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tools.research import aggregate, summarize
from tests.test_unified import fixture
from voxelform.experiment import (
    ExperimentConfig,
    read_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
    write_json,
)
from voxelform.unified import UnifiedLearner


class Experiment(unittest.TestCase):
    def test_configuration_rejects_data_leakage_and_invalid_values(self):
        for kwargs in [
            dict(test_sites=("synthetic:rectangle",)),
            dict(seeds=(9001,)),
            dict(cores_enabled="false"),
            dict(initial_storeys=9, ceiling_storeys=8),
            dict(target_far=float("nan")),
            dict(seeds=(123, 123)),
        ]:
            with self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text('{"episodez": 2}')
            with self.assertRaisesRegex(ValueError, "Unknown"):
                ExperimentConfig.read(p)

    def test_rng_and_safe_checkpoint_roundtrip(self):
        c = ExperimentConfig()
        seed_everything(4)
        state = rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(state)
        self.assertEqual(expected[:2], (random.random(), np.random.rand()))
        torch.testing.assert_close(expected[2], torch.rand(3))
        p = UnifiedLearner(
            "cpu", 4, organic_3d=True, net_guided=True, spatial_aux_weight=0.0
        )
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "c.pt"
            save_checkpoint(f, p, c, 4, 2, {"scene": "hash"})
            loaded = read_checkpoint(f, c)
            self.assertEqual(loaded["nextEpisode"], 2)
            q = UnifiedLearner("cpu", 99)
            q.load(loaded["policy"])
            self.assertEqual(q.spatial_aux_weight, 0)
            for key, value in p.model.state_dict().items():
                torch.testing.assert_close(value, q.model.state_dict()[key])
            with self.assertRaises(ValueError):
                read_checkpoint(f, ExperimentConfig(episodes=2))
            p.tape.append({})
            with self.assertRaises(ValueError):
                save_checkpoint(f, p, c, 4, 2, {})

    def test_nonfinite_metrics_are_not_written(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            write_json(p, {"value": 1})
            with self.assertRaises(ValueError):
                write_json(p, {"value": float("nan")})
            self.assertEqual(json.loads(p.read_text()), {"value": 1})

    def test_failed_mask_retries_and_step_rewards_reach_actor_loss(self):
        e, _ = fixture()
        p = UnifiedLearner(
            "cpu", 123, organic_3d=True, net_guided=True, spatial_aux_weight=0.0
        )
        with patch("voxelform.unified.project_organic_3d", return_value=None):
            p.choose(e, True, retries=3)
        self.assertEqual(len(p.tape[0]["latents"]), 3)
        self.assertIsNone(p.tape[0]["latent"])
        p.record(0.75)
        self.assertEqual(float(p.tape[0]["reward"]), 0.75)
        self.assertNotIn(3, p.tape[0])
        p.mark_done()
        stats = p.finish(0.0)
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertGreater(float(p.model.head[-1].weight.grad.abs().sum()), 0.0)

    def test_variable_scene_shapes_can_share_a_rollout(self):
        c = ExperimentConfig()
        from tools.research import Scenes, make_policy, episode

        scenes = Scenes(c)
        p = make_policy(c, "cpu", 123)
        e = scenes.get("synthetic:rectangle")
        episode(p, e, c, True)
        # Different XY shape, not merely a different number of floors.
        from dataclasses import replace

        other = Scenes(replace(c, synthetic_width=20, synthetic_length=28))
        e2 = other.get("synthetic:rectangle")
        episode(p, e2, c, True)
        if p.tape:
            self.assertGreater(p.finish(0.0)["gradientNorm"], 0)

    def test_failures_included_and_uncertainty_uses_model_seeds(self):
        def row(ok, q):
            return dict(
                success=ok,
                quality=q,
                seconds=1.0,
                reference={"quality": q},
                diagnostics={"usableGoodArea": 1.0},
                environment={"directSunProxy": 0.5, "diffuseFacadeProxy": 0.2},
            )

        s = summarize([row(True, 2), row(False, 0)])
        self.assertEqual(s["farSuccessRate"], 0.5)
        self.assertEqual(s["meanQuality"], 1.0)
        a = aggregate([dict(test=s), dict(test=s)])["test"]
        self.assertEqual(a["modelSeeds"], 2)
        self.assertIsNone(a["metrics"]["meanQuality"]["bootstrap95AcrossModelSeeds"])
        self.assertEqual(a["metrics"]["meanQuality"]["tCi95AcrossModelSeeds"], [1.0, 1.0])
        a1 = aggregate([dict(test=s)])["test"]
        self.assertIsNone(a1["metrics"]["meanQuality"]["tCi95AcrossModelSeeds"])

    def test_periodic_validation_checkpointing_selects_best_checkpoint(self):
        from types import SimpleNamespace
        from tools.research import train

        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "cfg.json"
            out_dir = Path(d) / "run"
            cfg_path.write_text(
                json.dumps(
                    dict(
                        seeds=[123],
                        train_sites=["synthetic:rectangle"],
                        validation_sites=["synthetic:courtyard"],
                        test_sites=["synthetic:branch"],
                        evaluation_seeds=[9001, 9002],
                        episodes=2,
                        initial_storeys=4,
                        ceiling_storeys=8,
                        synthetic_width=16,
                        synthetic_length=20,
                        checkpoint_interval=1,
                        strict_determinism=True,
                    )
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                train(
                    SimpleNamespace(
                        config=str(cfg_path),
                        output=str(out_dir),
                        resume=None,
                        stop_after=None,
                        only_seed=None,
                        finalize_only=False,
                    )
                )
            seed_dir = out_dir / "seed-123"
            for fname in (
                "checkpoint-000001.pt",
                "checkpoint-000002.pt",
                "best.pt",
                "latest.pt",
                "validation-history.jsonl",
                "validation.json",
                "test.json",
                "validation-final.json",
                "test-final.json",
                "summary.json",
            ):
                self.assertTrue((seed_dir / fname).is_file(), fname)
            val_hist = [
                json.loads(line)
                for line in (seed_dir / "validation-history.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            self.assertEqual([r["episode"] for r in val_hist], [1, 2])
            seed_summary = json.loads((seed_dir / "summary.json").read_text())
            best_ckpt = read_checkpoint(seed_dir / "best.pt")
            self.assertEqual(
                seed_summary["bestValidationEpisode"], best_ckpt["nextEpisode"]
            )
            best_row = max(
                val_hist,
                key=lambda r: (
                    round(float(r["farSuccessRate"]), 6),
                    float(r["meanQuality"]),
                    -int(r["episode"]),
                ),
            )
            self.assertEqual(seed_summary["bestValidationEpisode"], best_row["episode"])
            self.assertAlmostEqual(
                seed_summary["validation"]["meanQuality"],
                best_row["meanQuality"],
                places=7,
            )
            self.assertAlmostEqual(
                seed_summary["finalValidation"]["meanQuality"],
                val_hist[-1]["meanQuality"],
                places=7,
            )
            run_summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(
                run_summary["aggregate"]["bestValidationEpisodes"],
                [seed_summary["bestValidationEpisode"]],
            )
            self.assertIn("finalTest", run_summary["aggregate"])

    def test_checkpoint_cadence_and_exact_resume_preserve_optimizer_updates(self):
        from types import SimpleNamespace
        from tools.research import train

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = dict(
                seeds=[123],
                evaluation_seeds=[9001],
                episodes=3,
                initial_storeys=4,
                ceiling_storeys=8,
                synthetic_width=16,
                synthetic_length=20,
                multiscale_schedule="tier_adaptive",
                rollout_steps=1000,
                strict_determinism=True,
            )

            def run(name, interval, resume=None, stop_after=None):
                config_path = root / f"interval-{interval}.json"
                write_json(config_path, dict(cfg, checkpoint_interval=interval))
                with contextlib.redirect_stdout(io.StringIO()):
                    train(
                        SimpleNamespace(
                            config=config_path,
                            output=root / name,
                            resume=resume,
                            stop_after=stop_after,
                        )
                    )
                return root / name / "seed-123"

            scheduled = run("scheduled", 1)
            uninterrupted = run("uninterrupted", 2)
            stopped = run("stopped", 2, stop_after=1)
            interruption = read_checkpoint(stopped / "latest.pt")
            self.assertGreater(len(interruption["pendingRollout"]), 0)
            self.assertEqual(interruption["policy"]["updates"], 0)
            self.assertEqual((stopped / "validation-history.jsonl").read_text(), "")
            periodic = read_checkpoint(scheduled / "checkpoint-000001.pt")
            self.assertEqual(periodic["policy"]["updates"], 0)
            self.assertEqual(
                len(periodic["pendingRollout"]), len(interruption["pendingRollout"])
            )
            resumed = run("resumed", 2, resume=stopped / "latest.pt")

            expected = read_checkpoint(uninterrupted / "latest.pt")["policy"]
            for folder in (scheduled, resumed):
                with self.subTest(run=folder.parent.name):
                    actual = read_checkpoint(folder / "latest.pt")["policy"]
                    self.assertEqual(actual["updates"], expected["updates"])
                    self.assertEqual(actual["updates"], 1)
                    for key in ("model", "optimizer"):
                        torch.testing.assert_close(
                            actual[key], expected[key], rtol=0, atol=0
                        )

    def test_run_manifest_freezes_seed_roster(self):
        from types import SimpleNamespace
        from tools.research import train
        from voxelform.experiment import runtime_manifest

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            c = ExperimentConfig(seeds=(123, 456))
            write_json(out / "config.json", c.to_dict())
            write_json(
                out / "manifest.json",
                dict(runtime_manifest(), configHash=c.fingerprint),
            )
            reduced = out / "reduced.json"
            write_json(reduced, dict(c.to_dict(), seeds=[123]))
            args = SimpleNamespace(output=out, resume=None, finalize_only=True)
            with self.assertRaisesRegex(ValueError, "roster.*fixed"):
                train(SimpleNamespace(**vars(args), config=reduced))
            with self.assertRaisesRegex(FileNotFoundError, "Missing seed summary"):
                train(SimpleNamespace(**vars(args), config=out / "config.json"))
            self.assertFalse((out / "summary.json").exists())

    def test_distillation_config_fingerprint_annealing_and_teacher_loading(self):
        from dataclasses import replace
        from tools.research import make_policy, spatial_aux_weight_at
        from voxelform.coarse_to_fine import CoarseToFineLearner

        base = ExperimentConfig()
        self.assertNotIn("teacher_checkpoints", base.to_dict())
        alt = replace(base, teacher_checkpoints=())
        self.assertEqual(base.fingerprint, alt.fingerprint)

        sched = ExperimentConfig(
            episodes=100,
            multiscale_schedule="tier_macro4",
            spatial_aux_schedule="annealed_floor",
            spatial_aux_weight=1.0,
        )
        self.assertAlmostEqual(spatial_aux_weight_at(sched, 0), 1.0)
        self.assertAlmostEqual(spatial_aux_weight_at(sched, 24), 1.0)
        self.assertAlmostEqual(spatial_aux_weight_at(sched, 99), 0.15)

        with tempfile.TemporaryDirectory() as d:
            ckpt_path = Path(d) / "teacher.pt"
            t_cfg = ExperimentConfig(multiscale_schedule="tier_adaptive")
            teacher = CoarseToFineLearner("cpu", 77, input_channels=14)
            save_checkpoint(ckpt_path, teacher, t_cfg, 77, 1, {})

            s_cfg = ExperimentConfig(
                multiscale_schedule="tier_macro4",
                training_objective="on_policy_distill",
                spatial_aux_schedule="annealed_floor",
                spatial_aux_weight=1.2,
                reward_profile="dense_potential",
                teacher_checkpoints=(f"L_XL={ckpt_path}",),
            )
            self.assertIn("teacher_checkpoints", s_cfg.to_dict())
            student = make_policy(s_cfg, "cpu", 123)
            self.assertIn("L_XL", student.teachers)
            for key, val in teacher.model.state_dict().items():
                torch.testing.assert_close(
                    val, student.teachers["L_XL"].state_dict()[key]
                )

    def test_opt_in_research_flags_preserve_default_fingerprint_and_teacher_provenance(self):
        from dataclasses import replace

        base = ExperimentConfig()
        explicit_defaults = ExperimentConfig(
            spatial_aux_mode="reinforce",
            retry_logp_norm="sum",
            potential_shaping_gamma=None,
            continuous_entropy_coef=0.0,
        )
        self.assertEqual(base.fingerprint, explicit_defaults.fingerprint)

        zero_temp = replace(base, evaluation_temperature=0.0)
        self.assertEqual(zero_temp.evaluation_temperature, 0.0)
        with self.assertRaises(ValueError):
            replace(base, evaluation_temperature=-0.1)

        opt = replace(
            base,
            multiscale_schedule="tier_meso2",
            spatial_aux_mode="regression",
            retry_logp_norm="mean",
            potential_shaping_gamma=0.99,
            continuous_entropy_coef=0.01,
        )
        d = opt.to_dict()
        self.assertEqual(d["spatial_aux_mode"], "regression")
        self.assertEqual(d["retry_logp_norm"], "mean")
        self.assertEqual(d["potential_shaping_gamma"], 0.99)
        self.assertEqual(d["continuous_entropy_coef"], 0.01)

        for tname in ("teacher_xs_m_seed789.pt", "teacher_l_xl_ep100.pt"):
            tpath = Path(__file__).resolve().parents[1] / "data" / "teachers" / tname
            ckpt = torch.load(tpath, map_location="cpu", weights_only=True)
            self.assertIn("provenance", ckpt)
            self.assertIn("trainSites", ckpt["provenance"])
            self.assertIn("modelSHA256", ckpt["provenance"])
