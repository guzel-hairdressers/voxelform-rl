import json
from pathlib import Path
import random
import tempfile
import unittest
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
