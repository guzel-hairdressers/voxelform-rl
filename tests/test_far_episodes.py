import unittest
import numpy as np
import torch
from torch import nn

from voxelform.actions import ExteriorFacades, FacadeMaskAction
from voxelform.context import prepare_scene
from voxelform.environment import Environment
from voxelform.policy import Learner, Network
from voxelform.session import Session, DENSITY_PRESETS
from test_context import fixture

torch.set_num_threads(1)


def action(frontier, mask, operation=-1):
    return FacadeMaskAction(mask, operation, frontier.version, int(mask.sum()))


class FixedFields(nn.Module):
    """Independent oracle: direct network fields with no proposal/ranking input."""

    def __init__(self, fields):
        super().__init__()
        self.output = fields
        self.calls = 0

    def forward(self, observation, goal):
        self.calls += 1
        return self.output, torch.tensor(0.0)


class DirectNeuralMasks(unittest.TestCase):
    def test_network_outputs_dense_add_and_subtract_fields(self):
        e = Environment((4, 8, 10), sun_count=4)
        model = Network()
        fields, value = model(e.observation(), torch.tensor(e.goal_features))
        self.assertEqual(tuple(fields.shape), (2, 4, 8, 10))
        self.assertEqual(value.ndim, 0)
        self.assertTrue(bool(torch.isfinite(fields).all()))

    def test_exact_nonrectangular_mask_comes_from_model_pixels(self):
        e = Environment((4, 8, 10), sun_count=4)
        p = Learner()
        patterns = [
            [(0, 1, 0), (0, 3, 0), (1, 1, 0), (1, 3, 0)],
            [(2, 2, 0), (2, 4, 0), (3, 2, 0)],
        ]
        for points in patterns:
            fields = torch.full((2, *e.shape), -20.0)
            desired = np.zeros(e.shape, bool)
            for point in points:
                fields[(0, *point)] = 20
                desired[point] = True
            p.model = FixedFields(fields)
            generated = p.choose(e, training=False, greedy=True)
            self.assertIsNotNone(generated)
            np.testing.assert_array_equal(generated.mask, desired)
            self.assertEqual(p.model.calls, 1)

    def test_model_interior_logits_cannot_modify_hidden_cells(self):
        e = Environment((4, 8, 10), sun_count=4)
        p = Learner()
        fields = torch.full((2, *e.shape), -20.0)
        fields[0, 1:3, 2:5, 3:6] = 30  # high-confidence interior output is illegal
        fields[0, 0:2, 2, 0] = 20
        p.model = FixedFields(fields)
        generated = p.choose(e, training=False, greedy=True)
        self.assertIsNotNone(generated)
        self.assertFalse((generated.mask & ~e.frontier.remove_skin).any())
        before = e.occupied.clone()
        e.apply(generated, audit=False)
        self.assertTrue(torch.equal(before[:, 2:5, 3:6], e.occupied[:, 2:5, 3:6]))

    def test_addition_uses_the_separate_neural_output_channel(self):
        e = Environment((2, 4, 4), sun_count=4, target_far=1)
        occ = torch.zeros_like(e.allowed)
        occ[0] = True
        e.restore_occupancy(occ)
        e.configure_far(1.5)
        fields = torch.full((2, *e.shape), -20.0)
        fields[1, 1, 0, 0:2] = 20
        fields[0, 0] = 30
        p = Learner()
        p.model = FixedFields(fields)
        generated = p.choose(e, training=False, greedy=True)
        self.assertIsNotNone(generated)
        self.assertEqual(generated.operation, 1)
        self.assertEqual(generated.count, 2)
        self.assertTrue(generated.mask[1, 0, 0:2].all())
        before = e.count
        e.apply(generated)
        self.assertEqual(e.count, before + 2)

    def test_direct_head_receives_actor_gradients_and_updates(self):
        e = Environment((4, 12, 16), sun_count=4)
        p = Learner(seed=9)
        before = {
            k: v.detach().clone() for k, v in p.model.mask_head.state_dict().items()
        }
        for _ in range(4):
            generated = p.choose(e, training=True)
            self.assertIsNotNone(generated)
            error = e.far_error
            e.apply(generated, audit=False)
            p.record(error - e.far_error)
        stats = p.finish(p.bootstrap(e))
        self.assertGreater(stats["gradientNorm"], 0)
        self.assertEqual(stats["directMaskUpdates"], 1)
        self.assertTrue(
            any(
                not torch.equal(v, p.model.mask_head.state_dict()[k])
                for k, v in before.items()
            )
        )


class FacadeConstraints(unittest.TestCase):
    def test_mask_cannot_drill_into_a_newly_exposed_layer_in_one_action(self):
        occupied = np.ones((4, 6, 6), bool)
        f = ExteriorFacades(occupied, np.ones_like(occupied))
        outer = np.zeros_like(occupied)
        outer[:, 1:5, 0] = True
        inner = np.zeros_like(occupied)
        inner[:, 1:5, 1] = True
        projected = f.project_mask(outer | inner, np.ones(f.shape), -1, 100)
        self.assertIsNotNone(projected)
        np.testing.assert_array_equal(projected.mask, outer)
        f.apply(projected)
        self.assertTrue(f.occupied[:, 1:5, 1].all())
        self.assertTrue(f.remove_skin[:, 1:5, 1].all())
        with self.assertRaisesRegex(ValueError, "stale"):
            f.apply(projected)

    def test_sealed_air_is_not_facade_until_a_later_step_opens_it(self):
        occupied = np.ones((5, 5, 5), bool)
        occupied[2, 2, 2] = False
        f = ExteriorFacades(occupied, np.ones_like(occupied))
        outer = np.zeros_like(occupied)
        outer[1:4, 2, 0] = True
        inner = np.zeros_like(occupied)
        inner[1:4, 2, 1] = True
        self.assertFalse(f.validate_mask(inner, -1))
        f.apply(action(f, outer))
        self.assertFalse(f.exterior_air[2, 2, 2])
        f.apply(action(f, inner))
        self.assertTrue(f.exterior_air[2, 2, 2])

    def test_projection_only_prunes_unsafe_selected_bits(self):
        occupied = np.zeros((2, 5, 9), bool)
        occupied[:, 1:4, :3] = True
        occupied[:, 1:4, 6:] = True
        occupied[:, 2, 3:6] = True
        f = ExteriorFacades(occupied, np.ones_like(occupied))
        raw = np.zeros_like(occupied)
        raw[:, 2, 3:6] = True
        raw[:, 1, 1] = True
        raw[:, 1, 7] = True
        self.assertFalse(f.validate_mask(raw, -1))
        projected = f.project_mask(raw, np.ones(f.shape), -1, 100)
        self.assertIsNotNone(projected)
        self.assertFalse((projected.mask & ~raw).any())
        self.assertLess(projected.count, int(raw.sum()))
        self.assertTrue(f.validate_mask(projected.mask, -1))

    def test_added_mask_cannot_seal_an_air_cavity(self):
        occupied = np.ones((5, 5, 5), bool)
        occupied[1:, 1:3, 2] = False
        f = ExteriorFacades(occupied, np.ones_like(occupied))
        cap = np.zeros_like(occupied)
        cap[4, 1:3, 2] = True
        self.assertTrue(f.add_skin[cap].all())
        self.assertFalse(f.validate_mask(cap, 1))
        projected = f.project_mask(cap, np.ones(f.shape), 1, 100)
        self.assertIsNone(projected)

    def test_last_ground_anchors_are_protected(self):
        occupied = np.zeros((3, 3, 3), bool)
        occupied[:, :, 1] = True
        f = ExteriorFacades(occupied, np.ones_like(occupied))
        mask = np.zeros_like(occupied)
        mask[0, :, 1] = True
        self.assertFalse(f.validate_mask(mask, -1))

    def test_far_clipping_does_not_overshoot(self):
        e = Environment((4, 4, 4), sun_count=4, target_far=3.9375)
        p = Learner()
        generated = p.choose(e, training=False, greedy=True)
        self.assertIsNotNone(generated)
        self.assertEqual(generated.count, 1)
        self.assertTrue(e.apply(generated))
        self.assertEqual(e.count, e.target_count)
        with self.assertRaises(ValueError):
            e.apply(generated)


class FarEpisodes(unittest.TestCase):
    def session(self, target=3, shape=(4, 8, 10)):
        return Session(
            env=Environment(shape, sun_count=4, target_far=target), autostart=False
        )

    def test_optimizer_rollout_is_not_a_building_episode(self):
        s = self.session(1.5, (4, 24, 32))
        s.start(training=True)
        for _ in range(32):
            s.advance_one()
        self.assertEqual(s.episodes, 0)
        self.assertFalse(s.env.terminal)
        self.assertEqual(s.stats["directMaskUpdates"], 1)
        self.assertTrue(s.env.metrics()["environmentalFieldsCurrent"])
        s.advance_one()
        self.assertFalse(s.env.metrics()["environmentalFieldsCurrent"])
        s.pause()

    def test_generated_masks_continue_until_far_without_inference_updates(self):
        s = self.session()
        before = {k: v.clone() for k, v in s.policy.model.state_dict().items()}
        s.start(training=False)
        for _ in range(100):
            s.advance_one()
            if not s.running:
                break
        self.assertEqual(s.env.far, 3)
        self.assertEqual(s.env.voxels_edited, 80)
        self.assertGreater(s.steps, 1)
        self.assertFalse(s.running)
        self.assertEqual(s.env.termination, "far_reached")
        self.assertTrue(
            all(t["maskSource"] == "neural_per_voxel" for t in s.edit_trace)
        )
        for k, value in before.items():
            self.assertTrue(torch.equal(value, s.policy.model.state_dict()[k]))

    def test_raising_far_uses_direct_neural_addition_masks(self):
        s = self.session()
        s.start(training=False)
        while s.running:
            s.advance_one()
        occupied = s.env.occupied.clone()
        s.configure_far(3.5)
        self.assertTrue(torch.equal(occupied, s.env.occupied))
        s.start(training=False)
        for _ in range(100):
            s.advance_one()
            if not s.running:
                break
        self.assertEqual(s.env.far, 3.5)
        self.assertEqual(s.env.voxels_edited, 40)
        self.assertTrue(all(t["operation"] == "add" for t in s.edit_trace))

    def test_pause_keeps_pending_rollout_and_cancels_pending_step(self):
        s = self.session(1.5, (4, 24, 32))
        s.start()
        for _ in range(5):
            s.advance_one()
        s.pause()
        count = s.env.count
        tape = len(s.policy.tape)
        clock = s.clock()
        s.advance_one()
        self.assertEqual(s.env.count, count)
        self.assertEqual(len(s.policy.tape), tape)
        self.assertEqual(s.clock(), clock)
        s.start()
        s.advance_one()
        self.assertEqual(len(s.policy.tape), tape + 1)
        count = s.env.count
        s.start(training=False, single=True)
        s.pause()
        s.advance_one()
        self.assertEqual(s.env.count, count)

    def test_invalid_far_and_presets(self):
        s = self.session()
        self.assertEqual(s.state()["settings"]["densityPreset"], "medium")
        self.assertEqual(DENSITY_PRESETS, dict(low=1.5, medium=3.0, high=4.5))
        s.start()
        for value in (0, -1, True, float("nan"), float("inf"), 4.1):
            with self.assertRaises(ValueError):
                s.configure_far(value)
            self.assertEqual(s.target_far, 3)
            self.assertTrue(s.running)
        s.configure_far(1.5, "low")
        self.assertEqual(s.env.target_count, 120)
        with self.assertRaises(ValueError):
            s.configure_far(2, "medium")

    def test_real_parcel_area_and_nearest_cell_far(self):
        raw = fixture()
        raw["siteArea"] = 48.3
        scene = prepare_scene(raw, dict(site_id="test"), floors=4)
        e = Environment(context=scene, sun_count=4, target_far=3)
        p = Learner()
        for _ in range(100):
            if e.terminal:
                break
            generated = p.choose(e, training=False, greedy=True)
            self.assertIsNotNone(generated)
            e.apply(generated, audit=False)
        self.assertEqual(e.count, 145)
        self.assertLessEqual(
            e.metrics()["farError"], e.metrics()["farTolerance"] + 1e-12
        )

    def test_rejected_generation_is_failed_not_a_false_far_success(self):
        s = self.session()
        s.env.frontier.project_mask = lambda *_: None
        s.start()
        s.advance_one()
        self.assertFalse(s.running)
        self.assertFalse(s.recent[-1]["success"])
        self.assertEqual(s.env.termination, "mask_generation_failed")
        self.assertEqual(s.env.edits, 0)
        self.assertEqual(s.stats["directMaskUpdates"], 1)

    def test_already_reached_goal_cannot_spin_zero_edit_episodes(self):
        s = self.session(4)
        s.start()
        s.advance_one()
        self.assertFalse(s.running)
        self.assertEqual(s.episodes, 1)
        s.advance_one()
        self.assertEqual(s.episodes, 1)


class CheckpointMigration(unittest.TestCase):
    def test_old_mask_ranker_keeps_encoder_critic_but_gets_a_new_dense_head(self):
        p = Learner()
        old = {
            k: v.clone()
            for k, v in p.model.state_dict().items()
            if k.startswith(("encoder.", "critic."))
        }
        old["actor.0.weight"] = torch.ones((64, 84))
        p.load(
            dict(
                format="voxelform.policy.v4",
                actionSpace="current_facade_mask_far_v1",
                model=old,
                updates=176,
            )
        )
        self.assertTrue(torch.equal(p.model.encoder[0].weight, old["encoder.0.weight"]))
        self.assertTrue(torch.equal(p.model.critic[0].weight, old["critic.0.weight"]))
        self.assertEqual(p.direct_updates, 0)
        self.assertEqual(p.imported_updates, 176)
        self.assertEqual(p.state()["actionSpace"], "direct_facade_mask_far_v1")
        copy = Learner()
        copy.load(p.state())
        for key, value in p.model.state_dict().items():
            self.assertTrue(torch.equal(value, copy.model.state_dict()[key]))

    def test_invalid_direct_checkpoint_is_transactional(self):
        p = Learner()
        before = {k: v.clone() for k, v in p.model.state_dict().items()}
        state = p.state()
        state["model"] = dict(state["model"])
        state["model"]["mask_head.0.weight"] = torch.zeros((1, 1))
        with self.assertRaises(RuntimeError):
            p.load(state)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, p.model.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
