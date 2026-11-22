import functools, unittest
import numpy as np
import torch
from tests.test_unified import fixture
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.unified import (
    UnifiedLearner,
    commit,
    valid_move,
    copy_plates,
    supported_removal,
    project,
    clean_floor_numpy,
    clean_floor_tensor,
)
from voxelform.redistribution import Exchange


class FloorCopy(unittest.TestCase):
    def policy(self, device="cpu"):
        e, p = fixture(device)
        p.floor_copy = True
        p.net_guided = True
        p.amount_feedback = True
        return e, p

    def test_complete_plate_can_grow_on_first_edit_and_is_linked_to_source(self):
        e, p = self.policy()
        before = e.frontier.occupied.copy()
        move = p.choose(e, False, force_operation="grow")
        self.assertIsNotNone(move)
        np.testing.assert_array_equal(move.add[4], before[3])
        self.assertFalse(move.add[:4].any())
        commit(e, move)
        self.assertTrue((e.copy_labels[3][before[3]] > 0).all())
        np.testing.assert_array_equal(e.copy_labels[3], e.copy_labels[4])
        # A later top-only selection is pruned; selecting both linked cells survives.
        r = np.zeros(e.shape, bool)
        r[4, 0, :4] = True
        self.assertFalse(supported_removal(e, r).any())
        r[3, 0, :4] = True
        np.testing.assert_array_equal(supported_removal(e, r), r)

    def test_courtyard_is_preserved_by_copy(self):
        e, p = self.policy()
        solid = e.frontier.occupied.copy()
        solid[:4, 4:8, 6:10] = False
        e.restore_occupancy(solid)
        e.termination = None
        move = p.choose(e, False, force_operation="grow")
        self.assertIsNotNone(move)
        np.testing.assert_array_equal(move.add[4], solid[3])
        self.assertFalse(move.add[:, 4:8, 6:10].any())
        commit(e, move)
        self.assertFalse(e.frontier.occupied[:, 4:8, 6:10].any())

    def test_auto_exchange_grows_and_reduces_far_without_random_roof_bits(self):
        e, p = self.policy()
        torch.manual_seed(123)
        with torch.no_grad():
            p.model.operation_head.bias[2] = 20
        before = e.far_error
        move = p.choose(e, False)
        self.assertIsNotNone(move)
        self.assertTrue(move.remove.any() and move.add.any())
        np.testing.assert_array_equal(
            move.add[4], e.frontier.occupied[3] & ~move.remove[3]
        )
        self.assertFalse((move.remove & ~e.frontier.eligible(-1)).any())
        commit(e, move)
        self.assertLess(e.far_error, before)
        self.assertFalse((e.frontier.occupied[1:] & ~e.frontier.occupied[:-1]).any())

    def test_no_floating_upper_cells_or_fragmented_copy(self):
        e, p = self.policy()
        p.choose(e, False, force_operation="grow")  # selects new semantics
        r = np.zeros(e.shape, bool)
        a = r.copy()
        r[0, 0, :4] = True
        self.assertFalse(valid_move(e, Exchange(r, a, e.frontier.version)))
        r[:] = False
        a[4, 0, :4] = True
        self.assertFalse(valid_move(e, Exchange(r, a, e.frontier.version)))
        e.frontier.occupied[:] = True
        e.frontier.refresh()
        self.assertEqual(copy_plates(e), [])

    def test_checkpoint_preserves_new_decoder_and_old_defaults(self):
        e, p = self.policy()
        q = type(p)()
        q.load(p.state())
        self.assertTrue(q.floor_copy and q.net_guided)
        old = p.state()
        del old["floorCopy"]
        q.load(old)
        self.assertFalse(q.floor_copy)

    def test_linked_cut_remains_subset_of_neural_selection(self):
        e, p = self.policy()
        move = p.choose(e, False, force_operation="grow")
        commit(e, move)
        masks = np.zeros((2, *e.shape), bool)
        masks[0, 3:5, 0, :8] = True
        result = project(e, masks, np.ones_like(masks, float), "remove", True, False)
        self.assertIsNotNone(result)
        self.assertFalse((result.remove & ~masks[0]).any())
        np.testing.assert_array_equal(result.remove[3], result.remove[4])
        commit(e, result)

    def test_copy_exchange_rebalances_the_extra_loss_from_cutting_its_source(self):
        e, p = self.policy()
        e.configure_far(4.5)
        e.termination = None
        e.floor_copy_growth = True
        masks = np.zeros((2, *e.shape), bool)
        masks[0] = e.frontier.eligible(-1)
        masks[1] = copy_plates(e)[0]
        scores = np.broadcast_to(
            np.arange(e.nz)[None, :, None, None], masks.shape
        ).astype(float)
        move = project(e, masks, scores, "exchange", True, True)
        self.assertIsNotNone(move)
        self.assertFalse((move.remove & ~masks[0]).any())
        np.testing.assert_array_equal(
            move.add[4], e.frontier.occupied[3] & ~move.remove[3]
        )
        before = e.count
        commit(e, move)
        self.assertGreater(e.count, before)
        self.assertTrue(e.far_satisfied)

    def test_small_edit_cap_never_fragments_a_floor_to_make_it_fit(self):
        from dataclasses import replace

        e, p = self.policy()
        p.config = replace(p.config, maximum_edit_voxels=16)
        e.frontier.config = p.config
        self.assertEqual(copy_plates(e), [])
        self.assertFalse(p.legal_operations(e)[2])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_native_mps_copy_then_linked_backward(self):
        e, p = self.policy("mps")
        torch.manual_seed(123)
        # First on-policy exchange, followed by another pass through linked scores.
        with torch.no_grad():
            p.model.operation_head.bias[2] = 20
        m = p.choose(e, True)
        self.assertIsNotNone(m)
        commit(e, m)
        p.record(0.2)
        m = p.choose(e, True)
        if m:
            commit(e, m)
        p.record(0.1)
        self.assertGreater(p.finish(0.0)["gradientNorm"], 0)

    def test_whole_top_floor_can_be_unstacked_reversibly(self):
        e, p = self.policy()
        e.floor_copy_growth = True
        move = p.choose(e, False, force_operation="grow")
        self.assertIsNotNone(move)
        commit(e, move)
        self.assertTrue((e.copy_labels[4] > 0).all())
        masks = np.zeros((2, *e.shape), bool)
        masks[0, 4] = e.frontier.occupied[4]
        np.testing.assert_array_equal(supported_removal(e, masks[0]), masks[0])
        result = project(e, masks, np.ones_like(masks, float), "remove", True, False)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result.remove[4], e.frontier.occupied[4])
        self.assertFalse(result.remove[:4].any())
        commit(e, result)
        self.assertFalse(e.frontier.occupied[4].any())
        self.assertTrue(e.frontier.occupied[3].all())
        self.assertFalse((e.copy_labels[3] > 0).any())
        r_floor3 = np.zeros((2, *e.shape), bool)
        r_floor3[0, 3, 0, :4] = True
        cut3 = project(
            e, r_floor3, np.ones_like(r_floor3, float), "remove", True, False
        )
        self.assertIsNotNone(cut3)
        commit(e, cut3)

    def test_stepped_podium_and_tower_wings_both_expose_copy_plates(self):
        e, p = self.policy()
        solid = np.zeros(e.shape, bool)
        solid[:2, :, :10] = True
        solid[:4, :, 10:] = True
        e.restore_occupancy(solid)
        e.termination = None
        plates = copy_plates(e)
        levels = sorted(int(np.flatnonzero(pl.any((1, 2)))[0]) for pl in plates)
        self.assertEqual(levels, [2, 4])

    def test_sliver_top_plate_rejected_on_urban_scale_plot(self):
        c = PatchConfig(
            mode="combined",
            allocation="feedback",
            facade_layers=2,
            minimum_patch=4,
            maximum_patches=4,
        )
        e = Environment(
            (8, 20, 20),
            sun_count=4,
            device="cpu",
            far_tolerance=0.1,
            frontier_factory=functools.partial(
                PatchFacades, config=c, depth_device="cpu"
            ),
        )
        solid = np.zeros(e.shape, bool)
        solid[:4] = True
        # Thin 1-cell L fin with 24 cells (below 2x2 core threshold) must not copy to floor 5.
        solid[4, 0, :20] = True
        solid[4, 1:5, 0] = True
        e.restore_occupancy(solid)
        e.termination = None
        plates = copy_plates(e)
        self.assertFalse(any(pl[5].any() for pl in plates))
        self.assertTrue(any(pl[4].any() for pl in plates))

    def test_urban_scale_carving_rejects_slim_appendages_and_hollow_walls(self):
        c = PatchConfig(
            mode="combined",
            allocation="feedback",
            facade_layers=2,
            minimum_patch=4,
            maximum_patches=4,
        )
        e = Environment(
            (8, 20, 20),
            sun_count=4,
            device="cpu",
            far_tolerance=0.1,
            frontier_factory=functools.partial(
                PatchFacades, config=c, depth_device="cpu"
            ),
        )
        solid = np.zeros(e.shape, bool)
        solid[:4, 2:18, 2:18] = True
        e.restore_occupancy(solid)
        e.termination = None
        e.floor_copy_growth = True
        r = np.zeros(e.shape, bool)
        r[:4, 2:16, 2:18] = True
        pruned = supported_removal(e, r)
        after = solid & ~pruned
        self.assertFalse(
            ((after & ~clean_floor_numpy(after, 3)) & clean_floor_numpy(solid, 3)).any()
        )
        room = torch.zeros((1, 16, 16), dtype=torch.bool)
        room[0, 2:8, 2:8] = True
        room[0, 2:4, 8:15] = True
        for mw in (3, 4):
            cleaned = clean_floor_tensor(room, mw)
            self.assertTrue(cleaned[0, 2:8, 2:8].all())
            self.assertFalse(cleaned[0, 2:4, 8:15].any())

    def test_upper_floors_can_step_back_and_peel_outward_to_inward(self):
        c = PatchConfig(
            mode="combined",
            allocation="feedback",
            facade_layers=2,
            minimum_patch=4,
            maximum_patches=4,
        )
        e = Environment(
            (8, 20, 20),
            sun_count=4,
            device="cpu",
            far_tolerance=0.1,
            frontier_factory=functools.partial(
                PatchFacades, config=c, depth_device="cpu"
            ),
        )
        solid = np.zeros(e.shape, bool)
        solid[:6, 2:18, 2:18] = True
        e.restore_occupancy(solid)
        e.termination = None
        e.configure_far(2.5)
        p = UnifiedLearner(
            "cpu", 123, c, amount_feedback=True, net_guided=True, floor_copy=True
        )
        move = p.choose(e, False, force_operation="remove")
        self.assertIsNotNone(move)
        after = e.frontier.occupied & ~move.remove
        self.assertFalse(
            (
                (after & ~clean_floor_numpy(after, 3))
                & clean_floor_numpy(e.frontier.occupied, 3)
            ).any()
        )


if __name__ == "__main__":
    unittest.main()
