import unittest
import numpy as np
import torch
from voxelform.fields import facade_depth, depth_badness
from voxelform.actions import ExteriorFacades, FacadeMaskAction
from voxelform.environment import Environment


class FacadeDepth(unittest.TestCase):
    def test_metric_centres_not_roof_and_exact_remapping(self):
        occupied = np.ones((1, 41, 41), bool)
        d = facade_depth(occupied)
        self.assertEqual(float(d[0, 20, 20]), 20.5)
        self.assertEqual(float(d[0, 20, 0]), 0.5)
        np.testing.assert_allclose(
            depth_badness(np.array([0, 8, 12, 15, 16, 35])), [0, 0, 0.5, 0.875, 1, 1]
        )
        torch.testing.assert_close(
            depth_badness(torch.tensor([8.0, 12.0, 16.0])),
            torch.tensor([0.0, 0.5, 1.0]),
        )

    def test_sealed_void_does_not_reduce_depth_but_open_courtyard_does(self):
        solid = np.ones((3, 41, 41), bool)
        solid[1, 15:26, 15:26] = False
        before = facade_depth(solid)
        self.assertEqual(float(before[1, 20, 11]), 11.5)
        solid[2, 15:26, 15:26] = False
        after = facade_depth(solid)
        self.assertEqual(float(after[1, 20, 11]), 3.5)
        self.assertGreater(float(depth_badness(before)[1, 20, 11]), 0)
        self.assertEqual(float(depth_badness(after)[1, 20, 11]), 0)

    def test_distances_match_independent_continuous_segment_reference(self):
        o = np.ones((2, 11, 15), bool)
        o[:, 5:, 8:] = False
        o[1, 2:4, 0:6] = False
        f = ExteriorFacades(o, np.ones_like(o))
        d = facade_depth(o, f.exterior_air)
        for z in range(2):
            segments = []
            for y, x in np.argwhere(o[z]):
                for side, (dy, dx) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
                    yy, xx = y + dy, x + dx
                    if (
                        not (0 <= yy < o.shape[1] and 0 <= xx < o.shape[2])
                        or f.exterior_air[z, yy, xx]
                    ):
                        if side < 2:
                            segments.append(
                                ((x + (side == 1), y), (x + (side == 1), y + 1))
                            )
                        else:
                            segments.append(
                                ((x, y + (side == 3)), (x + 1, y + (side == 3)))
                            )
            a = np.array([s[0] for s in segments])
            b = np.array([s[1] for s in segments])
            v = b - a
            for y, x in np.argwhere(o[z]):
                p = np.array([x + 0.5, y + 0.5])
                t = np.clip(((p - a) * v).sum(1) / (v * v).sum(1), 0, 1)
                expected = np.sqrt(((p - (a + t[:, None] * v)) ** 2).sum(1)).min()
                self.assertAlmostEqual(float(d[z, y, x]), float(expected), places=5)

    def test_dynamic_depth_is_fresh_before_environment_audit(self):
        e = Environment((4, 24, 32), sun_count=4)
        old = e.depth.clone()
        mask = np.zeros(e.shape, bool)
        mask[:, 1:-1, 0] = True
        action = FacadeMaskAction(mask, -1, e.frontier.version, int(mask.sum()))
        e.apply(action, audit=False)
        self.assertEqual(e.depth_edit, e.edits)
        self.assertNotEqual(e.last_audit_edit, e.edits)
        self.assertLess(float(e.depth[1, 12, 10]), float(old[1, 12, 10]))
        self.assertEqual(e.observation(True).shape[1], 6)

    def test_depth_objective_is_explicit_and_default_quality_is_preserved(self):
        e = Environment((4, 24, 32), sun_count=4)
        base = float(e.quality)
        e.depth_weight = 0.1
        e.evaluate()
        self.assertAlmostEqual(
            float(e.quality),
            base - 0.1 * float(e.depth_badness.sum()) / e.capacity,
            places=6,
        )

    def test_pointwise_depth_gate_cannot_select_deep_voxels_on_facade(self):
        o = np.ones((3, 41, 41), bool)
        f = ExteriorFacades(o, o)
        d = facade_depth(o, f.exterior_air)
        bad = depth_badness(d)
        self.assertTrue((d[f.remove_skin] == 0.5).all())
        self.assertTrue((bad[f.remove_skin] == 0).all())
        self.assertTrue((bad > 0).any())
        self.assertFalse((f.remove_skin * bad).any())
        np.testing.assert_array_equal(f.remove_skin * (1 - bad), f.remove_skin)

    def test_local_actor_cannot_see_bad_depth_beyond_its_receptive_field(self):
        from voxelform.policy import Network

        torch.manual_seed(7)
        o = np.ones((3, 41, 41), bool)
        f = ExteriorFacades(o, o)
        observation = torch.rand(3, 6, 41, 41)
        observation[:, 5] = torch.tensor(depth_badness(facade_depth(o)))
        empty = observation.clone()
        empty[:, 5] = 0
        model = Network(input_channels=6)
        goal = torch.tensor([1.0, 0.75, 0.25])
        with torch.no_grad():
            a, _ = model(observation, goal)
            b, _ = model(empty, goal)
        legal = torch.tensor(f.remove_skin)
        torch.testing.assert_close(a[:, legal], b[:, legal], rtol=0, atol=0)

    def test_empty_cells_and_metric_scaling(self):
        o = np.ones((2, 9, 9), bool)
        o[:, 4:, 4:] = False
        d = facade_depth(o)
        self.assertTrue((d[~o] == 0).all())
        np.testing.assert_allclose(facade_depth(o, pitch=2), 2 * d)


class TensorFacadeDepth(unittest.TestCase):
    def test_tensor_matches_cpu_reference_on_irregular_and_sealed_geometry(self):
        from voxelform.fields import facade_depth_tensor
        from voxelform.actions import exterior_air

        rng = np.random.default_rng(77)
        for _ in range(8):
            o = rng.random((4, 9, 13)) > 0.3
            ext = exterior_air(o)
            a = facade_depth_tensor(torch.tensor(o), torch.tensor(ext)).numpy()
            b = facade_depth(o, ext)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_roof_opening_on_other_floor_does_not_shorten_depth(self):
        from voxelform.fields import facade_depth_tensor
        from voxelform.actions import exterior_air

        o = np.ones((4, 41, 41), bool)
        o[-1, 15:26, 15:26] = 0
        d = facade_depth_tensor(torch.tensor(o), torch.tensor(exterior_air(o)))
        self.assertEqual(float(d[0, 20, 20]), 20.5)
        self.assertEqual(float(d[2, 20, 20]), 20.5)
        self.assertEqual(float(d[-1, 20, 11]), 3.5)

    def test_tensor_frontier_never_calls_cpu_depth_and_reuses_device_field(self):
        import functools
        from unittest.mock import patch
        from voxelform.patch_policy import PatchFacades, PatchConfig

        cfg = PatchConfig(eligibility="depth")
        with (
            patch(
                "voxelform.patch_policy.facade_depth",
                side_effect=AssertionError("CPU depth called"),
            ),
            patch(
                "voxelform.environment.facade_depth",
                side_effect=AssertionError("CPU depth called"),
            ),
        ):
            e = Environment(
                (4, 16, 20),
                sun_count=4,
                frontier_factory=functools.partial(
                    PatchFacades, config=cfg, depth_device="cpu"
                ),
            )
            self.assertIs(e.depth, e.frontier.depth_tensor)
            self.assertIsNone(e.frontier.depth_numpy)
            w = e.frontier.weights_tensor(-1, "cpu")
            self.assertEqual(tuple(w.shape), e.shape)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_depth_matches_same_floor_cpu_reference(self):
        from voxelform.fields import facade_depth_tensor
        from voxelform.actions import exterior_air

        o = np.ones((4, 19, 23), bool)
        o[:, 8:12, 8:13] = 0
        o[1, 2:5, 2:5] = 0
        air = exterior_air(o)
        result = facade_depth_tensor(
            torch.tensor(o, device="mps"), torch.tensor(air, device="mps")
        )
        self.assertEqual(result.device.type, "mps")
        np.testing.assert_allclose(
            result.cpu().numpy(), facade_depth(o, air), rtol=0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
