import math, unittest
import numpy as np
import torch
from voxelform.shadows import (
    DirectShadows,
    BoundaryShadows,
    DepthShadows,
    SelfShadowMaps,
    ReceiverCache,
    dda_paths,
    sun_path,
    internal_indices,
)
from voxelform.fields import noise_prior, diffuse_access, receivers, usable_depth_proxy
from voxelform.environment import Environment
from voxelform.policy import Learner

torch.set_num_threads(1)


def reference_hit(point, direction, boxes):
    # Separate scalar reference, including rotated local coordinates and slabs.
    for b in boxes:
        x, y, z = np.asarray(point) - b[:3]
        c, s = b[6:8]
        origin = (c * x + s * y, -s * x + c * y, z)
        dx, dy, dz = direction
        direction_local = (c * dx + s * dy, -s * dx + c * dy, dz)
        enter, leave = 0.0, float("inf")
        for p, d, h in zip(origin, direction_local, b[3:6]):
            if abs(d) < 1e-10:
                if abs(p) > h:
                    leave = -1
                    break
            else:
                t1, t2 = sorted(((-h - p) / d, (h - p) / d))
                enter = max(enter, t1)
                leave = min(leave, t2)
        if leave > max(enter, 1e-5):
            return True
    return False


class Shadows(unittest.TestCase):
    def test_rotated_parallel_inside_and_missed_rays(self):
        a = 0.61
        boxes = np.array(
            [[3, 2, 4, 2, 1, 4, math.cos(a), math.sin(a)], [-3, 1, 2, 1, 2, 2, 1, 0]],
            np.float32,
        )
        suns = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, -0.4, 0.5]], float)
        suns /= np.linalg.norm(suns, axis=1)[:, None]
        points = np.random.default_rng(42).uniform([-7, -7, 0], [7, 7, 9], (200, 3))
        observed = (
            DirectShadows(boxes, suns)
            .query(torch.tensor(points, dtype=torch.float32))
            .numpy()
        )
        expected = np.array(
            [[not reference_hit(p, s, boxes) for p in points] for s in suns]
        )
        np.testing.assert_array_equal(observed, expected)

    def test_axis_maps_match_reference(self):
        boxes = np.array([[5, 5, 15, 3, 3, 2, 1, 0]], np.float32)
        suns = np.array([[0.0, 0.0, 1.0]])
        points = torch.tensor(
            [[x + 0.5, y + 0.5, 3.0] for x in range(10) for y in range(10)]
        )
        expected = DirectShadows(boxes, suns).query(points)
        for cls in (BoundaryShadows, DepthShadows):
            actual = cls(boxes, suns, ((0, 0, 0), (10, 10, 10)), pitch=1).query(points)
            self.assertTrue(torch.equal(actual, expected))

    def test_face_union_is_direction_dependent(self):
        ordinary = BoundaryShadows([], sun_path(), ((0, 0, 0), (10, 10, 12)))
        rotated = BoundaryShadows([], sun_path(grid_yaw=45), ((0, 0, 0), (10, 10, 12)))
        self.assertEqual(len(ordinary.face_union), 4)
        self.assertEqual(len(rotated.face_union), 5)

    def test_boundary_intrusion_fallback(self):
        boxes = np.array([[5, 5, 5, 1, 1, 2, 1, 0]], np.float32)
        s = np.array([[0.0, 0.0, 1.0]])
        p = torch.tensor([[5.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertEqual(internal_indices(boxes, ((0, 0, 0), (10, 10, 10))), [0])
        expected = DirectShadows(boxes, s).query(p)
        corrected = BoundaryShadows(boxes, s, ((0, 0, 0), (10, 10, 10)))
        raw = BoundaryShadows(boxes, s, ((0, 0, 0), (10, 10, 10)), fallback=False)
        self.assertTrue(torch.equal(expected, corrected.query(p)))
        self.assertFalse(torch.equal(expected, raw.query(p)))

    def test_cache_only_queries_new_ids(self):
        method = DirectShadows([], [[0, 0, 1]])
        cache = ReceiverCache(method, 20)
        p = torch.zeros((2, 3))
        a, n = cache.query(torch.tensor([1, 3]), p)
        self.assertEqual(n, 2)
        a, n = cache.query(torch.tensor([3, 7]), p)
        self.assertEqual(n, 1)
        a, n = cache.query(torch.tensor([1, 7]), p)
        self.assertEqual(n, 0)

    def test_unchanged_receiver_must_lose_old_self_shadow(self):
        shape = (3, 4, 4)
        s = np.array([[0.0, 0.0, 1.0]])
        occ = torch.zeros(shape, dtype=torch.bool)
        occ[0, 1, 1] = True
        occ[2, 1, 1] = True
        m = SelfShadowMaps(shape, s, pitch=0.5)
        p = torch.tensor([[1.5, 1.5, 3.0]])
        n = torch.tensor([[0.0, 0.0, 1.0]])
        m.update(occ)
        self.assertFalse(bool(m.query(p, n)[0, 0]))
        occ[2, 1, 1] = False
        m.update(occ)
        self.assertTrue(bool(m.query(p, n)[0, 0]))

    def test_dda_axis_and_diagonal(self):
        ids, t = dda_paths([[0.5, 0.5, 9]], [[0, 0, -1]], (3, 2, 2))
        valid = ids[0][ids[0] < 12]
        np.testing.assert_array_equal(valid, [8, 4, 0])
        ids, t = dda_paths([[0, 0, 0]], [[1, 1, 0]], (1, 3, 3))
        valid = ids[0][ids[0] < 9]
        np.testing.assert_array_equal(valid, [0, 4, 8])


class FieldsAndLearning(unittest.TestCase):
    def test_noise_decay_and_intensity_addition(self):
        p = torch.tensor([[0.0, 10.0, 0.0], [0.0, 20.0, 0.0], [0.0, 40.0, 0.0]])
        line = ((-1e5, 0, 0), (1e5, 0, 0), 1.0)
        value = noise_prior(p, [line], soft_radius=0.01)
        self.assertAlmostEqual(float(value[0] / value[1]), 2.0, places=3)
        self.assertTrue(
            torch.allclose(noise_prior(p, [line, line], soft_radius=0.01), 2 * value)
        )
        short = ((-1.0, 0, 0), (1.0, 0, 0), 1.0)
        far = noise_prior(p, [short], soft_radius=0.01)
        self.assertGreater(float(far[1] / far[2]), 3.9)

    def test_wells_get_darker_deeper_and_brighter_wider(self):
        occ = torch.ones((8, 15, 15), dtype=torch.bool)
        occ[:, 6:9, 6:9] = False
        small = diffuse_access(occ, iterations=160)
        self.assertGreater(float(small[7, 7, 7]), float(small[0, 7, 7]))
        occ[:, 4:11, 4:11] = False
        large = diffuse_access(occ, iterations=160)
        self.assertGreater(float(large[2, 7, 7]), float(small[2, 7, 7]))

    def test_receiver_count_and_ground_omission(self):
        ids, p, n, cells, faces = receivers(torch.ones((2, 3, 4), dtype=torch.bool))
        self.assertEqual(len(ids), 2 * (3 + 4) * 2 + 3 * 4)
        self.assertFalse(bool((faces == 5).any()))

    def test_obstructed_envelope_not_assumed_lit(self):
        occupied = torch.ones((2, 8, 8), dtype=torch.bool)
        air = torch.zeros_like(occupied, dtype=torch.float32)
        dark_boundary = torch.zeros((4, 10, 10))
        self.assertEqual(
            int(usable_depth_proxy(occupied, air, boundary=dark_boundary).sum()), 0
        )
        dark_boundary[:, 0, :] = 1
        self.assertGreater(
            int(usable_depth_proxy(occupied, air, boundary=dark_boundary).sum()), 0
        )

    def test_facade_removal_and_addition_preserve_envelope(self):
        e = Environment((4, 12, 16), sun_count=4)
        original = e.occupied.clone()
        learner = Learner()
        action = learner.choose(e, training=False, greedy=True)
        self.assertIsNotNone(action)
        size = action.count
        e.apply(action)
        self.assertEqual(int((original != e.occupied).sum()), size)
        self.assertGreater(size, 1)
        e.configure_far(e.capacity / e.plot_area)
        for _ in range(100):
            if e.terminal:
                break
            action = learner.choose(e, training=False, greedy=True)
            self.assertIsNotNone(action)
            e.apply(action)
        self.assertTrue(torch.equal(original, e.occupied))
        self.assertFalse(bool((e.occupied & ~e.allowed).any()))

    def test_actor_backprop_and_checkpoint_roundtrip(self):
        torch.manual_seed(9)
        e = Environment((4, 12, 16), sun_count=4)
        learner = Learner(seed=9)
        before = [p.detach().clone() for p in learner.model.actor.parameters()]
        for _ in range(4):
            action = learner.choose(e)
            self.assertIsNotNone(action)
            q = e.quality.detach().clone()
            done = e.apply(action)
            learner.record(e.quality - q - 0.002)
            if done:
                break
        stats = learner.finish()
        self.assertGreater(stats["gradientNorm"], 0.0)
        self.assertTrue(
            any(
                not torch.equal(a, b)
                for a, b in zip(before, learner.model.actor.parameters())
            )
        )
        copied = Learner(seed=111)
        copied.load(learner.state())
        for a, b in zip(copied.model.parameters(), learner.model.parameters()):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
