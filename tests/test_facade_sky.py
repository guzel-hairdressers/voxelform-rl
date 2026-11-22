import unittest
import numpy as np
import torch
from voxelform.fields import sky_boundary, diffuse_access, facade_sky_access
from voxelform.actions import exterior_air, neighbors_horizontal
from voxelform.environment import Environment

torch.set_num_threads(1)


class FacadeSky(unittest.TestCase):
    def test_empty_context_has_equal_open_facades_but_solid_air_values_are_zero(self):
        solid = torch.ones((5, 16, 20), dtype=torch.bool)
        boundary = sky_boundary(solid.shape, [])
        air = diffuse_access(solid, boundary)
        faces, mean = facade_sky_access(solid, air, boundary)
        self.assertEqual(float(air.max()), 0.0)
        self.assertEqual(float(faces[0, :, :, -1].mean()), 1.0)
        self.assertEqual(float(faces[1, :, :, 0].mean()), 1.0)
        self.assertEqual(float(faces[2, :, -1, :].mean()), 1.0)
        self.assertEqual(float(faces[3, :, 0, :].mean()), 1.0)
        self.assertEqual(
            float(mean[:, 3:-3, 3:-3].max()), 0.0
        )  # Interior is not a facade.

    def test_nearby_wall_reduces_diffuse_access_on_its_side(self):
        solid = torch.ones((5, 16, 20), dtype=torch.bool)
        wall = np.array([[10, -3, 12, 18, 1, 12, 1, 0]], np.float32)
        boundary = sky_boundary(solid.shape, wall, samples=32)
        air = diffuse_access(solid, boundary)
        faces, _ = facade_sky_access(solid, air, boundary)
        north = float(faces[3, :, 0, 1:-1].mean())
        south = float(faces[2, :, -1, 1:-1].mean())
        self.assertLess(north, south - 0.2)
        self.assertLess(north, 0.8)
        self.assertGreater(south, 0.8)

    def test_open_well_facade_is_darker_at_the_bottom(self):
        solid = torch.ones((8, 25, 25), dtype=torch.bool)
        solid[:, 10:15, 10:15] = False
        boundary = sky_boundary(solid.shape, [])
        air = diffuse_access(solid, boundary, iterations=160)
        faces, _ = facade_sky_access(solid, air, boundary)
        self.assertGreater(float(faces[0, -1, 12, 9]), float(faces[0, 0, 12, 9]) + 0.1)

    def test_diffuse_facade_input_is_independent_of_direct_sun_sampling(self):
        a = Environment((4, 12, 16), sun_count=4)
        b = Environment((4, 12, 16), sun_count=8)
        torch.testing.assert_close(a.facade_light, b.facade_light, rtol=0, atol=0)
        torch.testing.assert_close(a.sky_boundary, b.sky_boundary, rtol=0, atol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_gpu_sampling_matches_cpu_on_irregular_facades(self):
        solid = torch.ones((5, 14, 18), dtype=torch.bool)
        solid[:, 5:9, 6:10] = False
        solid[-1, 8:, 10:] = False
        boundary = sky_boundary(solid.shape, [])
        air = diffuse_access(solid, boundary)
        raw = solid.numpy()
        mask = torch.as_tensor(
            (raw[None] & neighbors_horizontal(exterior_air(raw), outside=True))[
                [1, 0, 3, 2]
            ]
        )
        expected = facade_sky_access(solid, air, boundary, mask)
        actual = facade_sky_access(
            solid.to("mps"), air.to("mps"), boundary.to("mps"), mask.to("mps")
        )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.cpu(), b, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
