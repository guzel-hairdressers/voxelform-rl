import copy, unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
from voxelform.context import prepare_scene, grid_orientation
from voxelform.environment import Environment
from voxelform.shadows import sun_path, MeshShadows
from test_context import fixture


def rotate(raw, angle, translation=(0, 0)):
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, s], [-s, c]])
    out = copy.deepcopy(raw)
    for mesh in [out["site"], *out["buildings"], *out["roads"]]:
        v = np.asarray(mesh["vertices"], float)
        v[:, :2] = v[:, :2] @ r + translation
        mesh["vertices"] = v.tolist()
    out["sitePerimeter"] = (np.asarray(out["sitePerimeter"]) @ r + translation).tolist()
    return out


class Alignment(unittest.TestCase):
    def test_area_principal_axes_and_long_edges(self):
        raw = rotate(fixture(), 0.37)
        for mode in ("principal_major", "longest_edge", "edge:0"):
            angle, _ = grid_orientation(raw, mode)
            self.assertAlmostEqual(angle, 0.37)
        angle, _ = grid_orientation(raw, "principal_minor")
        self.assertAlmostEqual(abs(angle - 0.37), np.pi / 2)
        translated = rotate(fixture(), 0.37, (5e7, -3e7))
        self.assertAlmostEqual(
            grid_orientation(translated, "principal_major")[0], 0.37, places=7
        )

    def test_all_meshes_and_perimeter_roundtrip_without_mutation(self):
        raw = rotate(fixture(), 0.37)
        before = copy.deepcopy(raw)
        e = prepare_scene(raw, {"site_id": "test"}, 4, "principal_major")
        a = e["gridAngle"]
        r = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        for source, mesh in [
            (raw["site"], e["site"]),
            *zip(raw["buildings"], e["buildings"]),
            *zip(raw["roads"], e["roads"]),
        ]:
            v = np.asarray(mesh["vertices"])
            world = v.copy()
            world[:, :2] = (v[:, :2] - e["offset"][:2]) @ r
            np.testing.assert_allclose(world, source["vertices"], atol=1e-12)
        self.assertEqual(raw, before)
        self.assertEqual(e["shape"], (4, 8, 10))
        self.assertEqual(e["allowed"].sum(), 48)

    def test_solar_directions_rotate_with_scene_not_relative_to_it(self):
        scene = prepare_scene(
            rotate(fixture(), 0.37), {"site_id": "test"}, 4, "principal_major"
        )
        e = Environment(context=scene, sun_count=4)
        a = scene["gridAngle"]
        r = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        world = e.suns_np.copy()
        world[:, :2] = world[:, :2] @ r
        np.testing.assert_allclose(world, sun_path(4), atol=1e-7)

    def test_external_rays_are_invariant_under_grid_transform(self):
        raw = rotate(fixture(), 0.37)
        scene = prepare_scene(raw, {"site_id": "test"}, 4, "principal_major")
        a = scene["gridAngle"]
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        points = (
            np.random.default_rng(9)
            .uniform([-8, -8, 0.1], [15, 15, 15], (100, 3))
            .astype("float32")
        )
        directions = sun_path(4).astype("float32")
        expected = MeshShadows(raw["buildings"], directions).query(torch.tensor(points))
        q = points.copy()
        q[:, :2] = q[:, :2] @ r + scene["offset"][:2]
        v = directions.copy()
        v[:, :2] = v[:, :2] @ r
        actual = MeshShadows(scene["buildings"], v).query(torch.tensor(q))
        self.assertTrue(torch.equal(actual, expected))

    def test_bad_alignment_rejected(self):
        for value in ("x", "edge:-1", "edge:500"):
            with self.assertRaises(ValueError):
                prepare_scene(fixture(), {"site_id": "test"}, 4, value)

    def test_auto_fits_full_squares_without_rotating_world_geometry(self):
        raw = rotate(fixture(), 0.37)
        scene = prepare_scene(raw, {"site_id": "test"}, 4, "auto_fit")
        self.assertEqual(scene["gridFit"]["fullSquares"], 48)
        self.assertEqual(scene["allowed"].sum(), 48)
        y, x = np.where(scene["allowed"])
        a = scene["gridAngle"]
        r = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        original_rotation = np.array(
            [[np.cos(0.37), -np.sin(0.37)], [np.sin(0.37), np.cos(0.37)]]
        )
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            world = (np.stack((x + dx, y + dy), 1) - scene["offset"][:2]) @ r
            unrotated = world @ original_rotation + np.array([2, 1])
            self.assertGreaterEqual(unrotated.min(), -1e-7)
            self.assertLessEqual(unrotated[:, 0].max(), 8 + 1e-7)
            self.assertLessEqual(unrotated[:, 1].max(), 6 + 1e-7)
        e = Environment(context=scene, sun_count=4)
        world = e.suns_np.copy()
        world[:, :2] = world[:, :2] @ r
        np.testing.assert_allclose(world, sun_path(4), atol=1e-7)

    def test_session_grid_reset_preserves_weights_and_autosave_orientation(self):
        from voxelform.session import Session

        class Library:
            def catalog(self):
                return [dict(site_id="test")]

            def load(self, site, floors=6, grid_alignment="world"):
                return prepare_scene(
                    rotate(fixture(), 0.37),
                    dict(site_id="test"),
                    floors,
                    grid_alignment,
                )

        library = Library()
        e = Environment(context=library.load("test", floors=4), sun_count=4)
        with tempfile.TemporaryDirectory() as directory:
            s = Session(library=library, env=e, autostart=False)
            s.persistence_root = Path(directory)
            weights = {k: v.clone() for k, v in s.policy.model.state_dict().items()}
            s.configure_grid("longest_edge")
            self.assertEqual(s.env.nz, 4)
            self.assertEqual(s.env.context["gridAlignment"], "longest_edge")
            for k, v in weights.items():
                torch.testing.assert_close(
                    s.policy.model.state_dict()[k], v, rtol=0, atol=0
                )
            q = Session(library=library, device="cpu", autostart=False)
            q.persistence_root = Path(directory)
            q.build()
            self.assertEqual(q.env.nz, 4)
            self.assertEqual(q.env.context["gridAlignment"], "longest_edge")
            torch.testing.assert_close(q.env.occupied, s.env.occupied)
        fresh = Session(library=library, device="cpu", autostart=False)
        fresh.build()
        self.assertEqual(fresh.env.context["gridAlignment"], "auto_fit")


if __name__ == "__main__":
    unittest.main()
