import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from voxelform.context import ContextLibrary, prepare_scene, triangle_mask
from voxelform.environment import Environment
from voxelform.shadows import DirectShadows, MeshShadows, ReceiverCache


def prism(x0=0, y0=0, x1=4, y1=3, height=6):
    return dict(vertices=[[x,y,z] for z in (0,height) for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))],
                faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                       [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])


def fixture():
    return dict(site=prism(-2,-1,6,5,.2),buildings=[prism(8,-1,12,5,12)],
                roads=[prism(-4,-5,14,-3,.01)],sitePerimeter=[[-2,-1],[6,-1],[6,5],[-2,5]],siteArea=48)


class ImportedContext(unittest.TestCase):
    def test_translation_keeps_metric_scale_and_relative_positions(self):
        raw=fixture();scene=prepare_scene(raw,dict(site_id='test',city_name='Test'),floors=4)
        self.assertEqual(scene['shape'],(4,8,10))
        self.assertEqual(int(scene['allowed'].sum()),48)
        self.assertEqual(scene['buildings'][0]['vertices'][0],[11.,1.,0.])
        np.testing.assert_allclose(np.diff(scene['buildings'][0]['vertices'][:2],axis=0),[[4,0,0]])
        self.assertEqual(raw['site']['vertices'][0],[-2,-1,0])

    def test_parcel_raster_keeps_holes_and_concavity(self):
        # Two disconnected strips: their bounding rectangle must not become buildable.
        faces=[[[0,0],[2,0],[2,6]],[[0,0],[2,6],[0,6]],
               [[4,0],[6,0],[6,6]],[[4,0],[6,6],[4,6]]]
        mask=triangle_mask(faces,6,6)
        self.assertEqual(mask.sum(),24)
        self.assertFalse(mask[:,2:4].any())

    def test_reader_parses_json_without_executing_embedded_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);public=root/'frontend/public';(public/'data').mkdir(parents=True);(public/'sites').mkdir()
            records=[dict(site_id='test',render_html='/sites/test.html?v=123',city_name='Test'),
                     dict(site_id='escape',render_html='../../outside.html')]
            (public/'data/master_urban_dataset.json').write_text(json.dumps(records))
            (public/'sites/test.html').write_text('const DATA = '+json.dumps(fixture())+'; throw new Error("do not execute");')
            library=ContextLibrary(root)
            self.assertEqual([r['site_id'] for r in library.catalog()],['test'])
            self.assertEqual(library.load('test')['source'],'cached_osm')
            with self.assertRaises(ValueError):library.load('escape')

    def test_environment_uses_imported_parcel_and_meshes_without_invented_noise(self):
        scene=prepare_scene(fixture(),dict(site_id='test'),floors=4)
        e=Environment(context=scene,sun_count=4)
        self.assertEqual(int(e.total_area),48*4)
        self.assertEqual(len(e.boxes),0)
        self.assertIsInstance(e.static,MeshShadows)
        self.assertIsNone(e.metrics()['noisePrior'])
        self.assertFalse(e.metrics()['noiseAvailable'])
        self.assertEqual(e.metrics()['contextSource'],'cached_osm')
        self.assertTrue(torch.equal(e.allowed[0],torch.tensor(scene['allowed'])))
        self.assertTrue(torch.isfinite(e.quality))

    def test_mesh_rays_match_independent_box_slab_reference(self):
        mesh=prism(-2,-1,2,1,6)
        suns=np.array([[1,0,0],[0,1,0],[0,0,1],[.3,.4,.7]],dtype=np.float32)
        suns/=np.linalg.norm(suns,axis=1)[:,None]
        points=torch.tensor(np.random.default_rng(19).uniform([-5,-5,.1],[5,5,9],(200,3)),dtype=torch.float32)
        reference=DirectShadows([[0,0,3,2,1,3,1,0]],suns).query(points)
        actual=MeshShadows([mesh],suns).query(points)
        self.assertTrue(torch.equal(actual,reference))

    def test_mesh_rays_do_not_fill_concave_empty_space(self):
        # A pair of arms leaves the upper-right quadrant open.
        method=MeshShadows([prism(0,0,4,1,6),prism(0,1,1,4,6)],[[0,0,1]])
        points=torch.tensor([[2.,2.,1.],[.5,2.,1.],[2.,.5,1.]])
        self.assertEqual(method.query(points).tolist(),[[True,False,False]])
        cache=ReceiverCache(method,10)
        _,fresh=cache.query(torch.tensor([1,3,5]),points);self.assertEqual(fresh,3)
        _,fresh=cache.query(torch.tensor([1,3,5]),points);self.assertEqual(fresh,0)

    def test_empty_mesh_context_is_unobstructed(self):
        self.assertEqual(MeshShadows([],[[0,0,1]]).query(torch.zeros((2,3))).tolist(),[[True,True]])

    def test_oversize_scene_rejected_before_simulation_allocation(self):
        raw=fixture();raw['site']=prism(0,0,200,200,.2)
        with self.assertRaisesRegex(ValueError,'grid limit'):prepare_scene(raw,dict(site_id='large'))


if __name__=='__main__':unittest.main()
