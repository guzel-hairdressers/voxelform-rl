import unittest
import numpy as np
from voxelform.devoxelize import devoxelize_occupancy

class Devoxelize(unittest.TestCase):
    def test_modes_have_finite_nonzero_closed_mesh_for_box(self):
        solid=np.zeros((4,12,16),bool);solid[:3,2:10,2:14]=True
        for mode in ('correlated_loft','extruded_slabs','vsa_planes'):
            with self.subTest(mode=mode):
                mesh=devoxelize_occupancy(solid,1,mode);v=np.array(mesh['vertices']);f=np.array(mesh['faces'])
                self.assertTrue(np.isfinite(v).all());self.assertGreater(len(f),0)
                self.assertTrue((f>=0).all() and (f<len(v)).all())
                self.assertAlmostEqual(mesh['stats']['volume_m3'],864,delta=1)
                edges={}
                for face in f:
                    for u,w in zip(face,np.roll(face,-1)):
                        key=tuple(sorted((u,w)));edges[key]=edges.get(key,0)+1
                self.assertTrue(all(n==2 for n in edges.values()),'box must be watertight')
    def test_cache_uses_geometry_not_reused_revision_numbers(self):
        a=np.zeros((3,10,10),bool);a[:2,2:8,2:8]=True;b=a.copy();b[2,2:8,2:8]=True
        ma=devoxelize_occupancy(a,7);mb=devoxelize_occupancy(b,7)
        self.assertNotEqual(ma['stats']['volume_m3'],mb['stats']['volume_m3'])
        self.assertIs(ma,devoxelize_occupancy(a,99))
    def test_empty_and_unknown_mode(self):
        for mode in ('correlated_loft','extruded_slabs','vsa_planes'):
            self.assertEqual(devoxelize_occupancy(np.zeros((3,5,5),bool),0,mode)['faces'],[])
        with self.assertRaises(ValueError):devoxelize_occupancy(np.ones((3,5,5),bool),0,'fake')
    def test_courtyard_is_not_capped_over(self):
        solid=np.zeros((4,16,16),bool);solid[:3,2:14,2:14]=True;solid[:,6:10,6:10]=False
        for mode in ('correlated_loft','extruded_slabs','vsa_planes'):
            m=devoxelize_occupancy(solid,1,mode);v=np.array(m['vertices']);f=np.array(m['faces'])
            centers=v[f].mean(1)
            cap=np.ptp(v[f][:,:,2],axis=1)<1e-6
            self.assertFalse((cap&(centers[:,0]>6)&(centers[:,0]<10)&(centers[:,1]>6)&(centers[:,1]<10)).any())

    def test_stepped_courtyard_caps_and_walls_have_conforming_edges(self):
        solid=np.zeros((5,15,16),bool);solid[:2,2:13,2:14]=True;solid[2:4,3:12,3:13]=True;solid[:,6:9,6:10]=False
        for mode in ('correlated_loft','extruded_slabs','vsa_planes'):
            m=devoxelize_occupancy(solid,0,mode)
            self.assertEqual(m['stats']['boundaryEdges'],0)
            self.assertEqual(m['stats']['nonmanifoldEdges'],0)
