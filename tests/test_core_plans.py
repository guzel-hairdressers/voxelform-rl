import heapq,math,unittest,xml.etree.ElementTree as ET
import numpy as np
import torch
from voxelform.core_plans import CorePlanParameters,DEFAULT_HEIGHT_BANDS,make_plan,render_plan_svg
from voxelform.cores import compute_core_distance,UNREACHABLE_DIST

class CorePlans(unittest.TestCase):
    def test_drawings_and_component_dimensions(self):
        for band in DEFAULT_HEIGHT_BANDS:
            p=make_plan(band);root=ET.fromstring(render_plan_svg(p));self.assertTrue(root.tag.endswith('svg'))
            self.assertEqual(p['minHeight'],band.minimum)
            for c in p['components']:
                self.assertGreaterEqual(c['x'],0);self.assertGreaterEqual(c['y'],0)
                self.assertLessEqual(c['x']+c['w'],p['metricWidth']+1e-6)
                self.assertLessEqual(c['y']+c['h'],p['metricLength']+1e-6)
            flights=[c for c in p['components'] if c['type']=='flight']
            self.assertAlmostEqual(2*flights[0]['risers']*p['riserHeight'],p['floorHeight'])
            self.assertAlmostEqual(flights[0]['treads']*p['tread'],flights[0]['h'])
    def test_bad_geometry_rejected(self):
        for kwargs in [dict(tread=-1),dict(landing_depth=.2),dict(car_width=4),dict(floor_height=float('nan'))]:
            with self.assertRaises(ValueError):CorePlanParameters(**kwargs)
    def test_no_corner_cutting_and_long_corridor_matches_independent_dijkstra(self):
        rng=np.random.default_rng(4)
        for k in range(5):
            occ=rng.random((15,18))>.27
            if k==0:
                occ[:]=False;occ[::2,:]=True
                for y in range(1,14,2):occ[y,17 if (y//2)%2==0 else 0]=True
            occ[0,0]=True;core=np.zeros_like(occ);core[0,0]=True
            expected=np.full(occ.shape,UNREACHABLE_DIST);expected[0,0]=0.;q=[(0.,0,0)]
            while q:
                dist,y,x=heapq.heappop(q)
                if dist>expected[y,x]+1e-8:continue
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        ny,nx=y+dy,x+dx
                        if not (dy or dx) or not(0<=ny<15 and 0<=nx<18 and occ[ny,nx]):continue
                        if dy and dx and not(occ[y,nx] and occ[ny,x]):continue
                        d=dist+math.hypot(dy,dx)
                        if d<expected[ny,nx]:expected[ny,nx]=d;heapq.heappush(q,(d,ny,nx))
            actual=compute_core_distance(torch.from_numpy(occ[None]),torch.from_numpy(core[None]))[0].numpy()
            np.testing.assert_allclose(actual[occ],expected[occ],atol=2e-4)
            if k==0:self.assertGreater(actual[14,0],64)
