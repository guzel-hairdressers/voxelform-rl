import heapq,math,unittest,xml.etree.ElementTree as ET
import numpy as np
import torch
from voxelform.core_plans import CorePlanParameters,HeightBand,DEFAULT_HEIGHT_BANDS,make_plan,render_plan_svg
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
    def test_scissor_routes_landings_and_repeated_section(self):
        plan=make_plan(next(b for b in DEFAULT_HEIGHT_BANDS if b.tier=='T4'))
        self.assertEqual(plan['stairLayout'],'scissor')
        self.assertEqual((plan['stairCount'],plan['stairEnclosureCount'],plan['liftCount']),(2,1,4))
        self.assertEqual((plan['width'],plan['length']),(7,9))
        self.assertAlmostEqual(plan['metricArea'],55.02)
        self.assertEqual(len([d for d in plan['doors'] if d['type']=='stair_door']),2)
        self.assertEqual(len([d for d in plan['doors'] if d['type']=='lift_door']),4)
        self.assertEqual(len([c for c in plan['components'] if c['type']=='car']),4)
        self.assertAlmostEqual(plan['endLandingDepth'],1.21)
        self.assertEqual(len([z for z in plan['zones'] if z['type']=='service']),0)
        flights=[c for c in plan['components'] if c['type']=='flight']
        for route in plan['routes']:
            self.assertEqual([n['z'] for n in route['nodes']],[0.,1.5,3.])
            self.assertNotEqual(route['laneOnEvenStorey'],route['laneOnOddStorey'])
            self.assertEqual(route['risersPerStorey'],18)
            mid=[c for c in plan['components'] if c.get('route')==route['id'] and c.get('level')=='intermediate']
            self.assertEqual(len(mid),1)
            self.assertAlmostEqual(mid[0]['elevation'],1.5)
        # The two routes have disjoint walking surfaces, including half landings.
        routes=[[c for c in plan['components'] if c.get('route')==r] for r in ('A','B')]
        for a in routes[0]:
            for b in routes[1]:
                dx=min(a['x']+a['w'],b['x']+b['w'])-max(a['x'],b['x'])
                dy=min(a['y']+a['h'],b['y']+b['h'])-max(a['y'],b['y'])
                self.assertFalse(dx>1e-6 and dy>1e-6)
        # Flights repeat after a FULL storey, never a 1.5m overlap. Test slopes
        # independently at points along each flight with slab depth subtracted.
        for f in flights:
            for t in (.1,.5,.9):
                z=f['zStart']+(t if f['direction']>0 else 1-t)*(f['zEnd']-f['zStart'])
                upper_z=z+plan['floorHeight']
                self.assertGreaterEqual(upper_z-plan['stairSlabThickness']-z,2.0)
        # Impossible headroom must fail instead of silently shrinking the core.
        with self.assertRaises(ValueError):
            make_plan(HeightBand('test',19,30,2,3,'scissor'),CorePlanParameters(floor_height=2.0))

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
