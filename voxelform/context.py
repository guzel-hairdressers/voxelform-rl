"""Read cached Module Lab scene data without importing or changing that project."""
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'Module-Lab'


class ContextLibrary:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv('VOXELFORM_CONTEXT_ROOT', DEFAULT_ROOT)).resolve()

    def catalog(self):
        records = {}
        for name in ('master_urban_dataset.json', 'custom_sites_dataset.json'):
            paths = [self.root / base / 'data' / name for base in ('frontend/public', 'public')]
            index = next((p for p in paths if p.is_file()), None)
            if index is None:
                continue
            for row in json.loads(index.read_text()):
                if row.get('site_id') and self._scene_path(row) is not None:
                    records[row['site_id']] = row
        return list(records.values())

    def _scene_path(self, record):
        url = urlsplit(record.get('render_html', ''))
        relative = Path(unquote(url.path).lstrip('/'))
        if url.scheme or url.netloc or relative.suffix != '.html':
            return None
        for base in ('frontend/public', 'public'):
            public = (self.root / base).resolve()
            candidate = (public / relative).resolve()
            if public in candidate.parents and candidate.is_file():
                return candidate
        return None

    def load(self, site_id, floors=6,grid_alignment='world'):
        record = next((r for r in self.catalog() if r['site_id'] == site_id), None)
        if record is None:
            raise ValueError('This cached site is unavailable. Select another site.')
        path = self._scene_path(record)
        text = path.read_text()
        # The existing viewer embeds JSON, never execute its JavaScript/HTML.
        marker = 'const DATA ='
        if marker not in text:
            raise ValueError('Cached scene does not contain the expected JSON data.')
        raw, _ = json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())
        return prepare_scene(raw, record, floors,grid_alignment)


def mesh(value, offset,rotation=None):
    vertices = np.asarray(value.get('vertices', []), dtype=np.float64)
    faces = np.asarray(value.get('faces', []), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid cached mesh vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('Invalid cached mesh triangles')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid cached mesh indices')
    if rotation is not None:
        vertices=vertices.copy();vertices[:,:2]=vertices[:,:2]@rotation
    return dict(vertices=(vertices + offset).tolist(), faces=faces.tolist(),
                height=value.get('height'), use=value.get('use'))


def triangle_mask(triangles, nx, ny):
    """Cell-centre rasterization of the actual triangulated parcel, including holes."""
    y, x = np.mgrid[:ny, :nx] + .5
    result = np.zeros((ny, nx), dtype=bool)
    for triangle in triangles:
        a, b, c = np.asarray(triangle)[:, :2]
        ab, ac = b-a, c-a
        det = ab[0]*ac[1]-ab[1]*ac[0]
        if abs(det) < 1e-10:
            continue
        px, py = x-a[0], y-a[1]
        u = (px*ac[1]-py*ac[0])/det
        v = (ab[0]*py-ab[1]*px)/det
        result |= (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
    return result


def grid_orientation(raw,alignment):
    """Area-weighted polygon principal axes, or an explicit perimeter edge."""
    points=np.asarray(raw['site']['vertices'],float);tri=points[np.asarray(raw['site']['faces'],int)]
    tri=tri[np.all(np.abs(tri[:,:,2]-points[:,2].min())<1e-6,axis=1),:,:2]
    tri=tri-tri.reshape(-1,2).mean(0)  # Stable even with large projected coordinates.
    u,v=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0];areas=np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])/2
    total=areas.sum()
    if total<=0:raise ValueError('Site has no positive ground area')
    centroid=(areas[:,None]*tri.mean(1)).sum(0)/total
    sums=tri.sum(1)
    second=(np.einsum('tvi,tvj->tij',tri,tri)+np.einsum('ti,tj->tij',sums,sums))/12
    covariance=(areas[:,None,None]*second).sum(0)/total-np.outer(centroid,centroid)
    values,vectors=np.linalg.eigh(covariance);major=vectors[:,-1]
    principal=float(np.arctan2(major[1],major[0]))
    perimeter=np.asarray(raw.get('sitePerimeter',[]),float)
    if len(perimeter)>1 and np.allclose(perimeter[0],perimeter[-1]):perimeter=perimeter[:-1]
    edges=[]
    for i in range(len(perimeter)):
        delta=perimeter[(i+1)%len(perimeter),:2]-perimeter[i,:2];length=float(np.linalg.norm(delta))
        if length>1e-8:edges.append(dict(index=i,length=length,angle=float(np.arctan2(delta[1],delta[0]))))
    edges.sort(key=lambda x:(-x['length'],x['index']))
    if alignment=='world':angle=0.
    elif alignment=='principal_major':angle=principal
    elif alignment=='principal_minor':angle=principal+np.pi/2
    elif alignment=='longest_edge':
        if not edges:raise ValueError('Site has no perimeter edges')
        angle=edges[0]['angle']
    elif isinstance(alignment,str) and alignment.startswith('edge:'):
        try:angle=next(e['angle'] for e in edges if e['index']==int(alignment[5:]))
        except (ValueError,StopIteration):raise ValueError('Unknown perimeter edge') from None
    else:raise ValueError('Unknown grid alignment')
    angle=float((angle+np.pi/2)%np.pi-np.pi/2)
    if abs(angle)<1e-12:angle=0.
    return angle,edges


def fit_grid(raw):
    """Fit full 1 m squares, searching angle and sub-cell translation in 2D.

    This is a bounded geometric search, not an environmental optimization.
    Triangle union preserves concavity and holes; corner tests alone would not.
    """
    import shapely
    points=np.asarray(raw['site']['vertices'],float)
    triangles=points[np.asarray(raw['site']['faces'],int)]
    triangles=triangles[np.all(np.abs(triangles[:,:,2]-points[:,2].min())<1e-6,axis=1),:,:2]
    anchor=points[:,:2].min(0);triangles=triangles-anchor
    principal,edges=grid_orientation(raw,'principal_major')
    angles=np.r_[np.deg2rad(np.arange(0,90,3)),principal,[e['angle'] for e in edges[:16]]]
    coarse={};evaluated=0
    def evaluate(angle,phases):
        nonlocal evaluated
        angle=float(angle%(np.pi/2))
        if abs(angle-np.pi/2)<1e-10 or abs(angle)<1e-10:angle=0.
        c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s],[s,c]])
        tri=triangles@r;bounds=(points[:,:2]-anchor)@r;minimum=bounds.min(0);extent=np.ptp(bounds,axis=0)
        nx,ny=np.ceil(extent-1e-9).astype(int)+2
        if max(nx,ny)>128 or nx*ny>10000:return None
        polygon=shapely.union_all(shapely.polygons(tri-minimum))
        # Tiny tolerance handles exact edge coincidences after rotation.
        polygon=shapely.buffer(polygon,1e-8)
        shapely.prepare(polygon)
        y,x=np.mgrid[:ny,:nx];best=None
        for px,py in phases:
            xx=x-1-px;yy=y-1-py
            mask=np.asarray(shapely.covers(polygon,shapely.box(xx,yy,xx+1,yy+1)))
            count=int(mask.sum());evaluated+=1
            rank=(count,-int(nx*ny),-angle,-px,-py)
            if best is None or rank>best[0]:best=(rank,angle,(px,py),mask)
        return best
    half=[(x,y) for x in (0.,.5) for y in (0.,.5)]
    for angle in sorted(set(np.round(angles%(np.pi/2),12))):
        result=evaluate(angle,half)
        if result is not None:coarse[angle]=result
    if not coarse:raise ValueError('No fitted grid fits the current grid limits')
    winners=sorted(coarse.values(),key=lambda v:v[0],reverse=True)[:2];best=winners[0]
    quarters=[(x,y) for x in (0.,.25,.5,.75) for y in (0.,.25,.5,.75)]
    for angle in sorted({round(float((v[1]+delta)%(np.pi/2)),12) for v in winners for delta in np.deg2rad(np.arange(-3,3.01,.5))}):
        result=evaluate(angle,quarters)
        if result is not None and result[0]>best[0]:best=result
    if best[0][0]<=0:raise ValueError('No full 1 m square fits inside this plot')
    return best[1],best[2],best[3],dict(fullSquares=best[0][0],placementsTested=evaluated,phase=list(best[2]),criterion='full_square_count',cellSize=1.)


def prepare_scene(raw, record, floors=6,grid_alignment='world'):
    phase=(0.,0.);fitted=None;fit=None
    if grid_alignment=='auto_fit':
        angle,phase,fitted,fit=fit_grid(raw)
        _,edges=grid_orientation(raw,'world')
    else:angle,edges=grid_orientation(raw,grid_alignment)
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    original = mesh(raw['site'], np.zeros(3))
    points = np.asarray(original['vertices'])
    points=points.copy();points[:,:2]=points[:,:2]@rotation
    minimum = points[:, :2].min(0)
    extent = points[:, :2].max(0)-minimum
    nx, ny = np.ceil(extent-1e-9).astype(int)+2
    if min(nx, ny) < 3 or max(nx, ny) > 128 or nx*ny > 10000:
        raise ValueError('This parcel exceeds the current 1 m grid limit (128 m per side / 10,000 cells). Select a smaller cached site.')
    offset = np.array([1-minimum[0]+phase[0], 1-minimum[1]+phase[1], 0.])
    site = mesh(raw['site'], offset,rotation)
    vertices = np.asarray(site['vertices'])
    triangles = vertices[np.asarray(site['faces'])]
    bottom = np.all(np.abs(triangles[:, :, 2]-vertices[:, 2].min()) < 1e-6, axis=1)
    allowed = fitted if fitted is not None else triangle_mask(triangles[bottom], int(nx), int(ny))
    if not allowed.any():
        raise ValueError('This cached parcel has no rasterizable ground faces.')
    perimeter = raw.get('sitePerimeter', [])
    return dict(site_id=record['site_id'], city=record.get('city_name', raw.get('cityName', 'Urban site')),
                tier=record.get('area_tier', ''), source='cached_osm',
                lat=record.get('lat'), lon=record.get('lon'),
                siteArea=raw.get('siteArea', record.get('site_area_m2')),
                radius=float(raw.get('radius', 100.)),
                shape=(int(floors), int(ny), int(nx)), allowed=allowed,
                site=site, buildings=[mesh(b, offset,rotation) for b in raw.get('buildings', [])],
                roads=[mesh(r, offset,rotation) for r in raw.get('roads', [])],
                greenSpaces=[mesh(g, offset,rotation) for g in raw.get('greenSpaces', [])],
                perimeter=[(np.asarray(p[:2])@rotation+offset[:2]).tolist() for p in perimeter],
                gridAlignment=grid_alignment,gridAngle=angle,gridEdges=edges,gridFit=fit,
                offset=offset.tolist(), noiseAvailable=False,
                attribution='© OpenStreetMap contributors · cached Module Lab context')


def public_scene(scene):
    return {k: v for k, v in scene.items() if k not in ('allowed', 'shape')}
