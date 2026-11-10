"""CPU-only geometric alignment study; no light/reward optimization."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import shapely
from scipy.ndimage import label
from voxelform.context import ContextLibrary

ROOT=Path(__file__).resolve().parents[1]
def full_mask(scene):
    mesh=scene['site'];v=np.asarray(mesh['vertices']);t=v[np.asarray(mesh['faces'])]
    t=t[np.all(np.abs(t[:,:,2]-v[:,2].min())<1e-6,axis=1),:,:2]
    polygon=shapely.buffer(shapely.union_all(shapely.polygons(t)),1e-8)
    y,x=np.mgrid[:scene['shape'][1],:scene['shape'][2]]
    return np.asarray(shapely.covers(polygon,shapely.box(x,y,x+1,y+1)))

def main():
    path=ROOT/'results/grid-fit-2026-11-09.json'
    if path.exists():raise ValueError('Preserve existing results')
    rows=[];lib=ContextLibrary()
    for site in ['m_bcn_0093','m_bcn_0094','m_nyc_0003','m_prs_0315']:
        for alignment in ['world','principal_major','principal_minor','longest_edge','auto_fit']:
            start=time.perf_counter();scene=lib.load(site,grid_alignment=alignment);seconds=time.perf_counter()-start
            full=full_mask(scene)
            if alignment=='auto_fit':assert np.array_equal(full,scene['allowed'])
            row=dict(site=site,alignment=alignment,angleDegrees=float(np.rad2deg(scene['gridAngle'])),
                fullSquares=int(full.sum()),allowedSquares=int(scene['allowed'].sum()),
                parcelArea=scene['siteArea'],gridCells=int(np.prod(scene['shape'][1:])),
                footprintComponents=int(label(full)[1]),setupSeconds=seconds,fit=scene['gridFit'])
            rows.append(row);print(row,flush=True)
            path.write_text(json.dumps(dict(rows=rows,hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'voxelform/context.py']}),indent=2)+'\n')
if __name__=='__main__':main()
