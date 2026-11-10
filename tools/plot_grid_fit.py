"""Draw two grid footprints in the same unchanged world coordinates."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from voxelform.context import ContextLibrary
from tools.grid_fit_study import full_mask

def main():
    fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained');lib=ContextLibrary()
    for ax,alignment in zip(axes,['world','auto_fit']):
        s=lib.load('m_bcn_0093',grid_alignment=alignment);a=s['gridAngle'];c,sn=np.cos(a),np.sin(a)
        inverse=np.array([[c,sn],[-sn,c]])
        def world(p):return (p-np.asarray(s['offset'][:2]))@inverse
        perimeter=world(np.asarray(s['perimeter']));perimeter=np.vstack((perimeter,perimeter[0]))
        mask=full_mask(s);y,x=np.where(s['allowed']);corners=np.array([[0,0],[1,0],[1,1],[0,1]])
        polygons=world(np.stack((x,y),1)[:,None,:]+corners)
        colours=np.where(mask[y,x],'#387d96','#e2a084')
        ax.add_collection(PolyCollection(polygons,facecolors=colours,edgecolors='white',linewidths=.2))
        ax.plot(*perimeter.T,color='#242b33',linewidth=1.5)
        title='World axes' if alignment=='world' else 'Auto: rotate and shift grid'
        ax.set_title(f'{title}\n{int(mask.sum())} full squares · {np.rad2deg(a):.1f}°')
        ax.autoscale();ax.set_aspect('equal');ax.set_xlabel('World east (m)');ax.set_ylabel('World north (m)')
        ax.annotate('N',xy=(.92,.93),xytext=(.92,.80),xycoords='axes fraction',ha='center',arrowprops=dict(arrowstyle='->'))
    fig.suptitle('Same Barcelona plot, fixed north. Blue: fully inside; orange: crosses plot edge.')
    out=Path(__file__).resolve().parents[1]/'agent_notes/grid_quality_2026-11-09/grid-fit.png'
    fig.savefig(out,dpi=160);plt.close(fig);print(out)
if __name__=='__main__':main()
