"""Independent cavity sky-ray and line-source quadrature probes."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from voxelform.fields import diffuse_access,noise_prior
from voxelform.shadows import dda_paths

def sky_fraction(occupied,point,samples=256):
    i=np.arange(samples);r=np.sqrt((i+.5)/samples);a=i*math.pi*(3-math.sqrt(5))
    directions=np.stack((r*np.cos(a),r*np.sin(a),np.sqrt(1-r*r)),axis=1)
    paths,_=dda_paths(np.broadcast_to(point,directions.shape),directions,occupied.shape)
    flat=np.append(occupied.flatten(),False)
    return float((~flat[paths].any(1)).mean())

def main():
    torch.set_num_threads(1);rows=[];root=Path(__file__).resolve().parents[1]
    for width in (3,7,11):
        occupied=np.ones((8,17,17),bool);lo=(17-width)//2;occupied[:,lo:lo+width,lo:lo+width]=False
        light=diffuse_access(torch.tensor(occupied),iterations=160).numpy()
        for floor in (0,3,7):
            rows.append(dict(kind='well',width=width,heightToOpening=24-(floor+.5)*3,
                diffuseProxy=float(light[floor,8,8]),directSkyFraction=sky_fraction(occupied,[8.5,8.5,(floor+.5)*3])))
    occupied=np.ones((4,17,17),bool);occupied[:,12:15,12:15]=False
    occupied[0,12:15,2:15]=False;occupied[0,2:15,2:5]=False
    light=diffuse_access(torch.tensor(occupied),iterations=160).numpy()
    rows.append(dict(kind='two-turn-alcove',diffuseProxy=float(light[0,3,3]),directSkyFraction=sky_fraction(occupied,[3.5,3.5,1.5])))
    p=torch.tensor([[0.,3.,2.],[10.,15.,5.],[50.,20.,0.]],dtype=torch.float32)
    observed=noise_prior(p,[((-20,0,0),(20,0,0),1.)]).numpy();x=np.linspace(-20,20,20001)
    expected=np.array([np.trapezoid(1/((x-v[0])**2+v[1]**2+v[2]**2+4),x) for v in p.numpy()])
    result=dict(note='Diffuse and direct-sky values have different meanings/units: compare ordering, not equality or daylight compliance.',
        samples=256,cavities=rows,noise=dict(analytic=observed.tolist(),independentQuadrature=expected.tolist(),maxAbsoluteError=float(np.max(abs(observed-expected)))),
        sourceHashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'voxelform/fields.py',root/'voxelform/shadows.py',Path(__file__)]})
    output=root/'results/field-probes.json';output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
