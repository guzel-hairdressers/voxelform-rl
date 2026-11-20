import test from 'node:test';
import assert from 'node:assert/strict';
import {voxelContours,voxelSilhouetteEdges} from '../public/voxel-contours.js';
const key=(a,b)=>[a.join(','),b.join(',')].sort().join(':');
function expand(lines){
 const result=new Set();
 for(let i=0;i<lines.length;i+=6){const a=Array.from(lines.slice(i,i+3)),b=Array.from(lines.slice(i+3,i+6)),axis=a.findIndex((v,k)=>v!==b[k]);
  for(let j=a[axis];j<b[axis];j++){const p=a.slice(),q=a.slice();p[axis]=j;q[axis]=j+1;result.add(key(p,q));}
 }
 return result;
}
function reference(v,[nz,ny,nx]){
 const at=(p)=>p[0]>=0&&p[0]<nx&&p[1]>=0&&p[1]<ny&&p[2]>=0&&p[2]<nz&&v[(p[2]*ny+p[1])*nx+p[0]],edges=new Map();
 for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)if(at([x,y,z])){
  for(let axis=0;axis<3;axis++)for(const side of [-1,1]){
   const p=[x,y,z],n=p.slice();n[axis]+=side;if(at(n))continue;
   const b=(axis+1)%3,c=(axis+2)%3;p[axis]+=side>0?1:0;
   const corners=[[0,0],[1,0],[1,1],[0,1]].map(([j,k])=>{const q=p.slice();q[b]+=j;q[c]+=k;return q;});
   for(let k=0;k<4;k++){const id=key(corners[k],corners[(k+1)%4]);if(!edges.has(id))edges.set(id,new Set());edges.get(id).add(axis);}
  }
 }
 return new Set([...edges].filter(([,axes])=>axes.size>1).map(([id])=>id));
}
test('solid cuboid has twelve merged edges and no face or floor seams',()=>{
 assert.equal(voxelContours(new Uint8Array(8*12*16).fill(1),[8,12,16]).length/6,12);
 assert.equal(voxelContours(new Uint8Array(8),[2,2,2]).length,0);
});
test('creases match an independent exposed-face reference including cavities and diagonal contacts',()=>{
 let seed=63;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
 for(let trial=0;trial<60;trial++){const shape=[3,4,5],v=Uint8Array.from({length:60},()=>random()>.4?1:0);assert.deepEqual(expand(voxelContours(v,shape)),reference(v,shape));assert.deepEqual(expand(voxelSilhouetteEdges(v,shape).positions),reference(v,shape));}
});
test('camera-facing test retains six cuboid silhouette edges and drops the other six creases',()=>{
 const {positions,faceA,faceB,concave}=voxelSilhouetteEdges(new Uint8Array(8*12*16).fill(1),[8,12,16]);
 assert.equal(concave.length,positions.length/6);
 assert.ok(concave.every(v=>v===0));
 for(const eye of [[1,1,1],[-1,1,1],[1,-1,-1]]){
  let visible=0;
  for(let i=0;i<positions.length;i+=6){const dot=v=>eye.reduce((sum,n,k)=>sum+n*v[i+k],0);if(dot(faceA)*dot(faceB)<=0)visible++;}
  assert.equal(visible,6);
 }
});
test('stepped massing separates concave inner creases and culls interior front creases under rotation',()=>{
 // Shape [nz=3, ny=4, nx=4]: base [0..1]x[0..4]x[0..2], upper cantilevered wing [1..3]x[0..2]x[0..3]
 const shape=[3,4,4],occ=new Uint8Array(3*4*4);
 const set=(z,y,x)=>{occ[(z*4+y)*4+x]=1;};
 for(let y=0;y<4;y++)for(let x=0;x<2;x++)set(0,y,x);
 for(let z=1;z<3;z++)for(let y=0;y<2;y++)for(let x=0;x<3;x++)set(z,y,x);
 const {positions,faceA,faceB,concave}=voxelSilhouetteEdges(occ,shape);
 assert.equal(concave.length,positions.length/6);
 // Line y=2, z=1 runs along X: [0..2] is concave (count=3), [2..3] is convex cantilever underside (count=1).
 // Must not be merged across x=2.
 const xSegs=[];
 for(let s=0;s<concave.length;s++){
  const p0=positions.slice(s*6,s*6+3),p1=positions.slice(s*6+3,s*6+6);
  if(p0[1]===2&&p1[1]===2&&p0[2]===1&&p1[2]===1)xSegs.push({x0:p0[0],x1:p1[0],concave:concave[s]});
 }
 assert.deepEqual(xSegs,[{x0:0,x1:2,concave:1},{x0:2,x1:3,concave:0}]);
 // With stepped podium + upper block (no cantilever), verify camera silhouette test drops all concave & front-facing convex edges
 const [nz,ny,nx]=[2,4,4],stepOcc=new Uint8Array(nz*ny*nx);
 for(let y=0;y<4;y++)for(let x=0;x<4;x++)stepOcc[y*nx+x]=1;
 for(let y=0;y<2;y++)for(let x=0;x<2;x++)stepOcc[ny*nx+y*nx+x]=1;
 const step=voxelSilhouetteEdges(stepOcc,[nz,ny,nx]);
 assert.ok(step.concave.some(v=>v===1));
 const angle=0.42,cosA=Math.cos(angle),sinA=Math.sin(angle);
 const toWorldNormal=n=>[cosA*n[0]-sinA*n[1],n[2],-(sinA*n[0]+cosA*n[1])];
 const eye=[1,0.9,1.2];
 const norm=Math.hypot(...eye),view=eye.map(v=>v/norm);
 let kept=0,droppedFront=0,droppedConcave=0;
 for(let s=0;s<step.concave.length;s++){
  if(step.concave[s]){droppedConcave++;continue;}
  const nA=toWorldNormal(step.faceA.slice(s*6,s*6+3)),nB=toWorldNormal(step.faceB.slice(s*6,s*6+3));
  const dotA=nA[0]*view[0]+nA[1]*view[1]+nA[2]*view[2];
  const dotB=nB[0]*view[0]+nB[1]*view[1]+nB[2]*view[2];
  const isSil=!(dotA*dotB>1e-4||(Math.abs(dotA)<1e-4&&Math.abs(dotB)<1e-4));
  if(isSil)kept++;
  else if(dotA>0&&dotB>0)droppedFront++;
 }
 assert.ok(droppedConcave>=2);
 assert.ok(droppedFront>=1);
 assert.ok(kept>=6);
});
