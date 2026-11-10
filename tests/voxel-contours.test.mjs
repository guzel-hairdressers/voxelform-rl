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
 const {positions,faceA,faceB}=voxelSilhouetteEdges(new Uint8Array(8*12*16).fill(1),[8,12,16]);
 for(const eye of [[1,1,1],[-1,1,1],[1,-1,-1]]){
  let visible=0;
  for(let i=0;i<positions.length;i+=6){const dot=v=>eye.reduce((sum,n,k)=>sum+n*v[i+k],0);if(dot(faceA)*dot(faceB)<=0)visible++;}
  assert.equal(visible,6);
 }
});
