// Illustrative placements on fetched green polygons, not surveyed tree data.
export function treePositions(greenSpaces,limit=100){
 const triangles=[];let total=0,seed=7147;
 const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 for(const mesh of greenSpaces){
  const low=Math.min(...mesh.vertices.map(v=>v[2]));
  for(const face of mesh.faces){
   const t=face.map(i=>mesh.vertices[i]);if(t.some(p=>Math.abs(p[2]-low)>1e-5))continue;
   const area=Math.abs((t[1][0]-t[0][0])*(t[2][1]-t[0][1])-(t[1][1]-t[0][1])*(t[2][0]-t[0][0]))/2;
   if(area>1e-6){total+=area;triangles.push({t,end:total});}
  }
 }
 const count=Math.min(limit,Math.floor(total/90)),points=[];
 for(let attempt=0;points.length<count&&attempt<count*12;attempt++){
  const pick=random()*total,{t}=triangles.find(q=>q.end>=pick),u=Math.sqrt(random()),v=random();
  const p=t[0].map((_,k)=>(1-u)*t[0][k]+u*(1-v)*t[1][k]+u*v*t[2][k]);
  if(points.every(q=>(p[0]-q[0])**2+(p[1]-q[1])**2>25))points.push(p);
 }
 return points;
}
