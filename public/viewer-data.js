// Geometry/protocol adapter only. Module Lab owns the entire graphic engine.
const faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]];
export function gridToWorld(x,y,offset,angle=0){const c=Math.cos(angle),s=Math.sin(angle),a=x-offset[0],b=y-offset[1];return [c*a-s*b,s*a+c*b];}
function originalMesh(mesh,offset,angle=0){return {...mesh,vertices:mesh.vertices.map(p=>[...gridToWorld(p[0],p[1],offset,angle),p[2]])};}
function boxMesh(b){
 const vertices=[];
 for(const z of [-1,1])for(const [x,y] of [[-1,-1],[1,-1],[1,1],[-1,1]]){
  vertices.push([b[0]+b[6]*x*b[3]-b[7]*y*b[4],b[1]+b[7]*x*b[3]+b[6]*y*b[4],b[2]+z*b[5]]);
 }
 return {vertices,faces,height:b[5]*2};
}
export function sceneData(state){
 const c=state.context,[nz,ny,nx]=state.shape;
 if(c){
  const offset=c.offset||[0,0,0],angle=c.gridAngle||0,buildings=c.buildings.map(m=>originalMesh(m,offset,angle));
  return {buildings,site:originalMesh(c.site,offset,angle),sitePerimeter:c.perimeter.map(p=>gridToWorld(p[0],p[1],offset,angle)),
   roads:c.roads.map(m=>originalMesh(m,offset,angle)),greenSpaces:c.greenSpaces.map(m=>originalMesh(m,offset,angle)),
   siteArea:c.siteArea,areaTier:c.tier,cityName:c.city,radius:c.radius||100,maxHeight:Math.max(nz*3,...buildings.map(b=>b.height||0)),coords:{lat:c.lat,lon:c.lon}};
 }
 const site={vertices:[],faces:[]};
 for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)if(state.allowed[y*nx+x]){
  const k=site.vertices.length;site.vertices.push([x-nx/2,y-ny/2,.2],[x+1-nx/2,y-ny/2,.2],[x+1-nx/2,y+1-ny/2,.2],[x-nx/2,y+1-ny/2,.2]);site.faces.push([k,k+1,k+2],[k,k+2,k+3]);
 }
 return {buildings:state.boxes.map(boxMesh).map(m=>originalMesh(m,[nx/2,ny/2,0])),site,roads:[],greenSpaces:[],siteArea:state.allowed.slice(0,nx*ny).reduce((a,b)=>a+b,0),radius:Math.max(75,nx,ny),maxHeight:nz*3,cityName:'Synthetic method fixture'};
}
export function voxelPosition(index,shape,offset,angle=0){
 const [,ny,nx]=shape,z=Math.floor(index/(nx*ny)),y=Math.floor(index/nx)%ny,x=index%nx;
 const [wx,wy]=gridToWorld(x+.5,y+.5,offset,angle);return [wx,(z+.5)*3,-wy];
}
export function sceneOffset(state){return state.context?.offset||[state.shape[2]/2,state.shape[1]/2,0];}
