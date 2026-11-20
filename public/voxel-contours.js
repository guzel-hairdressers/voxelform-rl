// Axis-aligned boundary creases, with collinear unit edges merged into runs.
// Four incident cells determine each lattice edge. Two adjacent occupied cells
// form a flat face; two diagonally touching cells retain their boundary edge.
export function voxelContours(occupied,shape){
 const [nz,ny,nx]=shape,size=[nx,ny,nz],segments=[];
 const at=(p)=>p[0]>=0&&p[0]<nx&&p[1]>=0&&p[1]<ny&&p[2]>=0&&p[2]<nz&&!!occupied[(p[2]*ny+p[1])*nx+p[0]];
 for(let axis=0;axis<3;axis++){
  const b=(axis+1)%3,c=(axis+2)%3,p=[0,0,0];
  for(let j=0;j<=size[b];j++)for(let k=0;k<=size[c];k++){
   p[b]=j;p[c]=k;let start=-1;
   for(let i=0;i<=size[axis];i++){
    p[axis]=i;let crease=false;
    if(i<size[axis]){
     const q=p.slice();q[b]--;q[c]--;const a=at(q);q[b]++;const d=at(q);q[c]++;const e=at(q);q[b]--;const f=at(q);
     const count=Number(a)+Number(d)+Number(e)+Number(f);
     crease=count===1||count===3||(count===2&&((a&&e)||(d&&f)));
    }
    if(crease&&start<0)start=i;
    if(!crease&&start>=0){p[axis]=start;segments.push(...p);p[axis]=i;segments.push(...p);start=-1;}
   }
  }
 }
 return new Float32Array(segments);
}

// Camera-independent edge geometry plus the two incident face directions.
// The material tests their facing on GPU as the camera moves. A concave pair
// may reverse both normals: their facing-product (and silhouette) is unchanged.
export function voxelSilhouetteEdges(occupied,shape){
 const [nz,ny,nx]=shape,size=[nx,ny,nz],positions=[],faceA=[],faceB=[],concave=[];
 const at=p=>p[0]>=0&&p[0]<nx&&p[1]>=0&&p[1]<ny&&p[2]>=0&&p[2]<nz&&!!occupied[(p[2]*ny+p[1])*nx+p[0]];
 for(let axis=0;axis<3;axis++){
  const b=(axis+1)%3,c=(axis+2)%3,p=[0,0,0];
  for(let j=0;j<=size[b];j++)for(let k=0;k<=size[c];k++){
   p[b]=j;p[c]=k;let start=0,previous=0;
   for(let i=0;i<=size[axis];i++){
    p[axis]=i;let code=0;
    if(i<size[axis]){
     const q=p.slice();q[b]--;q[c]--;const a=at(q);q[b]++;const d=at(q);q[c]++;const e=at(q);q[b]--;const f=at(q);
     const cells=[a,d,e,f],count=cells.filter(Boolean).length;
     if(count===1||count===3||(count===2&&((a&&e)||(d&&f)))){
      const corner=count===3?cells.indexOf(false):cells.indexOf(true);
      code=(count===3?4:0)+[1,2,4,3][corner];
     }
    }
    if(code!==previous){
     if(previous){
      p[axis]=start;positions.push(...p);p[axis]=i;positions.push(...p);
      const signs=[[1,1],[-1,1],[1,-1],[-1,-1]][(previous-1)&3],a=[0,0,0],bNormal=[0,0,0];a[b]=signs[0];bNormal[c]=signs[1];
      faceA.push(...a,...a);faceB.push(...bNormal,...bNormal);concave.push(previous>4?1:0);
     }
     start=i;previous=code;
    }
   }
  }
 }
 return {positions:new Float32Array(positions),faceA:new Float32Array(faceA),faceB:new Float32Array(faceB),concave:new Uint8Array(concave)};
}
