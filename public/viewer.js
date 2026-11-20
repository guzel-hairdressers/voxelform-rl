import * as THREE from 'three';
import {LineSegments2} from 'three/addons/lines/LineSegments2.js';
import {LineSegmentsGeometry} from 'three/addons/lines/LineSegmentsGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';
import {initUrbanContext} from '/shared-viewer/scripts/urban_viewer.js';
import {sceneData,sceneOffset,voxelPosition,gridToWorld} from './viewer-data.js?v=grid-2';
import {treePositions} from './context-trees.js?v=1';
import {voxelSilhouetteEdges} from './voxel-contours.js?v=3';

let initialized=false,mass=null,coreMass=null,coreShafts=null,lastRevision=-1,lastView=null,geometry=null,outline=null,outlineRevision=-1,outlineWorldWidth=null;
const dummy=new THREE.Object3D();
const blueHsl=new THREE.Color('#1645ff').getHSL({},THREE.SRGBColorSpace),outlineBlue=new THREE.Color().setHSL(blueHsl.h,blueHsl.s,.50,THREE.SRGBColorSpace);
const lightBlue=new THREE.Color().setHSL(blueHsl.h,blueHsl.s,.95,THREE.SRGBColorSpace).convertLinearToSRGB(),shadeBlue=new THREE.Color().setHSL(blueHsl.h,blueHsl.s,.70,THREE.SRGBColorSpace).convertLinearToSRGB();
const redHsl=new THREE.Color('#dc2626').getHSL({},THREE.SRGBColorSpace);
const lightCore=new THREE.Color().setHSL(redHsl.h,redHsl.s,.66,THREE.SRGBColorSpace).convertLinearToSRGB(),shadeCore=new THREE.Color().setHSL(redHsl.h,redHsl.s,.42,THREE.SRGBColorSpace).convertLinearToSRGB();
const solid=new THREE.Color('#ffffff'),usable=new THREE.Color('#59a59d'),unusable=new THREE.Color('#d4dce0'),coreColor=new THREE.Color('#dc2626'),unreachableCol=new THREE.Color('#6b21a8');
function heat(t){return new THREE.Color().setHSL(.65-Math.max(0,Math.min(1,t))*.61,.58,.54);}
function coreDistanceColor(d,maxDist){
 if(d>=900)return unreachableCol;
 const limit=Math.max(1,maxDist);
 if(d>limit){
  const over=Math.min(1,(d-limit)/Math.max(4,0.5*limit));
  return new THREE.Color().setHSL(0.95-over*0.08,0.84,0.50-over*0.08);
 }
 return new THREE.Color().setHSL(0.38-Math.max(0,Math.min(1,d/limit))*0.30,0.68,0.48);
}
function createSolidInkMaterial(lInk,sInk,cacheKey){
 const mat=new THREE.MeshLambertMaterial({flatShading:true,vertexColors:true,color:0xffffff,toneMapped:false,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1,stencilWrite:true,stencilWriteMask:0xff,stencilFunc:THREE.AlwaysStencilFunc,stencilRef:1,stencilFuncMask:0xff,stencilFail:THREE.KeepStencilOp,stencilZFail:THREE.KeepStencilOp,stencilZPass:THREE.ReplaceStencilOp});
 mat.customProgramCacheKey=()=>cacheKey||'solid-ink-1';
 mat.onBeforeCompile=shader=>{
  // Keep hue/saturation fixed in sRGB HSL; actual shared-engine illumination
  // and shadows move lightness between lightInk and shadeInk endpoints.
  shader.uniforms.lightInk={value:lInk};shader.uniforms.shadeInk={value:sInk};
  shader.fragmentShader='uniform vec3 lightInk; uniform vec3 shadeInk;\n'+shader.fragmentShader;
  shader.fragmentShader=shader.fragmentShader.replace('vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;','float illumination = clamp(max(max(reflectedLight.directDiffuse.r,reflectedLight.directDiffuse.g),reflectedLight.directDiffuse.b)*3.14159265/0.82,0.,1.);\n vec3 ink = mix(shadeInk,lightInk,illumination);\n vec3 outgoingLight = pow((ink+0.055)/1.055,vec3(2.4));');
 };
 return mat;
}
function addContextTrees(data){
 const positions=treePositions(data.greenSpaces||[]);if(!positions.length)return;
 const canvas=document.createElement('canvas');canvas.width=160;canvas.height=256;
 const pen=canvas.getContext('2d');let seed=138;
 const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 pen.strokeStyle='#7b837e';pen.lineWidth=3;pen.beginPath();pen.moveTo(80,250);pen.lineTo(80,88);pen.stroke();
 for(let i=0;i<85;i++){
  const a=random()*Math.PI*2,r=Math.sqrt(random());
  pen.fillStyle=i%3?'rgba(126,149,131,.32)':'rgba(174,190,174,.5)';
  pen.beginPath();pen.arc(80+Math.cos(a)*r*47,94+Math.sin(a)*r*68,9+random()*13,0,Math.PI*2);pen.fill();
 }
 const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
 const material=new THREE.SpriteMaterial({map:texture,transparent:true,alphaTest:.04,depthWrite:false});
 const group=new THREE.Group();group.name='Illustrative trees on fetched green areas';
 positions.forEach((p,i)=>{const tree=new THREE.Sprite(material),h=6.5+(i%4)*.6;tree.center.set(.5,0);tree.scale.set(h*160/256,h,1);tree.position.set(p[0],p[2]+.2,-p[1]);group.add(tree);});
 (window.scene.getObjectByName('surroundingContext')||window.scene).add(group);
}
function prepareGroundPlanes(){
 for(const child of window.scene?.children||[]){
  if(child.isMesh&&!child.receiveShadow&&child.geometry?.type==='PlaneGeometry'){
   child.renderOrder=-10;
   if(child.material&&child.material.depthWrite!==false){child.material.depthWrite=false;child.material.needsUpdate=true;}
  }
 }
 const context=window.scene?.getObjectByName('surroundingContext');
 if(context){
  for(const group of context.children){
   if(group.children.some(child=>child.userData?.isBuilding))continue;
   group.traverse(obj=>{
    if(obj.isMesh&&obj.material){obj.renderOrder=-5;if(obj.material.depthWrite!==false){obj.material.depthWrite=false;obj.material.needsUpdate=true;}}
   });
  }
 }
 const parcel=window.scene?.getObjectByName('testingSiteParcel');
 if(parcel){
  parcel.traverse(obj=>{
   if(obj.isMesh&&obj.material&&obj.material.depthWrite!==false){obj.material.depthWrite=false;obj.material.needsUpdate=true;}
  });
 }
}
function displayContext(visible,transparent){
 const context=window.scene?.getObjectByName('surroundingContext');if(!context)return;
 context.visible=visible;
 for(const group of context.children){
  if(!group.children.some(child=>child.userData?.isBuilding))continue;
  group.traverse(object=>{
   if(object.userData?.isBuilding){
    object.material.transparent=transparent;
    object.material.opacity=transparent ? .45 : 1;
    object.material.depthWrite=!transparent;
    object.renderOrder=transparent ? 30 : 0;
    object.material.needsUpdate=true;
   }else if(object.isMesh&&object.material?.colorWrite===false){
    object.material.depthWrite=!transparent;
    object.material.transparent=transparent;
    object.renderOrder=transparent ? 30 : 0;
    object.material.needsUpdate=true;
   }else if(object.isMesh&&object.receiveShadow){
    object.visible=!transparent;
   }else if(object.isLineSegments){
    object.material.transparent=transparent;
    object.material.opacity=transparent ? .55 : 1;
    object.material.depthWrite=!transparent;
    object.renderOrder=transparent ? 30 : 0;
    object.material.needsUpdate=true;
   }
  });
 }
}
function updateMassing(state,view){
 if(lastRevision===state.revision&&lastView===view)return;
 lastRevision=state.revision;lastView=view;
 if(mass){window.optimizerGroup.remove(mass);mass.dispose?.();mass.material.dispose();}
 if(coreMass){window.optimizerGroup.remove(coreMass);coreMass.dispose?.();coreMass.material.dispose();coreMass=null;}
 if(!geometry){
  geometry=new THREE.BoxGeometry(1,3,1);
 }
 // A blue model palette separates roofs and the two wall directions even in
 // full context shadow. Real shared-engine shadows still modulate the surface.
 const normals=geometry.getAttribute('normal'),colors=[],sun=new THREE.Vector3(130,220,90).normalize().applyAxisAngle(new THREE.Vector3(0,1,0),-(state.context?.gridAngle||0));
 for(let i=0;i<normals.count;i++){
  const n=new THREE.Vector3().fromBufferAttribute(normals,i);
  if(view==='solid'){
   colors.push(1,1,1);
  }else{const shade=.65+.35*Math.max(0,n.dot(sun));colors.push(shade,shade,shade);}
 }
 geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
 const hasCores=Boolean(state.coreMask&&state.coreMask.length===state.occupied.length);
 const showCoreVoxels=hasCores&&(view==='solid'||view==='usable'||view==='coreDistance');
 const ids=[],coreIds=[];
 for(let i=0;i<state.occupied.length;i++){
  if((view==='light'||view==='noise')?state.allowed[i]:state.occupied[i]){
   if(showCoreVoxels&&state.coreMask[i])coreIds.push(i);
   else ids.push(i);
  }
 }
 const material=view==='solid'
  ? createSolidInkMaterial(lightBlue,shadeBlue,'solid-ink-mass')
  : new THREE.MeshBasicMaterial({vertexColors:true,transparent:view==='light',opacity:view==='light' ? .72 : 1});
 mass=new THREE.InstancedMesh(geometry,material,ids.length);
 mass.name='VoxelForm massing';mass.castShadow=true;mass.receiveShadow=view==='solid';
 mass.renderOrder=10;mass.onBeforeRender=()=>{prepareGroundPlanes();};
 const offset=sceneOffset(state),displayLift=Math.max(.2,...(state.context?.site?.vertices||[]).map(v=>v[2]))+.03;
 const angle=state.context?.gridAngle||0,maxCoreDist=Number(state.settings?.maxCoreDist??12);
 ids.forEach((i,k)=>{
  dummy.position.set(...voxelPosition(i,state.shape,offset,angle));dummy.position.y+=displayLift;dummy.rotation.y=angle;dummy.updateMatrix();mass.setMatrixAt(k,dummy.matrix);
  let color=solid;
  if(view==='usable')color=state.usable[i]?usable:unusable;
  else if(view==='coreDistance')color=coreDistanceColor(state.coreDistance?.[i]??0,maxCoreDist);
  else if(view!=='solid')color=heat(state[view][i]);
  if(view==='light'&&state.occupied[i])color=unusable;mass.setColorAt(k,color);
 });
 mass.instanceMatrix.needsUpdate=true;if(mass.instanceColor)mass.instanceColor.needsUpdate=true;window.optimizerGroup.add(mass);
 if(coreIds.length){
  const coreMat=view==='solid'?createSolidInkMaterial(lightCore,shadeCore,'solid-ink-core'):new THREE.MeshBasicMaterial({vertexColors:true});
  coreMass=new THREE.InstancedMesh(geometry,coreMat,coreIds.length);
  coreMass.name='VoxelForm core shafts';coreMass.castShadow=true;coreMass.receiveShadow=view==='solid';coreMass.renderOrder=11;
  coreIds.forEach((i,k)=>{
   dummy.position.set(...voxelPosition(i,state.shape,offset,angle));dummy.position.y+=displayLift;dummy.rotation.y=angle;dummy.updateMatrix();coreMass.setMatrixAt(k,dummy.matrix);
   coreMass.setColorAt(k,view==='solid'?solid:coreColor);
  });
  coreMass.instanceMatrix.needsUpdate=true;if(coreMass.instanceColor)coreMass.instanceColor.needsUpdate=true;window.optimizerGroup.add(coreMass);
 }
 if(outlineRevision!==state.revision){
  if(coreShafts){
   window.optimizerGroup.remove(coreShafts);
   coreShafts.geometry?.dispose?.();coreShafts.material?.dispose?.();
   coreShafts=null;
  }
  if(Array.isArray(state.cores)&&state.cores.length){
   const cVerts=[];
   const pushEdge=(p0,p1)=>{cVerts.push(p0[0],p0[1],p0[2],p1[0],p1[1],p1[2]);};
   for(const c of state.cores){
    const corners=[[c.x0,c.y0],[c.x0+c.sizeX,c.y0],[c.x0+c.sizeX,c.y0+c.sizeY],[c.x0,c.y0+c.sizeY]]
     .map(([gx,gy])=>{const [wx,wy]=gridToWorld(gx,gy,offset,angle);return [wx,-wy];});
    const zBot=displayLift,zTop=c.floors*3+displayLift;
    for(let j=0;j<4;j++){
     const [xA,zA]=corners[j],[xB,zB]=corners[(j+1)%4];
     pushEdge([xA,zBot,zA],[xB,zBot,zB]);
     pushEdge([xA,zTop,zA],[xB,zTop,zB]);
     pushEdge([xA,zBot,zA],[xA,zTop,zA]);
    }
   }
   const shaftGeo=new THREE.BufferGeometry();
   shaftGeo.setAttribute('position',new THREE.Float32BufferAttribute(cVerts,3));
   const shaftMat=new THREE.LineBasicMaterial({color:0xdc2626,transparent:true,opacity:0.42,depthTest:false,depthWrite:false,toneMapped:false});
   coreShafts=new THREE.LineSegments(shaftGeo,shaftMat);
   coreShafts.name='Core shaft bounding edges';coreShafts.renderOrder=25;
   window.optimizerGroup.add(coreShafts);
  }
  if(outline){
   window.optimizerGroup.remove(outline);
   outline.traverse(obj=>{obj.geometry?.dispose?.();obj.material?.dispose?.();});
  }
  const started=performance.now(),cosA=Math.cos(angle),sinA=Math.sin(angle);
  const {positions:rawPos,faceA,faceB,concave}=voxelSilhouetteEdges(state.occupied,state.shape);
  let kept=0;for(let s=0;s<concave.length;s++)if(!concave[s])kept++;
  const allPositions=new Float32Array(rawPos.length),silPositions=new Float32Array(kept*6),normalsA=new Float32Array(kept*3),normalsB=new Float32Array(kept*3);
  for(let s=0,w=0;s<concave.length;s++){
   const pOff=s*6;
   for(let e=0;e<2;e++){
    const idx=pOff+e*3,[x,y]=gridToWorld(rawPos[idx],rawPos[idx+1],offset,angle),z=rawPos[idx+2],wy=z*3+displayLift;
    allPositions[idx]=x;allPositions[idx+1]=wy;allPositions[idx+2]=-y;
    if(!concave[s]){
     const dst=w*6+e*3;
     silPositions[dst]=x;silPositions[dst+1]=wy;silPositions[dst+2]=-y;
    }
   }
   if(!concave[s]){
    const wNorm=w*3;
    normalsA[wNorm]=cosA*faceA[pOff]-sinA*faceA[pOff+1];normalsA[wNorm+1]=faceA[pOff+2];normalsA[wNorm+2]=-(sinA*faceA[pOff]+cosA*faceA[pOff+1]);
    normalsB[wNorm]=cosA*faceB[pOff]-sinA*faceB[pOff+1];normalsB[wNorm+1]=faceB[pOff+2];normalsB[wNorm+2]=-(sinA*faceB[pOff]+cosA*faceB[pOff+1]);
    w++;
   }
  }
  const silLines=new LineSegmentsGeometry();
  silLines.setPositions(silPositions);
  silLines.setAttribute('instanceNormalA',new THREE.InstancedBufferAttribute(normalsA,3));
  silLines.setAttribute('instanceNormalB',new THREE.InstancedBufferAttribute(normalsB,3));
  const lineMaterial=new LineMaterial({color:outlineBlue,linewidth:1,worldUnits:true,transparent:true,opacity:1,depthTest:true,depthWrite:true,toneMapped:false,stencilWrite:true,stencilWriteMask:0x00,stencilFunc:THREE.NotEqualStencilFunc,stencilRef:1,stencilFuncMask:0xff,stencilFail:THREE.KeepStencilOp,stencilZFail:THREE.KeepStencilOp,stencilZPass:THREE.KeepStencilOp});
  lineMaterial.customProgramCacheKey=()=>'voxel-silhouette-1';
  lineMaterial.onBeforeCompile=shader=>{
   const attrAnchor='attribute vec3 instanceEnd;';
   const perspAnchor='bool perspective = ( projectionMatrix[ 2 ][ 3 ] == - 1.0 ); // 4th entry in the 3rd column';
   if(!shader.vertexShader.includes(attrAnchor)||!shader.vertexShader.includes(perspAnchor))throw new Error('Incompatible LineMaterial shader');
   shader.vertexShader=shader.vertexShader
    .replace(attrAnchor,attrAnchor+'\nattribute vec3 instanceNormalA;\nattribute vec3 instanceNormalB;')
    .replace(perspAnchor,perspAnchor+'\nvec3 silView = perspective ? normalize( - 0.5 * ( start.xyz + end.xyz ) ) : vec3( 0.0, 0.0, 1.0 );\nmat3 silNormMat = mat3( modelViewMatrix );\nfloat silDotA = dot( normalize( silNormMat * instanceNormalA ), silView );\nfloat silDotB = dot( normalize( silNormMat * instanceNormalB ), silView );\nif ( silDotA * silDotB > 1e-4 || ( abs( silDotA ) < 1e-4 && abs( silDotB ) < 1e-4 ) ) {\n gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 );\n return;\n}');
  };
  const silhouetteLines=new LineSegments2(silLines,lineMaterial);
  silhouetteLines.renderOrder=20;
  silhouetteLines.onBeforeRender=(renderer,_scene,camera)=>{
   renderer.getSize(lineMaterial.resolution);
   if(outlineWorldWidth===null&&silLines.boundingBox&&!silLines.boundingBox.isEmpty()){
    // Reference width: 3 px when the mass spans about 400 px, rather than
    // calibrating against the initial far-away view of the whole neighbourhood.
    const size=silLines.boundingBox.getSize(new THREE.Vector3());
    outlineWorldWidth=3*Math.max(size.x,size.y,size.z)/400;
   }
   if(outlineWorldWidth!==null)lineMaterial.linewidth=outlineWorldWidth*2;
  };
  const innerGeo=new LineSegmentsGeometry();
  innerGeo.setPositions(allPositions);
  const innerMaterial=new LineMaterial({color:outlineBlue,linewidth:1.7,worldUnits:false,transparent:true,opacity:0.65,depthTest:true,depthWrite:false,toneMapped:false,stencilWrite:true,stencilWriteMask:0x00,stencilFunc:THREE.EqualStencilFunc,stencilRef:1,stencilFuncMask:0xff,stencilFail:THREE.KeepStencilOp,stencilZFail:THREE.KeepStencilOp,stencilZPass:THREE.KeepStencilOp});
  const innerLines=new LineSegments2(innerGeo,innerMaterial);
  innerLines.renderOrder=15;
  innerLines.onBeforeRender=renderer=>{renderer.getSize(innerMaterial.resolution);};
  outline=new THREE.Group();
  outline.name='Silhouette and interior voxel outlines';
  outline.add(innerLines,silhouetteLines);
  outline.userData={segments:allPositions.length/6,silhouetteSegments:silPositions.length/6,buildMilliseconds:performance.now()-started};
  window.optimizerGroup.add(outline);outlineRevision=state.revision;
 }
 outline.visible=view==='solid';
}
window.addEventListener('message',event=>{
 if(event.source!==window.parent||event.origin!==location.origin)return;
 const message=event.data;if(!message||!['voxel_initialize','voxel_update','voxel_display'].includes(message.type))return;
 try{
  if(message.type==='voxel_initialize'&&!initialized){
   const contextData=sceneData(message.state);initUrbanContext(contextData);addContextTrees(contextData);prepareGroundPlanes();initialized=true;
   // Same native reset/projection messages used by Module Lab's ThreeViewer.
   window.postMessage({type:'set_camera_mode',mode:message.mode||'axonometric'},location.origin);
   window.postMessage({type:'reset_camera'},location.origin);
   window.parent.postMessage({type:'voxel_viewer_ready',engine:'Module Lab urban_viewer.js'},location.origin);
  }
  if(!initialized)return;
  displayContext(message.visible!==false,message.transparent===true);
  if(message.state)updateMassing(message.state,message.view||'solid');
 }catch(exc){const el=document.getElementById('viewer-error');el.textContent=exc.message;el.hidden=false;window.parent.postMessage({type:'voxel_viewer_error',message:exc.message},location.origin);}
});
