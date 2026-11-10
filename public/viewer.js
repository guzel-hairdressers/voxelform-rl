import * as THREE from 'three';
import {LineSegments2} from 'three/addons/lines/LineSegments2.js';
import {LineSegmentsGeometry} from 'three/addons/lines/LineSegmentsGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';
import {initUrbanContext} from '/shared-viewer/scripts/urban_viewer.js';
import {sceneData,sceneOffset,voxelPosition,gridToWorld} from './viewer-data.js?v=grid-1';
import {treePositions} from './context-trees.js?v=1';
import {voxelContours} from './voxel-contours.js?v=2';

let initialized=false,mass=null,lastRevision=-1,lastView=null,geometry=null,outline=null,outlineRevision=-1,outlineWorldWidth=null;
const dummy=new THREE.Object3D();
const blueHsl=new THREE.Color('#1645ff').getHSL({},THREE.SRGBColorSpace),outlineBlue=new THREE.Color().setHSL(blueHsl.h,blueHsl.s,.50,THREE.SRGBColorSpace);
const lightBlue=new THREE.Color().setHSL(blueHsl.h,blueHsl.s,.95,THREE.SRGBColorSpace).convertLinearToSRGB(),shadeBlue=new THREE.Color().setHSL(blueHsl.h,blueHsl.s,.70,THREE.SRGBColorSpace).convertLinearToSRGB();
const solid=new THREE.Color('#ffffff'),usable=new THREE.Color('#59a59d'),unusable=new THREE.Color('#d4dce0');
function heat(t){return new THREE.Color().setHSL(.65-Math.max(0,Math.min(1,t))*.61,.58,.54);}
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
function displayContext(visible,transparent){
 const context=window.scene?.getObjectByName('surroundingContext');if(!context)return;
 context.visible=visible;
 for(const group of context.children){
  if(!group.children.some(child=>child.userData?.isBuilding))continue;
  group.traverse(object=>{
   if(object.userData?.isBuilding){object.material.transparent=transparent;object.material.opacity=transparent ? .28 : 1;object.material.depthWrite=!transparent;object.material.needsUpdate=true;}
   if(object.isLineSegments){object.material.transparent=transparent;object.material.opacity=transparent ? .4 : 1;object.material.needsUpdate=true;}
  });
 }
}
function updateMassing(state,view){
 if(lastRevision===state.revision&&lastView===view)return;
 lastRevision=state.revision;lastView=view;
 if(mass){window.optimizerGroup.remove(mass);mass.dispose?.();mass.material.dispose();}
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
 const ids=[];for(let i=0;i<state.occupied.length;i++)if((view==='light'||view==='noise')?state.allowed[i]:state.occupied[i])ids.push(i);
 const material=view==='solid'
  ? new THREE.MeshLambertMaterial({flatShading:true,vertexColors:true,color:0xffffff,toneMapped:false,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1})
  : new THREE.MeshBasicMaterial({vertexColors:true,transparent:view==='light',opacity:view==='light' ? .72 : 1});
 if(view==='solid')material.onBeforeCompile=shader=>{
  // Keep hue/saturation fixed in sRGB HSL; actual shared-engine illumination
  // and shadows move lightness between the requested 70% and 95% endpoints.
  shader.uniforms.lightInk={value:lightBlue};shader.uniforms.shadeInk={value:shadeBlue};
  shader.fragmentShader='uniform vec3 lightInk; uniform vec3 shadeInk;\n'+shader.fragmentShader;
  shader.fragmentShader=shader.fragmentShader.replace('vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;','float illumination = clamp(max(max(reflectedLight.directDiffuse.r,reflectedLight.directDiffuse.g),reflectedLight.directDiffuse.b)*3.14159265/0.82,0.,1.);\n vec3 ink = mix(shadeInk,lightInk,illumination);\n vec3 outgoingLight = pow((ink+0.055)/1.055,vec3(2.4));');
 };
 mass=new THREE.InstancedMesh(geometry,material,ids.length);
 mass.name='VoxelForm massing';mass.castShadow=true;mass.receiveShadow=view==='solid';
 const offset=sceneOffset(state),displayLift=Math.max(.2,...(state.context?.site?.vertices||[]).map(v=>v[2]))+.03;
 ids.forEach((i,k)=>{
  dummy.position.set(...voxelPosition(i,state.shape,offset,state.context?.gridAngle||0));dummy.position.y+=displayLift;dummy.rotation.y=state.context?.gridAngle||0;dummy.updateMatrix();mass.setMatrixAt(k,dummy.matrix);
  let color=view==='solid'?solid:view==='usable'?(state.usable[i]?usable:unusable):heat(state[view][i]);
  if(view==='light'&&state.occupied[i])color=unusable;mass.setColorAt(k,color);
 });
 mass.instanceMatrix.needsUpdate=true;if(mass.instanceColor)mass.instanceColor.needsUpdate=true;window.optimizerGroup.add(mass);
 if(outlineRevision!==state.revision){
  if(outline){window.optimizerGroup.remove(outline);outline.geometry.dispose();outline.material.dispose();}
  const started=performance.now(),positions=voxelContours(state.occupied,state.shape);
  for(let i=0;i<positions.length;i+=3){const [x,y]=gridToWorld(positions[i],positions[i+1],offset,state.context?.gridAngle||0),z=positions[i+2];positions[i]=x;positions[i+1]=z*3+displayLift;positions[i+2]=-y;}
  const lines=new LineSegmentsGeometry();lines.setPositions(positions);
  const lineMaterial=new LineMaterial({color:outlineBlue,linewidth:1,worldUnits:true,depthWrite:false,toneMapped:false});
  outline=new LineSegments2(lines,lineMaterial);
  outline.onBeforeRender=(renderer,_scene,camera)=>{
   renderer.getSize(lineMaterial.resolution);
   if(outlineWorldWidth===null){
    // Reference width: 3 px when the mass spans about 400 px, rather than
    // calibrating against the initial far-away view of the whole neighbourhood.
    const size=lines.boundingBox.getSize(new THREE.Vector3());
    outlineWorldWidth=3*Math.max(size.x,size.y,size.z)/400;
   }
   lineMaterial.linewidth=outlineWorldWidth;
  };
  outline.name='Merged voxel outlines';outline.userData={segments:positions.length/6,buildMilliseconds:performance.now()-started};
  window.optimizerGroup.add(outline);outlineRevision=state.revision;
 }
 outline.visible=view==='solid';
}
window.addEventListener('message',event=>{
 if(event.source!==window.parent||event.origin!==location.origin)return;
 const message=event.data;if(!message||!['voxel_initialize','voxel_update','voxel_display'].includes(message.type))return;
 try{
  if(message.type==='voxel_initialize'&&!initialized){
   const contextData=sceneData(message.state);initUrbanContext(contextData);addContextTrees(contextData);initialized=true;
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
