import {THREE,OrbitControls} from './vendor/three.min.js';
const $=id=>document.getElementById(id);let data=null,lastRevision=-1,view='solid';
const scene=new THREE.Scene();scene.background=new THREE.Color('#eef0eb');
let renderer=null;try{renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('viewport').appendChild(renderer.domElement);}catch(e){$('viewport').textContent='3D is unavailable in this browser. The floor plan and training controls remain available.';}
const camera=new THREE.PerspectiveCamera(42,1,.1,2000);camera.position.set(57,45,60);
const controls=renderer?new OrbitControls(camera,renderer.domElement):null;if(controls){controls.target.set(16,6,12);controls.enableDamping=true;}
scene.add(new THREE.HemisphereLight(0xffffff,0x8c9784,2.5));const sun=new THREE.DirectionalLight(0xffffff,2);sun.position.set(-20,60,30);scene.add(sun);
const grid=new THREE.GridHelper(140,70,0xc1cab9,0xdce1d6);grid.position.set(16,-.1,12);scene.add(grid);
let mass=null,context=null;const geometry=new THREE.BoxGeometry(.98,2.98,.98);const dummy=new THREE.Object3D();
const colors={solid:new THREE.Color('#b9c6a5'),usable:new THREE.Color('#76aa86')};
function heat(t){return new THREE.Color().setHSL(.65-Math.max(0,Math.min(1,t))*.61,.58,.54);}
function updateGeometry(){
 if(!data)return;const [nz,ny,nx]=data.shape;
 if(mass){scene.remove(mass);mass.material.dispose();}
 const ids=[];for(let i=0;i<data.occupied.length;i++)if((view==='light'||view==='noise')?data.allowed[i]:data.occupied[i])ids.push(i);
 mass=new THREE.InstancedMesh(geometry,new THREE.MeshStandardMaterial({roughness:.8,metalness:0,transparent:view==='light',opacity:view==='light'?.72:1}),ids.length);
 ids.forEach((i,k)=>{const z=Math.floor(i/(nx*ny)),y=Math.floor(i/nx)%ny,x=i%nx;dummy.position.set(x+.5,(z+.5)*3,y+.5);dummy.updateMatrix();mass.setMatrixAt(k,dummy.matrix);
 let c=view==='solid'?colors.solid:view==='usable'?(data.usable[i]?colors.usable:new THREE.Color('#b9bdb7')):heat(data[view][i]);if(view==='light'&&data.occupied[i])c=new THREE.Color('#b7bcb0');mass.setColorAt(k,c);});
 mass.instanceMatrix.needsUpdate=true;scene.add(mass);
 if(context){scene.remove(context);context.geometry.dispose();context.material.dispose();}
 context=new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshStandardMaterial({color:0xc8cbc4,roughness:1,transparent:true,opacity:.35}),data.boxes.length);
 data.boxes.forEach((b,k)=>{dummy.position.set(b[0],b[2],b[1]);dummy.scale.set(b[3]*2,b[5]*2,b[4]*2);dummy.rotation.set(0,-Math.atan2(b[7],b[6]),0);dummy.updateMatrix();context.setMatrixAt(k,dummy.matrix);});
 dummy.scale.set(1,1,1);dummy.rotation.set(0,0,0);scene.add(context);drawPlan();
}
function drawPlan(){if(!data)return;const c=$('planCanvas'),ctx=c.getContext('2d'),[nz,ny,nx]=data.shape,f=Math.min(nz-1,+$('floor').value),s=Math.min((c.width-40)/nx,(c.height-40)/ny),ox=(c.width-nx*s)/2,oy=(c.height-ny*s)/2;ctx.clearRect(0,0,c.width,c.height);$('floor').max=nz-1;$('floorLabel').textContent=f+1;
 for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const i=f*nx*ny+y*nx+x;if(!data.allowed[i])continue;let color='#f0f0e9';if(view==='light'||view==='noise')color=(view==='light'&&data.occupied[i])?'#b7bcb0':'#'+heat(data[view][i]).getHexString();else if(data.occupied[i])color=view==='usable'?(data.usable[i]?'#76aa86':'#b9bdb7'):'#b9c6a5';ctx.fillStyle=color;ctx.fillRect(ox+x*s,oy+(ny-y-1)*s,s+.2,s+.2);}ctx.fillStyle='#647365';ctx.font='16px system-ui';ctx.fillText('1 m cells · floor '+(f+1),18,24);
}
function metric(value,label){return `<div class="metric"><b>${value}</b><span>${label}</span></div>`;}
function render(){if(!data?.ready)return;const m=data.metrics,l=data.learning;
 $('status').textContent=`${data.running?(data.training?'Training':'Inference'):'Paused'} · ${m.device.toUpperCase()} · ${data.activeSeconds.toFixed(1)} s active`;
 $('metrics').innerHTML=metric((100*m.retention).toFixed(1)+'%','Retained floor area')+metric((100*m.potentialUsableFraction).toFixed(1)+'%','Potential usable / original area')+metric((100*m.directSunProxy).toFixed(1)+'%','Sampled direct-sun proxy')+metric((100*m.roofOpportunity).toFixed(1)+'%','Sunny roof / plot area');
 $('details').textContent=`Episode ${data.episodes} · edit ${data.step}/${data.maxSteps} · optimizer updates ${l.updates} · loss ${l.loss.toFixed(4)} · gradient norm ${l.gradientNorm.toFixed(4)}. Evaluation ${(m.evaluationSeconds*1000).toFixed(1)} ms · action ${(data.actionSeconds*1000).toFixed(1)} ms · backprop ${(data.learningSeconds*1000).toFixed(1)} ms. ${m.newReceivers} new / ${m.receivers} exposed receivers. Seed ${data.seed}. Sun, sky access and noise are separate uncalibrated proxies.`;
 const c=$('history'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.font='13px system-ui';ctx.fillStyle='#647365';ctx.fillText('Episode metrics: retained area (grey), potential usable area (green)',15,20);
 for(const [key,color] of [['retention','#a4aba0'],['potentialUsableFraction','#497b59']]){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();data.recent.forEach((r,i)=>{const x=20+i*(c.width-40)/Math.max(20,data.recent.length-1),y=120-r[key]*85;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
 $('error').textContent=data.error||'';
 if(lastRevision!==data.revision){lastRevision=data.revision;updateGeometry();}
}
async function control(command,extra={}){try{const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,...extra})});if(!r.ok)throw new Error(await r.text());await poll();}catch(e){$('error').textContent=e.message;}}
$('train').onclick=()=>control('run',{training:true});$('infer').onclick=()=>control('run',{training:false});$('pause').onclick=()=>control('pause');$('step').onclick=()=>control('step',{training:false});$('reset').onclick=()=>control('reset');$('new').onclick=()=>control('site',{plan:$('plan').value});$('field').onchange=()=>{view=$('field').value;updateGeometry();};$('floor').oninput=drawPlan;
$('load').onchange=async()=>{const file=$('load').files[0];if(!file)return;const r=await fetch('/api/policy',{method:'POST',body:await file.arrayBuffer()});if(!r.ok)$('error').textContent=await r.text();await poll();};
let polling=false;async function poll(){if(polling)return;polling=true;try{const r=await fetch('/api/state');data=await r.json();if(data.error)$('error').textContent=data.error;if(data.ready)render();else $('status').textContent='Preparing context and ray caches… '+data.device;}catch(e){$('status').textContent='Connecting…';}finally{polling=false;}}setInterval(poll,700);poll();
function animate(){requestAnimationFrame(animate);if(!renderer)return;const el=$('viewport');if(renderer.domElement.width!==el.clientWidth*renderer.getPixelRatio()||renderer.domElement.height!==el.clientHeight*renderer.getPixelRatio()){renderer.setSize(el.clientWidth,el.clientHeight);camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();}controls.update();renderer.render(scene,camera);}animate();
