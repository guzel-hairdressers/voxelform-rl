// Host controls only. The iframe runs Module Lab's actual urban_viewer.js.
const $=id=>document.getElementById(id);
let data=null,lastRevision=-1,lastSite=null,view='solid',catalog=[],busy=false,frameReady=false,cameraMode='axonometric',lastSettingsKey=null;
const frame=$('viewport');
function error(message){$('error').textContent=message||'';$('error').hidden=!message;}
function option(value,label){const o=document.createElement('option');o.value=value;o.textContent=label;return o;}
function viewerMessage(message){frame.contentWindow?.postMessage(message,location.origin);}
function displayOptions(){return {visible:$('showContext').checked,transparent:$('transparentContext').checked};}
function updateContext(){frameReady=false;frame.src='/assets/viewer.html?renderer=outlines-hsl-4&site='+encodeURIComponent(data.context?.site_id||'synthetic-'+data.seed);}
function updateGeometry(){if(frameReady&&data)viewerMessage({type:'voxel_update',state:data,view,...displayOptions()});}
function fitViews(){if(frameReady)viewerMessage({type:'reset_camera'});}
function setProjection(mode){cameraMode=mode;viewerMessage({type:'set_camera_mode',mode});updateCameraUI();}
function updateCameraUI(){
 $('axono').setAttribute('aria-pressed',String(cameraMode==='axonometric'));
 $('perspective').setAttribute('aria-pressed',String(cameraMode==='perspective'));
 $('cameraStatus').textContent=cameraMode==='axonometric'?'Axonometric':'Perspective';
 frame.dataset.projection=cameraMode;frame.dataset.engine='Module Lab urban_viewer.js';
}
frame.onload=()=>{if(data?.ready)viewerMessage({type:'voxel_initialize',state:data,view,mode:cameraMode,...displayOptions()});};
window.addEventListener('message',event=>{
 if(event.source!==frame.contentWindow||event.origin!==location.origin)return;
 if(event.data?.type==='voxel_viewer_ready'){frameReady=true;updateGeometry();updateCameraUI();}
 if(event.data?.type==='voxel_viewer_error')error('3D viewer: '+event.data.message);
 if(event.data?.type==='camera_mode_change'){cameraMode=event.data.mode;updateCameraUI();}
});
document.addEventListener('visibilitychange',()=>{if(frameReady)viewerMessage({type:'set_visuals_enabled',enabled:!document.hidden});});
function canvasSize(canvas){
 const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);
 if(rect.width<1||rect.height<1)return null;
 const w=Math.round(rect.width*ratio),h=Math.round(rect.height*ratio);
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);
 return {ctx,width:rect.width,height:rect.height};
}
function metric(value,label,note){return `<div class="metric"><b>${value}</b><span>${label}</span><small>${note}</small></div>`;}
function drawHistory(){
 if(!data||!$('progress').open)return;
 const size=canvasSize($('history'));if(!size)return;
 const {ctx,width,height}=size;ctx.clearRect(0,0,width,height);ctx.font='8px system-ui';
 const rows=data.recent.filter(r=>Number.isFinite(r.far)&&Number.isFinite(r.targetFar));
 const maximum=Math.max(1,Math.ceil(Math.max(data.metrics.far||0,data.settings?.targetFar||3,...rows.flatMap(r=>[r.far,r.targetFar]))));
 for(let i=0;i<=4;i++){const value=i*maximum/4,y=height-22-(height-40)*i/4;ctx.strokeStyle='#eee';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(31,y);ctx.lineTo(width-31,y);ctx.stroke();ctx.fillStyle='#999';ctx.fillText(value.toFixed(1),3,y+3);ctx.fillText(value.toFixed(1),width-25,y+3);}
 if(!rows.length){ctx.fillStyle='#999';ctx.textAlign='center';ctx.fillText('Episodes appear here when a FAR condition ends the sequence.',width/2,height/2);ctx.textAlign='left';return;}
 for(const [key,color] of [['targetFar','#999'],['far','#a04476']]){
  ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;ctx.beginPath();let previous=null;
  rows.forEach((r,i)=>{const x=35+i*(width-70)/Math.max(1,rows.length-1),y=height-22-r[key]*(height-40)/maximum,identity=`${r.siteId??r.seed}:${r.targetFar}:${r.training}`;if(previous!==identity)ctx.moveTo(x,y);else ctx.lineTo(x,y);previous=identity;});ctx.stroke();
  rows.forEach((r,i)=>{const x=35+i*(width-70)/Math.max(1,rows.length-1),y=height-22-r[key]*(height-40)/maximum;ctx.beginPath();if(r.success===false){ctx.moveTo(x-3,y-3);ctx.lineTo(x+3,y+3);ctx.moveTo(x+3,y-3);ctx.lineTo(x-3,y+3);ctx.stroke();}else{ctx.arc(x,y,2,0,Math.PI*2);ctx.fill();}});
 }
}
function render(){
 if(!data?.ready)return;
 const m=data.metrics,l=data.learning,c=data.context,siteKey=c?`${c.site_id}:${c.gridAlignment||'world'}`:`synthetic:${data.seed}`,changedSite=siteKey!==lastSite;
 $('status').textContent=`${data.running?(data.training?'Training':'Generating'):'Paused'} · ${m.device.toUpperCase()} · ${data.activeSeconds.toFixed(1)} s active`;
 $('sourceLabel').textContent=c?'Fetched urban context':'Synthetic method fixture';
 $('train').textContent=data.running&&data.training?'Pause training':'▶ Train';$('train').setAttribute('aria-pressed',String(data.running&&data.training));
 $('infer').textContent=data.running&&!data.training?'Pause generation':'Generate one';
 const settings=data.settings||{targetFar:3,densityPreset:'medium',densityPresets:{low:1.5,medium:3,high:4.5}};
 const settingsKey=`${settings.targetFar}:${settings.densityPreset}:${settings.maskMode}:${settings.deepInterior}:${settings.maximumInteriorVoids}:${settings.maximumEditVoxels}:${settings.editAllocation}`;
 if(settingsKey!==lastSettingsKey){lastSettingsKey=settingsKey;$('targetFar').value=settings.targetFar;$('densityPreset').value=settings.densityPreset;$('maskMode').value=settings.maskMode||'independent';$('deepInterior').checked=!!settings.deepInterior;$('maxVoids').value=settings.maximumInteriorVoids??0;$('maxEditVoxels').value=settings.maximumEditVoxels??0;$('editAllocation').value=settings.editAllocation||'probability';}
 if(Number.isFinite(settings.maximumFar)){$('targetFar').max=settings.maximumFar;for(const o of $('densityPreset').options){o.disabled=o.value!=='custom'&&settings.densityPresets[o.value]>settings.maximumFar;}}
 $('farDefinition').textContent=`Gross occupied floor area / parcel area. Editable study targets. Envelope maximum FAR ${(settings.maximumFar||6).toFixed(3)}.`;
 $('patchRules').textContent=`Deep cuts ramp from 8 to 16 m from the facade. Interior openings must connect to outside air. Ordinary patches retain at least ${settings.minimumPatch??1} cells; the final FAR residual can be smaller. New interior openings require at least ${settings.minimumVoidArea??1} m² per floor. Opening limit 0 means no cap.`;
 const currentFar=m.far??m.occupiedFloorArea/(c?.siteArea||data.shape[1]*data.shape[2]);
 const auditNote=m.environmentalFieldsCurrent===false?`Last audited at mask ${m.environmentalAuditEdit}`:'Current environmental audit';
 $('metrics').innerHTML=metric(currentFar.toFixed(3),'FAR',`Target ${settings.targetFar.toFixed(3)}${settings.farTolerance?` ±${settings.farTolerance.toFixed(2)}`:''}${m.targetReached?' · reached':''}`)+metric(Math.round(m.occupiedFloorArea).toLocaleString()+' m²','Gross floor area',`${m.remainingVoxels??m.remainingEdits??'—'} voxels to FAR target`)+metric((100*m.potentialUsableFraction).toFixed(1)+'%','Potential usable',auditNote)+metric((100*m.directSunProxy).toFixed(1)+'%','Direct sun proxy',auditNote);
 $('studyTitle').textContent=c?c.city:'Synthetic '+data.plan+' study';
 $('buildingCount').textContent=c?`${c.buildings.length} fetched buildings`:`${data.boxes.length} synthetic boxes`;
 $('contextStatus').textContent=c?`${c.site_id} · ${c.tier} · ${c.buildings.length} neighbours`:`Synthetic plot · seed ${data.seed}`;
 const termination=data.termination==='far_reached'?' · FAR reached':data.termination==='no_admissible_edits'?' · No facade cells':data.termination==='mask_generation_failed'?' · Generated mask rejected':data.termination==='step_budget_exhausted'?' · Step budget exhausted · FAR not reached':'';
 $('episodeStatus').textContent=`${data.episodes} completed episode${data.episodes===1?'':'s'} · ${m.edits} ${m.edits===1?'mask':'masks'}${m.stepBudget!=null?` / ${m.stepBudget} budget`:''} · ${m.voxelsEdited??0} voxels changed · ${l.directMaskUpdates??0} mask-head updates${l.importedUpdates?` · ${l.importedUpdates} imported updates`:""}${termination}`;
 $('progressCount').textContent=`${data.recent.length} recorded result${data.recent.length===1?'':'s'}`;
 $('details').textContent=`Loss ${l.loss.toFixed(4)} · gradient ${l.gradientNorm.toFixed(4)}. Evaluation ${(m.evaluationSeconds*1000).toFixed(1)} ms · action ${(data.actionSeconds*1000).toFixed(1)} ms · learning ${(data.learningSeconds*1000).toFixed(1)} ms. ${m.newReceivers} new / ${m.receivers} exposed receivers. Seed ${data.seed}. Optimizer rollout: ${data.rolloutSteps||32} masks; this does not end the episode. Environmental fields audited every ${data.environmentAuditSteps||8} masks and at completion. Last audit: mask ${m.environmentalAuditEdit??0}. Target tolerance: ±${(m.farTolerance||0).toFixed(6)} FAR. Facade snapshot ${(1000*(m.facadeEvaluationSeconds||0)).toFixed(2)} ms · neural mask generation ${(1000*(data.neuralGenerationSeconds||0)).toFixed(2)} ms · constraint projection ${(1000*(m.maskProjectionSeconds||0)).toFixed(2)} ms. Last mask: ${m.lastChangedVoxels||0} applied / ${m.lastGeneratedVoxels||0} generated voxels. Direct mask-head updates: ${l.directMaskUpdates??0}; imported historical updates: ${l.importedUpdates??0}. Last mask: ${data.maskStructure?.components??'—'} patches; largest patch ${((data.maskStructure?.largestShare??0)*100).toFixed(0)}%. Deep floor area (>8 m): ${Math.round(m.deepFloorArea||0)} m².`;
 $('field').querySelector('option[value=noise]').disabled=!m.noiseAvailable;
 if(!m.noiseAvailable&&view==='noise'){view='solid';$('field').value=view;updateLayerLegend();lastRevision=-1;}
 if(changedSite){
  lastSite=siteKey;
  $('gridAlignment').replaceChildren(option('auto_fit','Auto'),option('world','World'),option('principal_major','Principal 1'),option('longest_edge','Longest edge'));
  $('gridAlignment').value=c?.gridAlignment||'world';$('applyGrid').disabled=!c;
  $('envelopeLabel').textContent=`${data.shape[0]}-floor envelope · 1 m resolution`;
  $('contextNote').textContent=c?`${c.buildings.length} original building meshes and ${c.roads.length} street meshes from the Module Lab OSM cache. Parcel rasterized at 1 m. Road traffic data is unavailable; noise is unmeasured.`:'Synthetic plot, surrounding boxes and line-source noise fixture. Not a fetched urban site.';
  updateContext();
 }
 if(lastRevision!==data.revision){lastRevision=data.revision;updateGeometry();}
 $('connection').textContent='Optimizer connected';
 if(data.error)error(data.error);
 drawHistory();
}
function updateLayerLegend(){
 const labels={solid:'Occupied volume',usable:'Potential usable area / depth proxy',light:'Low → high diffuse sky access',noise:'Low → high relative noise',depthBadness:'Depth: ≤8 m → ≥16 m'};
 $('layerLegend').className='layer-legend'+(['light','noise','depthBadness'].includes(view)?' heat':'');$('layerLegend').lastElementChild.textContent=labels[view];
 $('layerLegend').firstElementChild.style.background=view==='usable'?'#86b4ad':'';
}
async function control(command,extra={}){
 if(busy)return false;busy=true;error('');
 document.querySelectorAll('button').forEach(b=>b.disabled=true);
 $('notice').hidden=false;$('notice').textContent=command==='site'?'Preparing parcel, context and visibility caches…':'Applying command…';
 try{
  const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,...extra})});
  if(!r.ok){const text=await r.text();let message=text;try{message=JSON.parse(text).detail||text;}catch{}throw new Error(message);}
  await poll();return true;
 }catch(e){error(e.message);return false;}
 finally{busy=false;document.querySelectorAll('button').forEach(b=>b.disabled=false);$('notice').hidden=true;}
}
function filteredSites(){return catalog.filter(r=>(!$('city').value||r.cityCode===$('city').value)&&(!$('tier').value||r.tier===$('tier').value));}
function filterSites(preferred){
 const rows=filteredSites(),current=preferred||$('site').value;
 $('site').replaceChildren(...rows.map(r=>option(r.siteId,`${r.siteId} · ${Math.round(r.area||0).toLocaleString()} m²`)));
 if(rows.some(r=>r.siteId===current))$('site').value=current;
 $('catalogCount').textContent=`${rows.length} matching cached sites. Original footprints and heights.`;
 $('applySite').disabled=!rows.length;
}
async function loadSite(id=$('site').value){if(!id){error('No cached sites match these filters.');return;}const ok=await control('site',{source:'cached_osm',siteId:id});if(ok){$('source').value='cached_osm';sourceSettings();}}
function sourceSettings(){$('urbanSettings').hidden=$('source').value!=='cached_osm';$('syntheticSettings').hidden=$('source').value!=='synthetic';}
async function navigateSite(direction){
 const rows=filteredSites();if(!rows.length)return;
 const current=rows.findIndex(r=>r.siteId===data?.siteId),next=rows[(current+direction+rows.length)%rows.length];$('site').value=next.siteId;await loadSite(next.siteId);
}
$('train').onclick=()=>control(data?.running&&data.training?'pause':'run',{training:true});$('infer').onclick=()=>control(data?.running&&!data.training?'pause':'run',{training:false});$('step').onclick=()=>control('step',{training:false});$('reset').onclick=()=>control('reset');
$('new').onclick=()=>navigateSite(1);$('previous').onclick=()=>navigateSite(-1);$('applySite').onclick=()=>loadSite();$('applySynthetic').onclick=()=>control('site',{source:'synthetic',plan:$('plan').value});
$('densityPreset').onchange=()=>{const presets=data?.settings?.densityPresets||{low:1.5,medium:3,high:4.5};if($('densityPreset').value!=='custom')$('targetFar').value=presets[$('densityPreset').value];};
$('targetFar').oninput=()=>{$('densityPreset').value='custom';};
$('applyFar').onclick=()=>control('settings',{targetFar:Number($('targetFar').value),densityPreset:$('densityPreset').value});
$('applyMasks').onclick=()=>control('masks',{mode:$('maskMode').value,deepInterior:$('deepInterior').checked,maximumInteriorVoids:Number($('maxVoids').value),maximumEditVoxels:Number($('maxEditVoxels').value),editAllocation:$('editAllocation').value});
$('source').onchange=sourceSettings;$('city').onchange=()=>filterSites();$('tier').onchange=()=>filterSites();
$('field').onchange=()=>{view=$('field').value;updateLayerLegend();updateGeometry();};
$('showContext').onchange=()=>{if(frameReady)viewerMessage({type:'voxel_display',...displayOptions()});};$('fit').onclick=fitViews;
$('transparentContext').onchange=()=>{if(frameReady)viewerMessage({type:'voxel_display',...displayOptions()});};
$('axono').onclick=()=>setProjection('axonometric');$('perspective').onclick=()=>setProjection('perspective');
$('progress').ontoggle=drawHistory;
$('load').onchange=async()=>{const file=$('load').files[0];if(!file)return;error('');try{const r=await fetch('/api/policy',{method:'POST',body:await file.arrayBuffer()});if(!r.ok)throw new Error(await r.text());await poll();}catch(e){error(e.message);}finally{$('load').value='';}};
let polling=false;
async function poll(){
 if(polling)return;polling=true;
 try{const r=await fetch('/api/state');if(!r.ok)throw new Error('State unavailable');data=await r.json();if(data.error)error(data.error);if(data.ready)render();else $('status').textContent=data.error?'Context unavailable':'Preparing parcel and visibility caches…';}
 catch(e){$('status').textContent='Connecting…';$('connection').textContent='Optimizer disconnected';}
 finally{polling=false;}
}
async function init(){
 await poll();
 try{const r=await fetch('/api/sites');if(!r.ok)throw new Error('Cached-site catalog unavailable. Open the updated backend.');catalog=await r.json();
 const cities=new Map(catalog.map(r=>[r.cityCode,r.city]));$('city').replaceChildren(option('','All cities'),...Array.from(cities,([code,name])=>option(code,name)));
 filterSites(data?.siteId||'m_nyc_0003');
 }catch(e){error(e.message);}
}
new ResizeObserver(drawHistory).observe($('history'));
setInterval(poll,700);init();
document.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea,button,summary,a')||e.metaKey||e.ctrlKey||e.altKey||e.repeat)return;if(e.code==='Space'){e.preventDefault();control(data?.running?'pause':'run',{training:true});}if(e.key==='0')fitViews();});

$('applyGrid').onclick=()=>control('grid',{alignment:$('gridAlignment').value});
