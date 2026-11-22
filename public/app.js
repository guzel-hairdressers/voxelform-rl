// Host controls only. The iframe runs Module Lab's actual urban_viewer.js.
const $=id=>document.getElementById(id);
let data=null,lastGeomRevision=-1,lastRenderedEpisode=-1,lastGeomAt=0,lastSite=null,view='solid',catalog=[],busy=false,frameReady=false,cameraMode='axonometric',lastSettingsKey=null,polling=false,pendingPoll=false;
let savedPreview = null;
let archiveOptionsKey = '';
let requestedLayout = null;
let trainingModeLoaded = false;
const frame=$('viewport');
function error(message){$('error').textContent=message||'';$('error').hidden=!message;}
function option(value,label){const o=document.createElement('option');o.value=value;o.textContent=label;return o;}
function viewerMessage(message){frame.contentWindow?.postMessage(message,location.origin);}
function displayOptions(){return {visible:$('showContext').checked,transparent:$('transparentContext').checked};}
function updateContext(){frameReady=false;frame.src='/assets/viewer.html?renderer=outlines-hsl-13&site='+encodeURIComponent(data.context?.site_id||'synthetic-'+data.seed);}
function displayedGeometry() {
 if (!savedPreview) return {state: data, view};
 return {
  state: {...data, ...savedPreview.geometry, revision: savedPreview.geometry.layout.id},
  view: 'solid',
 };
}
function updateGeometry(){
 if (!frameReady || !data) return;
 const shown = displayedGeometry();
 if (shown.view === 'solid' || shown.state[shown.view] != null) {
  viewerMessage({type:'voxel_update', ...shown, ...displayOptions()});
 }
}
function fitViews(){if(frameReady)viewerMessage({type:'reset_camera'});}
function setProjection(mode){cameraMode=mode;viewerMessage({type:'set_camera_mode',mode});updateCameraUI();}
function updateCameraUI(){
 $('axono').setAttribute('aria-pressed',String(cameraMode==='axonometric'));
 $('perspective').setAttribute('aria-pressed',String(cameraMode==='perspective'));
 $('cameraStatus').textContent=cameraMode==='axonometric'?'Axonometric':'Perspective';
 frame.dataset.projection=cameraMode;frame.dataset.engine='Module Lab urban_viewer.js';
}
frame.onload=()=>{if(data?.ready)viewerMessage({type:'voxel_initialize',...displayedGeometry(),mode:cameraMode,...displayOptions()});};
window.addEventListener('message',event=>{
 if(event.source!==frame.contentWindow||event.origin!==location.origin)return;
 if(event.data?.type==='voxel_viewer_ready'){frameReady=true;if(data){lastGeomRevision=data.revision;lastRenderedEpisode=data.episodes;lastGeomAt=performance.now();}updateGeometry();updateCameraUI();}
 if(event.data?.type==='voxel_viewer_error')error('3D viewer: '+event.data.message);
 if(event.data?.type==='camera_mode_change'){cameraMode=event.data.mode;updateCameraUI();}
});
document.addEventListener('visibilitychange',()=>{if(frameReady)viewerMessage({type:'set_visuals_enabled',enabled:!document.hidden});});
function canvasSize(canvas){
 if(!canvas)return null;
 const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);
 if(rect.width<1||rect.height<1)return null;
 const w=Math.round(rect.width*ratio),h=Math.round(rect.height*ratio);
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);
 return {ctx,width:rect.width,height:rect.height};
}
function escapeAttr(s){return String(s??'').replace(/&/g,'&amp;').replace(/"/g,'&quot;');}
function metric(value,label,note){return `<div class="metric"><b>${value}</b><span>${label}</span><small title="${escapeAttr(note)}">${note}</small></div>`;}
function kpiCard(label,value,note,cls=''){return `<div class="dash-kpi"><span>${label}</span><b>${value}</b><small class="${cls}" title="${escapeAttr(note)}">${note}</small></div>`;}
let chartHistory = [];
let chartScope = null;
let chartEpisode = null;
let chartRequest = null;
const legendFocus = Object.fromEntries(['quality', 'far'].map(chart =>
 [chart, {hover: null, focus: null, pinned: null}]));

function activeSeries(chart) {
 const state = legendFocus[chart];
 return state.hover || state.focus || state.pinned;
}
function seriesOpacity(chart, series) {
 const active = activeSeries(chart);
 return !active || active === series ? 1 : 0.15;
}
function showLegendFocus(chart) {
 const legend = document.querySelector(`.dash-legend[data-chart="${chart}"]`);
 for (const button of legend.querySelectorAll('button')) {
  button.classList.toggle('dimmed', seriesOpacity(chart, button.dataset.series) < 1);
  button.setAttribute('aria-pressed', String(button.dataset.series === legendFocus[chart].pinned));
 }
 if (data?.ready) {
  if (chart === 'quality') drawQualityChart();
  else drawHistory();
 }
}
for (const button of document.querySelectorAll('[data-series]')) {
 const chart = button.closest('[data-chart]').dataset.chart;
 const state = legendFocus[chart];
 const series = button.dataset.series;
 for (const [event, key, next] of [
  ['pointerenter', 'hover', series], ['pointerleave', 'hover', null],
  ['focus', 'focus', series], ['blur', 'focus', null],
 ]) {
  button.addEventListener(event, () => {
   state[key] = next;
   showLegendFocus(chart);
  });
 }
 button.addEventListener('click', () => {
  state.pinned = state.pinned === series ? null : series;
  showLegendFocus(chart);
 });
}
function chartRows() {
 if (chartScope && chartScope === data.layoutMemory?.scope) return chartHistory;
 return (data.recent || []).filter(row =>
  row.siteId === data.siteId && row.plan === data.plan && row.targetFar === data.settings.targetFar);
}
async function refreshChartHistory() {
 const scope = data.layoutMemory?.scope;
 if (!scope || !$('progress').open || chartRequest) return;
 const episode = data.episodes;
 if (chartScope === scope && chartEpisode === episode) return;
 const request = {scope, episode};
 chartRequest = request;
 const sameScope = chartScope === scope;
 const after = sameScope ? chartHistory.at(-1)?.id ?? 0 : 0;
 try {
  const response = await fetch(`/api/episodes/${encodeURIComponent(scope)}?after=${after}`);
  if (!response.ok || scope !== data?.layoutMemory?.scope) return;
  const {rows} = await response.json();
  if (scope !== data?.layoutMemory?.scope) return;
  chartHistory = [...(sameScope ? chartHistory : []), ...rows].slice(-5000);
  chartScope = scope;
  chartEpisode = episode;
  drawDashboard();
 } catch {
  // The live tail supplies the charts while a history request is unavailable.
 } finally {
  chartRequest = null;
 }
}

function drawQualityChart(){
 const size=canvasSize($('qualityChart'));if(!size)return;
 const {ctx,width,height}=size;ctx.globalAlpha=1;ctx.clearRect(0,0,width,height);ctx.font='8px system-ui';
 const rows = chartRows().filter(r => Number.isFinite(r.organicObjective) || Number.isFinite(r.directSunProxy));
 const allVals=rows.flatMap(r=>[r.organicObjective,r.directSunProxy,r.diffuseFacadeProxy,r.usableGoodFraction,r.coreScore,r.longNarrowFraction]).filter(Number.isFinite);
 const lo=allVals.length?Math.min(0,Math.floor(Math.min(...allVals)*10)/10):0;
 const hi=allVals.length?Math.max(1,Math.ceil(Math.max(...allVals)*10)/10):1;
 const span=Math.max(1e-6,hi-lo);
 for(let i=0;i<=4;i++){const val=lo+span*i/4,y=height-18-(height-30)*i/4;ctx.strokeStyle='#ececec';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(30,y);ctx.lineTo(width-10,y);ctx.stroke();ctx.fillStyle='#999';ctx.fillText(val.toFixed(2),2,y+3);}
 if(!rows.length){ctx.fillStyle='#999';ctx.textAlign='center';ctx.fillText('Run training or Generate one to populate learning curves.',width/2,height/2);ctx.textAlign='left';return;}
 const xAt=i=>34+i*(width-48)/Math.max(1,rows.length-1);
 const yAt=v=>height-18-((v-lo)/span)*(height-30);
 const series=[['longNarrowFraction','#9333ea',1.2],['coreScore','#dc2626',1.4],['usableGoodFraction','#0f766e',1.3],['directSunProxy','#d97706',1.3],['diffuseFacadeProxy','#0891b2',1.3],['organicObjective','#1645ff',1.8]];
 for(const [key,color,lw] of series){
  ctx.globalAlpha=seriesOpacity('quality',key);ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.beginPath();let started=false;
  rows.forEach((r,i)=>{if(!Number.isFinite(r[key])){started=false;return;}const x=xAt(i),y=yAt(r[key]);if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);});
  ctx.stroke();
 }
 const objs=rows.map(r=>r.organicObjective);
 if(objs.filter(Number.isFinite).length>=2){
  ctx.save();ctx.globalAlpha=seriesOpacity('quality','objectiveAverage');ctx.setLineDash([4,3]);ctx.strokeStyle='#1645ff';ctx.lineWidth=1.5;ctx.beginPath();let started=false;
  objs.forEach((_,i)=>{const win=objs.slice(Math.max(0,i-4),i+1).filter(Number.isFinite);if(!win.length){started=false;return;}const avg=win.reduce((a,b)=>a+b,0)/win.length,x=xAt(i),y=yAt(avg);if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);});
  ctx.stroke();ctx.restore();
 }
}
function drawHistory(){
 if(!data||!$('progress').open)return;
 const size=canvasSize($('history'));if(!size)return;
 const {ctx,width,height}=size;ctx.globalAlpha=1;ctx.clearRect(0,0,width,height);ctx.font='8px system-ui';
 const rows=chartRows().filter(r=>Number.isFinite(r.far)&&Number.isFinite(r.targetFar));
 const maximum=Math.max(1,Math.ceil(Math.max(data.metrics.far||0,data.settings?.targetFar||3,...rows.flatMap(r=>[r.far,r.targetFar]))));
 for(let i=0;i<=4;i++){const value=i*maximum/4,y=height-18-(height-30)*i/4;ctx.strokeStyle='#ececec';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(30,y);ctx.lineTo(width-30,y);ctx.stroke();ctx.fillStyle='#999';ctx.fillText(value.toFixed(1),3,y+3);ctx.fillText(value.toFixed(1),width-24,y+3);}
 if(!rows.length){ctx.fillStyle='#999';ctx.textAlign='center';ctx.fillText('Episodes appear here as training or generation completes.',width/2,height/2);ctx.textAlign='left';return;}
 const maxSec=Math.max(0.5,...rows.map(r=>r.episodeSeconds||0));
 const barW=Math.max(1,Math.min(8,(width-68)/Math.max(1,rows.length)*0.6));
 ctx.globalAlpha=seriesOpacity('far','episodeSeconds');
 ctx.fillStyle='rgba(148,163,184,.55)';
 rows.forEach((r,i)=>{if(!Number.isFinite(r.episodeSeconds))return;const x=34+i*(width-68)/Math.max(1,rows.length-1)-barW/2,h=(r.episodeSeconds/maxSec)*(height-30)*0.45;ctx.fillRect(x,height-18-h,barW,h);});
 for(const [key,color] of [['targetFar','#999'],['far','#a04476']]){
  ctx.globalAlpha=seriesOpacity('far',key);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;ctx.beginPath();let previous=null;
  rows.forEach((r,i)=>{const x=34+i*(width-68)/Math.max(1,rows.length-1),y=height-18-r[key]*(height-30)/maximum,identity=`${r.siteId??r.seed}:${r.targetFar}:${r.training}`;if(previous!==identity)ctx.moveTo(x,y);else ctx.lineTo(x,y);previous=identity;});ctx.stroke();
  rows.forEach((r,i)=>{const x=34+i*(width-68)/Math.max(1,rows.length-1),y=height-18-r[key]*(height-30)/maximum;ctx.beginPath();if(r.success===false){ctx.moveTo(x-3,y-3);ctx.lineTo(x+3,y+3);ctx.moveTo(x+3,y-3);ctx.lineTo(x-3,y+3);ctx.stroke();}else{ctx.arc(x,y,2,0,Math.PI*2);ctx.fill();}});
 }
}
function drawDashboard(){
 if(!data||!$('progress')?.open)return;
 const m=data.metrics,l=data.learning||{},d=data.dashboard||{};
 const score=Number.isFinite(m.organicObjective)?m.organicObjective:(m.quality??0);
 const base=Number.isFinite(d.baselineObjective)?d.baselineObjective:score;
 const delta=score-base;
 const deltaStr=(delta>=0?'+':'')+delta.toFixed(3)+' vs start';
 const goodArea=m.usableGoodArea??m.potentialUsableArea??0;
 const goodPct=100*(m.usableGoodFraction??m.potentialUsableFraction??0);
 const sunPct=100*(m.directSunProxy||0),skyPct=100*(m.diffuseFacadeProxy||0);
 const epSec=d.meanEpisodeSeconds,epMin=d.episodesPerMinute;
 const maxCoreD=data.settings?.maxCoreDist??m.maxCoreDist??12;
 $('dashKpis').innerHTML=
  kpiCard('Layout score',score.toFixed(3),`${deltaStr} · Best ${Number.isFinite(d.bestObjective)?d.bestObjective.toFixed(3):'—'}${d.bestEpisode?` (#${d.bestEpisode})`:''}`,delta>=0.005?'pos':delta<=-0.01?'neg':'')+
  kpiCard('Usable good area',Math.round(goodArea).toLocaleString()+' m²',`${goodPct.toFixed(1)}% GFA · Best ${Math.round(d.bestUsableGoodArea??goodArea).toLocaleString()} m²`)+
  kpiCard('Direct sun · Sky',`${sunPct.toFixed(1)}% · ${skyPct.toFixed(1)}%`,`Sun ${sunPct.toFixed(1)}% · Diffuse sky ${skyPct.toFixed(1)}%`)+
  kpiCard('Health & cores',`Slim ${m.slimVoxels??0} · Core ${m.coreCount??0} (${(100*(m.coreRatio||0)).toFixed(1)}%)`,`Max ${(m.coreDistanceMax??0).toFixed(1)}/${maxCoreD} vox · Over ${(100*(m.coreUncoveredFraction||0)).toFixed(1)}% · <6m ${(100*(m.longNarrowFraction||0)).toFixed(1)}%`,((m.slimVoxels??0)>0||(m.coreUncoveredFraction||0)>0.05)?'neg':'pos')+
  kpiCard('RL policy',`${l.updates??0} upd · ${data.episodes} ep`,`Loss ${(l.loss??0).toFixed(3)} · |g| ${(l.gradientNorm??0).toFixed(2)} · Op ${data.maskStructure?.lastOperation??'—'}`)+
  kpiCard('Throughput',epMin?`${epMin.toFixed(1)} ep/min`:`${Math.round((data.actionSeconds||0)*1000)} ms/act`,`${epSec?epSec.toFixed(2)+' s/ep · ':''}${Math.round((m.evaluationSeconds||0)*1000)} ms/eval · ${(m.device||'cpu').toUpperCase()}`);
 drawQualityChart();
 drawHistory();
 const tb=$('episodeTable');
 if(tb){
  const recent=[...(data.recent||[])].slice(-12).reverse();
  if(!recent.length){tb.innerHTML='<tr><td colspan="12" class="empty-row">No completed episodes recorded yet. Press ▶ Train or Generate one.</td></tr>';}
  else{
   const bestEp=d.bestEpisode;
   tb.innerHTML=recent.map(r=>{
    const cls=r.episode===bestEp?'best-row':'';
    const sc=Number.isFinite(r.organicObjective)?r.organicObjective.toFixed(3):'—';
    const ga=Number.isFinite(r.usableGoodArea)?Math.round(r.usableGoodArea).toLocaleString():'—';
    const su=Number.isFinite(r.directSunProxy)?(100*r.directSunProxy).toFixed(1)+'%':'—';
    const sk=Number.isFinite(r.diffuseFacadeProxy)?(100*r.diffuseFacadeProxy).toFixed(1)+'%':'—';
    const cr=r.coresEnabled&&Number.isFinite(r.coreCoverage)?`${r.coreCount??0} (${Math.round(100*r.coreCoverage)}%)`:'—';
    const ln=Number.isFinite(r.longNarrowFraction)?(100*r.longNarrowFraction).toFixed(1)+'%':'—';
    const sl=r.slimVoxels??'—';
    const ed=r.edits??'—';
    const tm=Number.isFinite(r.episodeSeconds)?r.episodeSeconds.toFixed(2)+'s':'—';
    return `<tr class="${cls}"><td>${r.episode}</td><td>${r.training?'Train':'Eval'}</td><td>${Number.isFinite(r.far)?r.far.toFixed(2):'—'}${r.success===false?' ✕':''}</td><td>${sc}</td><td>${ga}</td><td>${su}</td><td>${sk}</td><td>${cr}</td><td>${ln}</td><td>${sl}</td><td>${ed}</td><td>${tm}</td></tr>`;
   }).join('');
  }
 }
}
function renderSavedLayouts() {
 const memory = data.layoutMemory;
 const picker = $('savedLayout');
 const layouts = memory?.layouts ?? [];
 if (savedPreview && (savedPreview.scope !== memory?.scope ||
     !layouts.some(layout => layout.id === savedPreview.geometry.layout.id))) {
  savedPreview = null;
  updateGeometry();
 }
 const key = JSON.stringify([memory?.scope, layouts.map(layout => layout.id)]);
 if (key !== archiveOptionsKey) {
  archiveOptionsKey = key;
  picker.replaceChildren(
   option('', layouts.length ? 'Live layout' : 'Live layout · no saved designs'),
   ...layouts.map((layout, i) => option(layout.id,
    `#${i + 1} · Score ${layout.score.toFixed(3)} · FAR ${layout.far.toFixed(2)}`)),
  );
 }
 picker.disabled = layouts.length === 0;
 picker.title = memory?.warning || 'Five best distinct FAR-valid layouts for this site and setup';
 if (requestedLayout?.scope !== memory?.scope) requestedLayout = null;
 picker.value = requestedLayout?.id || savedPreview?.geometry.layout.id || '';
 const label = $('savedLayoutStatus');
 const layout = savedPreview?.geometry.layout;
 label.hidden = !layout;
 label.textContent = layout ? `Saved layout · FAR ${layout.far.toFixed(2)} · ${Math.round(layout.usableArea)} m² usable · dashboard shows live run` : '';
 $('field').disabled = Boolean(layout);
}

$('savedLayout').onchange = async () => {
 const id = $('savedLayout').value;
 const scope = data?.layoutMemory?.scope;
 const request = id ? {id, scope} : null;
 requestedLayout = request;
 if (!id) {
  savedPreview = null;
  renderSavedLayouts();
  updateGeometry();
  return;
 }
 try {
  const response = await fetch(`/api/layouts/${encodeURIComponent(scope)}/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error('This layout is no longer available for the active site.');
  const geometry = await response.json();
  if (request !== requestedLayout || scope !== data?.layoutMemory?.scope) return;
  requestedLayout = null;
  savedPreview = {scope, geometry};
  renderSavedLayouts();
  updateGeometry();
 } catch (err) {
  if (request !== requestedLayout) return;
  requestedLayout = null;
  renderSavedLayouts();
  error(err.message);
 }
};

function trainToggle() {
 return control(data?.running && data.training ? 'pause' : 'run', {
  training: true,
  acrossSites: $('trainingSiteMode').value === 'rotation',
 });
}

function render(){
 if(!data?.ready)return;
 renderSavedLayouts();
 const rotation = data.siteRotation;
 const trainingMode = $('trainingSiteMode');
 if (!trainingModeLoaded && rotation) {
  trainingMode.value = rotation.enabled ? 'rotation' : 'current';
  trainingModeLoaded = true;
 }
 trainingMode.disabled = Boolean(data.running);
 const rotationSites = rotation?.sites ?? [];
 const rotateOption = trainingMode.querySelector('option[value=rotation]');
 rotateOption.disabled = rotationSites.length < 2;
 const trainingRecords = catalog.filter(site => rotationSites.includes(site.siteId));
 const tierCount = new Set(trainingRecords.map(site => site.tier)).size;
 rotateOption.textContent = rotationSites.length
  ? `Rotate ${rotationSites.length} training sites`
  : 'Rotate training sites';
 trainingMode.title = rotationSites.length
  ? `${rotation.block} episodes per site · ${tierCount} plot tiers`
  : 'No training sites configured';
 if (rotation?.error) error(`Training rotation paused: ${rotation.error}`);
 const m=data.metrics,l=data.learning,c=data.context,siteKey=c?`${c.site_id}:${c.gridAlignment||'world'}:${data.shape[0]}`:`synthetic:${data.seed}:${data.shape[0]}`,changedSite=siteKey!==lastSite;
 $('status').textContent=`${data.running?(data.training?'Training':'Generating'):'Paused'} · ${m.device.toUpperCase()} · ${data.activeSeconds.toFixed(1)} s active`;
 $('sourceLabel').textContent=c?(c.isCustom?'Module Lab custom plot':'Fetched urban context'):'Synthetic method fixture';
 $('train').textContent=data.running&&data.training?'Pause training':'▶ Train';$('train').setAttribute('aria-pressed',String(data.running&&data.training));
 $('infer').textContent=data.running&&!data.training?'Pause generation':'Generate one';
 const settings=data.settings||{targetFar:3,densityPreset:'medium',densityPresets:{low:1.5,medium:3,high:4.5}};
 const settingsKey=`${settings.targetFar}:${settings.densityPreset}:${settings.maskMode}:${settings.deepInterior}:${settings.maximumInteriorVoids}:${settings.maximumEditVoxels}:${settings.editAllocation}:${settings.initialStoreys}:${settings.ceilingStoreys}:${settings.coresEnabled}:${settings.minCoreSpacing}:${settings.maxCoreDist}`;
 if(settingsKey!==lastSettingsKey){lastSettingsKey=settingsKey;$('targetFar').value=settings.targetFar;$('densityPreset').value=settings.densityPreset;$('maskMode').value=settings.maskMode||'independent';$('deepInterior').checked=!!settings.deepInterior;$('maxVoids').value=settings.maximumInteriorVoids??0;$('maxEditVoxels').value=settings.maximumEditVoxels??0;$('editAllocation').value=settings.editAllocation||'probability';if($('initialStoreys')&&settings.initialStoreys!=null)$('initialStoreys').value=settings.initialStoreys;if($('ceilingStoreys')&&settings.ceilingStoreys!=null)$('ceilingStoreys').value=settings.ceilingStoreys;if($('coresEnabled')&&settings.coresEnabled!=null)$('coresEnabled').checked=Boolean(settings.coresEnabled);if($('minCoreSpacing')&&settings.minCoreSpacing!=null)$('minCoreSpacing').value=settings.minCoreSpacing;if($('maxCoreDist')&&settings.maxCoreDist!=null)$('maxCoreDist').value=settings.maxCoreDist;}
 if(Number.isFinite(settings.maximumFar)){$('targetFar').max=settings.maximumFar;for(const o of $('densityPreset').options){o.disabled=o.value!=='custom'&&settings.densityPresets[o.value]>settings.maximumFar;}}
 $('farDefinition').textContent=`Gross occupied floor area / parcel area. Editable study targets. Envelope maximum FAR ${(settings.maximumFar||6).toFixed(3)}.`;
 $('patchRules').textContent=`Deep cuts ramp from 8 to 16 m from the facade. Interior openings must connect to outside air. Ordinary patches retain at least ${settings.minimumPatch??1} cells; the final FAR residual can be smaller. New interior openings require at least ${settings.minimumVoidArea??1} m² per floor. Opening limit 0 means no cap.`;
 if($('coreNote')){
  if(settings.coresEnabled===false){$('coreNote').textContent='Vertical cores disabled (toggle checkbox above to include core shapes, travel limits & Core score).';}
  else{
   const counts=new Map();
   for(const cr of (data.cores||[])){const layout=cr.stairLayout==='scissor'?' scissor':'';const lifts=Number.isInteger(cr.liftCount)?` · ${cr.liftCount} lift${cr.liftCount===1?'':'s'}`:'';const k=`${cr.tier}${layout} ${cr.sizeY}×${cr.sizeX} m · ${cr.floors}F${lifts}`;counts.set(k,(counts.get(k)||0)+1);}
   const coreList=Array.from(counts,([k,n])=>n>1?`${n}× ${k}`:k).join(', ')||'none fit';
   $('coreNote').textContent=`${data.coreLayoutVersion?data.coreLayoutVersion+' · ':''}${m.coreCount??0} core${(m.coreCount??0)===1?'':'s'} (${coreList}) · ${(100*(m.coreRatio||0)).toFixed(1)}% GFA · avg ${(m.coreDistanceMean??0).toFixed(1)} / max ${(m.coreDistanceMax??0).toFixed(1)} m · ${(100*(m.coreUncoveredFraction||0)).toFixed(1)}% > ${settings.maxCoreDist??12} m.`;
  }
 }
 const currentFar=m.far??m.occupiedFloorArea/(c?.siteArea||data.shape[1]*data.shape[2]);
 const auditNote=m.environmentalFieldsCurrent===false?`Last audited at mask ${m.environmentalAuditEdit}`:'Current environmental audit';
 const goodArea=m.usableGoodArea??m.potentialUsableArea??0,goodPct=100*(m.usableGoodFraction??m.potentialUsableFraction??0),scoreVal=m.organicObjective??m.quality??0;
 $('metrics').innerHTML=metric(currentFar.toFixed(3),'FAR',`Target ${settings.targetFar.toFixed(3)}${settings.farTolerance?` ±${settings.farTolerance.toFixed(2)}`:''}${m.targetReached?' · reached':''}`)+metric(Math.round(goodArea).toLocaleString()+' m²','Usable good',`${goodPct.toFixed(1)}% usable · ${(m.coreCount??0)} core${(m.coreCount??0)===1?'':'s'} · Max ${(m.coreDistanceMax??0).toFixed(1)}/${settings.maxCoreDist??12} m`)+metric((100*m.directSunProxy).toFixed(1)+'%','Direct sun proxy',`Diffuse sky ${(100*(m.diffuseFacadeProxy||0)).toFixed(1)}% · ${auditNote}`)+metric(scoreVal.toFixed(3),'Layout score',`${settings.coresEnabled!==false&&Number.isFinite(m.coreScore)?`Core ${(100*m.coreScore).toFixed(0)}% · `:''}Narrow <6m ${(100*(m.longNarrowFraction||0)).toFixed(1)}% · Over ${(100*(m.coreUncoveredFraction||0)).toFixed(1)}%`);
 $('studyTitle').textContent=c?c.city:'Synthetic '+data.plan+' study';
 $('buildingCount').textContent=c?`${c.buildings.length} fetched buildings`:`${data.boxes.length} synthetic boxes`;
 $('contextStatus').textContent=c?`${c.site_id} · ${c.tier} · ${c.buildings.length} neighbours`:`Synthetic plot · seed ${data.seed}`;
 const termination=data.termination==='far_reached'?' · FAR reached':data.termination==='no_admissible_edits'?' · No facade cells':data.termination==='mask_generation_failed'?' · Generated mask rejected':data.termination==='step_budget_exhausted'?' · Step budget exhausted · FAR not reached':'';
 const epStatusText=`${data.episodes} completed episode${data.episodes===1?'':'s'} · ${m.edits} ${m.edits===1?'mask':'masks'}${data.attempts!=null?` · ${data.attempts} attempts`:''}${m.stepBudget!=null?` / ${m.stepBudget} budget`:''} · ${m.voxelsEdited??0} voxels changed · ${l.directMaskUpdates??0} mask-head updates${l.importedUpdates?` · ${l.importedUpdates} imported updates`:""}${termination}${data.lastRejection&&!data.episodeComplete?' · Last draw rejected; try again':''}`;
 const rotationText = rotation?.enabled
  ? ` · Site ${rotationSites.indexOf(data.siteId) + 1}/${rotationSites.length} · ${rotation.completed}/${rotation.block} episodes`
  : '';
 $('episodeStatus').textContent=epStatusText+rotationText;$('episodeStatus').title=epStatusText+rotationText;
 const chartCount = chartRows().length;
 $('progressCount').textContent=`${chartCount} site episode${chartCount===1?'':'s'} charted`;
 $('details').textContent=`Loss ${l.loss.toFixed(4)} · gradient ${l.gradientNorm.toFixed(4)}. Evaluation ${(m.evaluationSeconds*1000).toFixed(1)} ms · action ${(data.actionSeconds*1000).toFixed(1)} ms · learning ${(data.learningSeconds*1000).toFixed(1)} ms. ${m.newReceivers} new / ${m.receivers} exposed receivers. Seed ${data.seed}. Optimizer rollout: ${data.rolloutSteps||16} masks; this does not end the episode. Environmental fields audited every ${data.environmentAuditSteps||5} masks and at completion. Last audit: mask ${m.environmentalAuditEdit??0}. Target tolerance: ±${(m.farTolerance||0).toFixed(6)} FAR. Facade snapshot ${(1000*(m.facadeEvaluationSeconds||0)).toFixed(2)} ms · neural mask generation ${(1000*(data.neuralGenerationSeconds||0)).toFixed(2)} ms · constraint projection ${(1000*(m.maskProjectionSeconds||0)).toFixed(2)} ms. Last mask: ${m.lastChangedVoxels||0} applied / ${m.lastGeneratedVoxels||0} generated voxels. Direct mask-head updates: ${l.directMaskUpdates??0}; imported historical updates: ${l.importedUpdates??0}. Last mask: ${data.maskStructure?.components??'—'} patches; largest patch ${((data.maskStructure?.largestShare??0)*100).toFixed(0)}%. Deep floor area (>8 m): ${Math.round(m.deepFloorArea||0)} m².`;
 $('field').querySelector('option[value=noise]').disabled=!m.noiseAvailable;
 if(!m.noiseAvailable&&view==='noise'){view='solid';$('field').value=view;updateLayerLegend();lastGeomRevision=-1;}
 $('envelopeLabel').textContent=`${settings.initialStoreys??6}F start · ${data.shape[0]}F ceiling · 1 m resolution`;
 if($('envelopeNote')&&settings.contextStoreys){$('envelopeNote').textContent=`Context avg ~${settings.contextStoreys.avg}F · max ~${settings.contextStoreys.max}F (3 m/floor). Rebuilds cache for new ceilings; keeps learned weights.`;}
 if(changedSite){
  lastSite=siteKey;
  $('gridAlignment').replaceChildren(option('auto_fit','Auto'),option('world','World'),option('principal_major','Principal 1'),option('longest_edge','Longest edge'));
  $('gridAlignment').value=c?.gridAlignment||'world';$('applyGrid').disabled=!c;
  $('contextNote').textContent=c?`${c.buildings.length} original building meshes and ${c.roads.length} street meshes from the Module Lab OSM cache. Parcel rasterized at 1 m. Road traffic data is unavailable; noise is unmeasured.`:'Synthetic plot, surrounding boxes and line-source noise fixture. Not a fetched urban site.';
  lastGeomRevision=data.revision;lastRenderedEpisode=data.episodes;lastGeomAt=performance.now();
  updateContext();
 }
 if(lastGeomRevision!==data.revision){
  const fastTrain=Boolean(data.running&&data.training&&$('fastTrainView')?.checked);
  const now=performance.now();
  if(!fastTrain||data.episodes!==lastRenderedEpisode||(now-lastGeomAt>1500)){
   lastGeomRevision=data.revision;lastRenderedEpisode=data.episodes;lastGeomAt=now;
   updateGeometry();
  }
 }
 $('connection').textContent='Optimizer connected';
 if(data.error)error(data.error);
 drawDashboard();
 refreshChartHistory();
}
function updateLayerLegend(){
 const maxD=data?.settings?.maxCoreDist??12;
 const labels={solid:'Occupied volume · crimson cores',usable:'Potential usable area · crimson cores',coreDistance:`Core (red) · ≤${maxD} vox (green→amber) · Over (magenta/purple)`,light:'Low → high diffuse sky access',noise:'Low → high relative noise',depthBadness:'Depth: ≤8 m → ≥16 m'};
 $('layerLegend').className='layer-legend'+(['light','noise','depthBadness','coreDistance'].includes(view)?' heat':'');$('layerLegend').lastElementChild.textContent=labels[view];
 $('layerLegend').firstElementChild.style.background=view==='usable'?'#86b4ad':view==='coreDistance'?'linear-gradient(90deg,#dc2626 0%,#28a869 25%,#d97706 65%,#e11d48 85%,#6b21a8 100%)':'';
}
async function control(command,extra={}){
 if(busy)return false;busy=true;error('');
 document.querySelectorAll('button').forEach(b=>b.disabled=true);
 $('notice').hidden=false;$('notice').textContent=command==='site'?'Preparing parcel, context and visibility caches…':'Applying command…';
 try{
  const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,...extra})});
  if(!r.ok){const text=await r.text();let message=text;try{message=JSON.parse(text).detail||text;}catch{}throw new Error(message);}
  await poll(true);return true;
 }catch(e){error(e.message);return false;}
 finally{busy=false;document.querySelectorAll('button').forEach(b=>b.disabled=false);$('applySite').disabled=!filteredSites().length;$('notice').hidden=true;}
}
function customSource(){return $('source').value==='module_lab_custom';}
function filteredSites(){
 const custom=customSource();
 return catalog.filter(r=>Boolean(r.isCustom)===custom&&(custom||((!$('city').value||r.cityCode===$('city').value)&&(!$('tier').value||r.tier===$('tier').value))));
}
function filterSites(preferred){
 const rows=filteredSites(),current=preferred||$('site').value,custom=customSource();
 $('site').replaceChildren(...(rows.length?rows.map(r=>option(r.siteId,`${custom?r.city:r.siteId}${r.area!=null?` · ${Math.round(r.area).toLocaleString()} m²`:''}`)):[option('',custom?'No saved custom plots':'No matching sites')]));
 if(rows.some(r=>r.siteId===current))$('site').value=current;
 $('catalogCount').textContent=custom
  ?(rows.length?`${rows.length} saved ${rows.length===1?'plot':'plots'} from Module Lab.`:'Save a custom plot in Module Lab, then refresh this list.')
  :`${rows.length} matching cached sites. Original footprints and heights.`;
 $('site').disabled=!rows.length;
 $('applySite').disabled=!rows.length;
}
async function loadSite(id=$('site').value){
 if(!id)return;
 const ok=await control('site',{source:'cached_osm',siteId:id});
 if(ok){$('source').value=catalog.find(r=>r.siteId===id)?.isCustom?'module_lab_custom':'cached_osm';sourceSettings(id);}
}
function sourceSettings(preferred){
 const synthetic=$('source').value==='synthetic',custom=customSource();
 $('urbanSettings').hidden=synthetic;
 $('syntheticSettings').hidden=!synthetic;
 $('catalogFilters').hidden=custom;
 $('refreshSites').hidden=!custom;
 $('siteLabel').textContent=custom?'Saved custom plot':'Cached site';
 if(!synthetic)filterSites(preferred);
}
async function refreshCatalog(preferred){
 const response=await fetch('/api/sites',{cache:'no-store'});
 if(!response.ok)throw new Error('Site catalog unavailable.');
 catalog=await response.json();
 const selectedCity=$('city').value;
 const cities=new Map(catalog.filter(r=>!r.isCustom).map(r=>[r.cityCode,r.city]));
 $('city').replaceChildren(option('','All cities'),...Array.from(cities,([code,name])=>option(code,name)));
 if(cities.has(selectedCity))$('city').value=selectedCity;
 filterSites(preferred);
}
async function navigateSite(direction){
 const rows=filteredSites();if(!rows.length)return;
 const current=rows.findIndex(r=>r.siteId===data?.siteId),next=rows[(current+direction+rows.length)%rows.length];$('site').value=next.siteId;await loadSite(next.siteId);
}
$('train').onclick=trainToggle;$('infer').onclick=()=>control(data?.running&&!data.training?'pause':'run',{training:false});$('step').onclick=()=>control('step',{training:false});$('reset').onclick=()=>control('reset');
$('new').onclick=()=>navigateSite(1);$('previous').onclick=()=>navigateSite(-1);$('applySite').onclick=()=>loadSite();$('applySynthetic').onclick=()=>control('site',{source:'synthetic',plan:$('plan').value});
$('densityPreset').onchange=()=>{const presets=data?.settings?.densityPresets||{low:1.5,medium:3,high:4.5};if($('densityPreset').value!=='custom')$('targetFar').value=presets[$('densityPreset').value];};
$('targetFar').oninput=()=>{$('densityPreset').value='custom';};
$('applyFar').onclick=()=>control('settings',{targetFar:Number($('targetFar').value),densityPreset:$('densityPreset').value});
$('applyMasks').onclick=()=>control('masks',{mode:$('maskMode').value,deepInterior:$('deepInterior').checked,maximumInteriorVoids:Number($('maxVoids').value),maximumEditVoxels:Number($('maxEditVoxels').value),editAllocation:$('editAllocation').value});
$('applyEnvelope')?.addEventListener('click',()=>control('envelope',{initialStoreys:Number($('initialStoreys').value),ceilingStoreys:Number($('ceilingStoreys').value)}));
$('autoEnvelope')?.addEventListener('click',()=>control('envelope',{auto:true}));
$('applyCores')?.addEventListener('click',()=>control('cores',{coresEnabled:$('coresEnabled').checked,minCoreSpacing:Number($('minCoreSpacing').value),maxCoreDist:Number($('maxCoreDist').value)}));
$('coresEnabled')?.addEventListener('change',()=>control('cores',{coresEnabled:$('coresEnabled').checked,minCoreSpacing:Number($('minCoreSpacing').value),maxCoreDist:Number($('maxCoreDist').value)}));
$('resetWeights')?.addEventListener('click',async()=>{const menu=$('resetWeights').closest('details');if(menu)menu.open=false;await control('reset_weights');});
$('source').onchange=()=>sourceSettings();$('city').onchange=()=>filterSites();$('tier').onchange=()=>filterSites();
$('refreshSites').onclick=async()=>{
 $('refreshSites').disabled=true;error('');
 try{await refreshCatalog();}catch(e){error(e.message);}finally{$('refreshSites').disabled=false;}
};
$('field').onchange=async()=>{view=$('field').value;updateLayerLegend();if(!data)return;if(view!=='solid'&&data[view]==null){await poll(true);}updateGeometry();};
$('fastTrainView')?.addEventListener('change',()=>{if(data&&lastGeomRevision!==data.revision&&!$('fastTrainView').checked){lastGeomRevision=data.revision;lastRenderedEpisode=data.episodes;lastGeomAt=performance.now();updateGeometry();}});
$('showContext').onchange=()=>{if(frameReady)viewerMessage({type:'voxel_display',...displayOptions()});};$('fit').onclick=fitViews;
$('transparentContext').onchange=()=>{if(frameReady)viewerMessage({type:'voxel_display',...displayOptions()});};
$('axono').onclick=()=>setProjection('axonometric');$('perspective').onclick=()=>setProjection('perspective');
$('progress').ontoggle=drawDashboard;
$('load').onchange=async()=>{const file=$('load').files[0];if(!file)return;error('');try{const r=await fetch('/api/policy',{method:'POST',body:await file.arrayBuffer()});if(!r.ok)throw new Error(await r.text());await poll(true);}catch(e){error(e.message);}finally{$('load').value='';}};
async function poll(force=false){
 if(polling){if(force)pendingPoll=true;return;}
 polling=true;
 try{const r=await fetch('/api/state?view='+encodeURIComponent(view));if(!r.ok)throw new Error('State unavailable');data=await r.json();if(data.error)error(data.error);if(data.ready)render();else $('status').textContent=data.error?'Context unavailable':'Preparing parcel and visibility caches…';}
 catch(e){$('status').textContent='Connecting…';$('connection').textContent='Optimizer disconnected';}
 finally{polling=false;if(pendingPoll){pendingPoll=false;poll();}}
}
async function init(){
 await poll(true);
 try{await refreshCatalog(data?.siteId||'m_bcn_0093');
 if(catalog.find(r=>r.siteId===data?.siteId)?.isCustom)$('source').value='module_lab_custom';
 sourceSettings(data?.siteId);
 }catch(e){error(e.message);}
}
const ro=new ResizeObserver(drawDashboard);ro.observe($('history'));if($('qualityChart'))ro.observe($('qualityChart'));
setInterval(poll,500);init();
document.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea,button,summary,a')||e.metaKey||e.ctrlKey||e.altKey||e.repeat)return;if(e.code==='Space'){e.preventDefault();data?.running ? control('pause') : trainToggle();}if(e.key==='0')fitViews();});

$('applyGrid').onclick=()=>control('grid',{alignment:$('gridAlignment').value});
