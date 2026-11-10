// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
// Shared helpers, page switching, and the two form pages (1-D column, 2-D section). The fault editor is in faults.js.
"use strict";
const $ = (id) => document.getElementById(id);
const PB = { opts: null, onShow: {} };

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  return r;
}
const apiJson = async (path, body) => (await api(path, body)).json();

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove("show"), 2500);
}
function setStatus(id, msg, err) { const s = $(id); s.textContent = msg; s.classList.toggle("err", !!err); }

function tableHTML(t) {
  const esc = (v) => (v === null ? "" : String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;"));
  return `<div class="tbl"><table><thead><tr>${t.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
    t.data.map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
}
function csvOf(t) {
  const q = (v) => (v === null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return [t.columns.map(q).join(",")].concat(t.data.map((r) => r.map(q).join(","))).join("\n") + "\n";
}
function download(name, text) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = name; a.click(); URL.revokeObjectURL(a.href);
}
function readText(input) {      // file input -> Promise<string|null>
  const f = input.files && input.files[0];
  return f ? f.text() : Promise.resolve(null);
}
async function busy(btn, fn) {  // disable a button while a request runs
  btn.disabled = true; const old = btn.textContent; btn.textContent = "Running...";
  try { await fn(); } finally { btn.disabled = false; btn.textContent = old; }
}

// ---------------------------------------------------------------- focused workspaces
const titles = {c:["Open inputs","Compaction model","Geohistory","Thermal history","Property evolution","Output / export"],
                s:["Open section","Compaction model","Run section","Growth / rates","Thermal history","Property evolution","Output / export"]};
const states = {c:{step:1,version:0,input:null,model:null,run:null,last:null,properties:null,curves:"",figures:{},busy:false,stale:new Set()},
                s:{step:1,version:0,input:null,model:null,run:null,last:null,thermal:null,properties:null,curves:"",figures:{},busy:false,stale:new Set()}};
let activePage="home", growthFrames=[], growthTimer=null;
function resultReady(m,i) {const st=states[m];return !!((m==='s'?[null,st.input,st.model,st.run,st.last,st.thermal,st.properties,st.run]:[null,st.input,st.model,st.run,st.last,st.properties,st.run])[i]);}
function renderNav(m) {
  const st=states[m]; $(m+"_steps").innerHTML=titles[m].map((name,i)=>`<button data-step="${i+1}" class="${st.step===i+1?'current ':''}${resultReady(m,i+1)&&!st.stale.has(i+1)?'done':''} ${st.stale.has(i+1)?'stale':''}" aria-current="${st.step===i+1?'step':'false'}">${i+1} · ${name}</button>`).join("");
  $(m+"_steps").querySelectorAll('button').forEach(b=>b.onclick=()=>goStep(m,+b.dataset.step));
  for(let i=1;i<=titles[m].length;i++) {$(m+"_step"+i).hidden=st.step!==i;$(m+"_result"+i).hidden=!resultReady(m,i);}
  $(m+"_prev").disabled=st.step===1;$(m+"_next").disabled=st.step===titles[m].length;
  $(m+"_model").disabled=!st.input||st.busy; $(m+"_run").disabled=!st.model||st.busy;
  $(m+"_open").disabled=st.busy;$(m+"_export").disabled=!st.run||st.busy;
  $(m==="c"?"c_thermal":"s_growth").disabled=!st.run||st.busy;
  if(m==='c') $("c_prop").disabled=!st.run||st.busy;
  if(m==='s') {$("s_thermal").disabled=!st.last||st.busy;$("s_prop").disabled=!st.last||st.busy;}
}
function goStep(m,i) {
  stopColumnPlayback();stopSectionThermalPlayback();stopSectionPropPlayback();states[m].step=i;renderNav(m);if(i===titles[m].length) renderExport(m);
  if(i===4&&m==='s') showGrowthFrame();
  if(i===5&&m==='s') showSectionThermalFrame();
  if(i===6&&m==='s') showSectionPropFrame();
  window.scrollTo(0,0);
}
function invalidate(m,from) {
  const st=states[m];for(let i=from;i<=titles[m].length;i++)if(resultReady(m,i))st.stale.add(i);st.version++;
  if(from<=1) st.input=null;
  if(from<=2) st.model=null;
  if(from<=3) st.run=null;
  if(from<=4) {st.last=null;if(m==='s'){stopGrowthPlayback();growthFrames=[];$("s_growth_box").hidden=true;}}
  for(let i=from;i<=titles[m].length;i++)setStatus(m+"_status"+i,"Inputs changed — this step's results are not current. Recompute from step "+from+".");
  if(m==='s'&&from<=5){st.thermal=null;sectionThermalFrames=[];stopSectionThermalPlayback();}
  if(from<=(m==='c'?5:6)){st.properties=null;if(m==='s'){sectionPropFrames={};stopSectionPropPlayback();}}
  if(m==='c') {
    stopColumnPlayback();
    if(from<=3) {columnFrames.geo=[];$('c_geo_box').hidden=true;}
    if(from<=4) {columnFrames.thermo=[];$('c_thermo_box').hidden=true;}
    if(from<=5) {columnFrames.prop=[];$('c_prop_box').hidden=true;}
  }
  renderNav(m);if(st.step===titles[m].length)renderExport(m);
}
function showPage(name) {
  if(["column","section"].includes(name)&&["column","section"].includes(activePage)&&name!==activePage) return;
  stopGrowthPlayback(); stopColumnPlayback(); stopSectionThermalPlayback(); stopSectionPropPlayback(); activePage=name;
  for(const p of ["home","column","section","faults"]) $("page-"+p).hidden=p!==name;
  document.querySelectorAll('[data-home-nav]').forEach(b=>b.hidden=name!=="home");
  $("workspace_label").hidden=name==="home";$("workspace_label").textContent=name==="column"?"1-D column · guided analysis":name==="section"?"2-D section · guided analysis":"Fault editor";
  $("editor_return").hidden=name!=="faults";
  if(name==='column')goStep('c',states.c.step);if(name==='section')goStep('s',states.s.step);
  history.replaceState(null,"","#"+name);if(PB.onShow[name])PB.onShow[name]();window.scrollTo(0,0);
}
document.querySelectorAll('[data-page],[data-go]').forEach(b=>b.onclick=()=>showPage(b.dataset.page||b.dataset.go));
$("editor_return").onclick=()=>{showPage('section');goStep('s',1);};
$("s_open_editor").onclick=()=>showPage('faults');
$("f_use_section").onclick=async()=>{try{const st=await apiJson('/api/faults/state');$("s_fl_text").value='fault_id,x,z\n'+st.faults.flatMap(f=>f.nodes.map(p=>[f.id,...p].join(','))).join('\n')+'\n';invalidate('s',1);showPage('section');goStep('s',1);toast('Editor sticks copied to step 1. Open / plot explicitly to apply.');}catch(e){toast(e.message);}};
$("quitBtn").onclick=async()=>{try{await api('/api/quit',{});}catch(e){}document.body.innerHTML='<p>pyBasinSubs stopped. You can close this tab.</p>';};
function image(id,data){$(id).src='data:image/png;base64,'+data;$(id).hidden=false;}
function clearFile(input){input.value='';}
function showJobProgress(label, percent) {
  const box = $("job_progress"), fill = $("job_progress_fill"), text = $("job_progress_label");
  if (!box) return;
  box.hidden = false;
  if (text) text.textContent = label || (`Computing properties… ${Math.round(percent || 0)}%`);
  if (fill) fill.style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
}
function hideJobProgress() {
  const box = $("job_progress"), fill = $("job_progress_fill");
  if (box) box.hidden = true;
  if (fill) fill.style.width = "0%";
}
function watchJobProgress(jobId) {
  const stop = { done: false };
  (async () => {
    while (!stop.done) {
      try {
        const rec = await apiJson(`/api/wizard/progress/${encodeURIComponent(jobId)}`);
        if (stop.done) break;
        if (rec && rec.status && rec.status !== "unknown") {
          const pct = Number(rec.percent) || 0;
          showJobProgress(rec.message || (`Computing properties… ${Math.round(pct)}%`), pct);
          if (rec.status === "done" || rec.status === "error") break;
        }
      } catch (e) { /* keep polling until the POST finishes */ }
      await new Promise((r) => setTimeout(r, 250));
    }
  })();
  return stop;
}
async function action(m,step,path,body,apply) {
 invalidate(m,step);const st=states[m],version=st.version;st.busy=true;renderNav(m);setStatus(m+'_status'+step,'Computing this step from the opened inputs…');
 const wantProgress=/\/properties$/.test(path);
 const payload = body && typeof body === "object" ? Object.assign({}, body) : body;
 let stop = null;
 try {
  if (wantProgress) {
    const jobId = (crypto.randomUUID && crypto.randomUUID()) || (`job-${Date.now()}`);
    payload.job_id = jobId;
    showJobProgress("Computing properties… 0%", 0);
    stop = watchJobProgress(jobId);
  }
  const r=await apiJson(path,payload);if(st.version!==version){setStatus(m+'_status'+step,'Inputs changed during computation; old result discarded. Recompute.');return;}apply(r);st.stale.delete(step);setStatus(m+'_status'+step,r.status||r.note||(r.notes?r.notes.join(' '):'Current result ready.'));
 }
 catch(e){setStatus(m+'_status'+step,'Error: '+e.message,true);}
 finally{if(stop) stop.done=true;hideJobProgress();st.busy=false;renderNav(m);}
}
function selectedDataset(){return PB.opts.datasets[+$("c_dataset").value];}
$("c_dataset").onchange=()=>{$("c_input_text").value='';clearFile($("c_csv"));invalidate('c',1);};
function renderCurvePicker(m,options,selected){
 const picked=new Set(selected===undefined?options.filter(o=>o.used).map(o=>o.name):selected);
 const box=$(m+'_curve_options');box.replaceChildren();
 for(const option of options){
  const label=document.createElement('label'),check=document.createElement('input'),text=document.createElement('span');
  check.type='checkbox';check.value=option.name;check.checked=picked.has(option.name);text.textContent=option.label;
  label.append(check,text);if(option.used){const tag=document.createElement('span');tag.className='curve-used';tag.textContent='used';label.append(tag);}
  check.onchange=()=>{updateCurveCount(m);invalidate(m,2);};box.append(label);
 }
 updateCurveCount(m);
}
function plottedCurves(m){return [...$(m+'_curve_options').querySelectorAll('input:checked')].map(e=>e.value);}
function updateCurveCount(m){$(m+'_curve_count').textContent='('+plottedCurves(m).length+' selected)';}
async function refreshCurveOptions(m){
 const st=states[m];if(!st.input)return;const version=st.version;
 try{const r=await apiJson('/api/wizard/'+(m==='c'?'column':'section')+'/curve-options',{input_id:st.input,curves:$(m+'_curves').value});if(st.version===version)renderCurvePicker(m,r.curve_options,r.selected);}
 catch(e){if(st.version===version)setStatus(m+'_status2','Curve import: '+e.message,true);}
}
$("c_clear_edits").onclick=()=>{$("c_input_text").value='';clearFile($("c_csv"));invalidate('c',1);};
$("c_csv").onchange=async()=>{const text=await readText($("c_csv"));if(text!==null){$("c_input_text").value=text;invalidate('c',1);}};
$("c_open").onclick=()=>{const d=selectedDataset();return action('c',1,'/api/wizard/column/input',{source:d.source,well:d.well,csv:$("c_input_text").value||null,name:d.well||d.source,wscale:+$("c_ws").value,eroded:$("c_ero").value===''?null:+$("c_ero").value},r=>{
 const st=states.c;st.input=r.input_id;st.curves=r.curves;st.openedCurveOptions=r.curve_options;renderCurvePicker('c',r.curve_options);$("c_curves").value=r.curves;$("c_input_text").value=r.files['column.csv'];$("c_ws").value=1;$("c_ero").value='';st.figures={input:r.figure};image('c_strat',r.figure);
});};
for(const [file,text] of [['s_hz','s_hz_text'],['s_un','s_un_text'],['s_fl','s_fl_text']]) $(file).onchange=async()=>{const t=await readText($(file));if(t!==null){$(text).value=t;invalidate('s',1);}};
$("s_reset").onclick=()=>{for(const k of ['s_hz_text','s_un_text','s_fl_text'])$(k).value='';for(const k of ['s_hz','s_un','s_fl'])clearFile($(k));$("s_cut").checked=false;invalidate('s',1);};
$("s_dataset").onchange=()=>$("s_reset").click();
$("s_open").onclick=()=>action('s',1,'/api/wizard/section/input',{horizons:$("s_hz_text").value||null,units:$("s_un_text").value||null,faults:$("s_fl_text").value||null,cutoff_planes:$("s_cut").checked},r=>{
 const st=states.s;st.input=r.input_id;st.curves=r.curves;st.openedCurveOptions=r.curve_options;renderCurvePicker('s',r.curve_options);$("s_curves").value=r.curves;st.figures={input:r.figure};
 for(const [key,file] of [['s_hz_text','horizons.csv'],['s_un_text','units.csv'],['s_fl_text','faults.csv']])$(key).value=r.files[file];image('s_input_fig',r.figure);
});
for(const m of ['c','s']) {
 $(m+'_prev').onclick=()=>goStep(m,Math.max(1,states[m].step-1));$(m+'_next').onclick=()=>goStep(m,Math.min(titles[m].length,states[m].step+1));
 for(const [f,t] of [['curve_file','curves'],['point_file','points']])$(m+'_'+f).onchange=async()=>{const text=await readText($(m+'_'+f));if(text!==null){$(m+'_'+t).value=text;invalidate(m,2);if(f==='curve_file')await refreshCurveOptions(m);}};
 $(m+'_defaults').onclick=()=>{$(m+'_curves').value=states[m].curves;invalidate(m,2);renderCurvePicker(m,states[m].openedCurveOptions||PB.opts.porosity_curves);};
 $(m+'_curves').addEventListener('change',()=>refreshCurveOptions(m));
 $(m+'_model').onclick=()=>action(m,2,'/api/wizard/'+(m==='c'?'column':'section')+'/compaction',{input_id:states[m].input,curves:$(m+'_curves').value,points:$(m+'_points').value,fit_mode:$(m+'_fit_mode').value,plot_curves:plottedCurves(m)},r=>{
  states[m].model=r.model_id;renderCurvePicker(m,r.curve_options,r.plot_curves);states[m].figures.compaction=r.figure;image(m+'_compaction',r.figure);$(m+'_equations').innerHTML=tableHTML(r.table);
 });
}
$("c_run").onclick=()=>action('c',3,'/api/wizard/column/run',{model_id:states.c.model,n:+$("c_n").value,mixing:$("c_mix").value},r=>{states.c.run=r.result_id;states.c.figures.result=r.figure;image('c_geo',r.figure);installColumnFrames('geo',r.frames);$("c_tab").innerHTML=tableHTML(r.table);});
$("s_run").onclick=()=>action('s',3,'/api/wizard/section/run',{model_id:states.s.model,method:$("s_method").value,Te_km:+$("s_te").value,Hd_km:+$("s_hd").value,n_mc:+$("s_mc").value},r=>{states.s.run=r.result_id;states.s.figures.result=r.figure;image('s_fig',r.figure);$("s_tab").innerHTML=tableHTML(r.table);});
$("c_geothermal_file").onchange=async()=>{const text=await readText($("c_geothermal_file"));if(text!==null){$("c_geothermal").value=text;invalidate('c',4);}};
$("c_thermal").onclick=()=>action('c',4,'/api/wizard/column/thermal',{result_id:states.c.run,heat_flow_mW_m2:+$("c_hf").value,surface_temperature_C:+$("c_ts").value,geothermal_csv:$("c_geothermal").value},r=>{states.c.last=r.thermal_id;states.c.figures.thermal=r.figure;image('c_mat',r.figure);installColumnFrames('thermo',r.frames);$("c_mat_tab").innerHTML=tableHTML(r.table);});
// 1-D sliders use negative AGE as their value, not equally spaced frame indices.
const columnFrames={geo:[],thermo:[],prop:[]}, columnIndices={geo:0,thermo:0,prop:0};
let columnTimer=null;
function stopColumnPlayback(){clearInterval(columnTimer);columnTimer=null;for(const k of ['geo','thermo','prop'])$('c_'+k+'_play').textContent='Play timesteps';}
function showColumnFrame(k,index){
 const frames=columnFrames[k];if(!frames.length)return;
 index=Math.max(0,Math.min(frames.length-1,index));columnIndices[k]=index;const f=frames[index];
 image('c_'+k+'_frame',f.figure);
 if(k==='thermo'&&f.maturity_figure)image('c_thermo_maturity',f.maturity_figure);
 $('c_'+k+'_time').value=-f.age_Ma;
 $('c_'+k+'_age').textContent=f.label;$('c_'+k+'_time').setAttribute('aria-valuetext',f.label);
 $('c_'+k+'_prev').disabled=index===0;$('c_'+k+'_next').disabled=index===frames.length-1;
 if(k==='prop'){
  const layers=f.layers||[];
  const num=(v,n)=>Number.isFinite(+v)?(+v).toFixed(n):'—';
  const kmd=v=>{v=+v;if(!Number.isFinite(v))return '—';if(Math.abs(v)>=10)return v.toFixed(0);if(Math.abs(v)>=0.01)return v.toFixed(3);return v.toExponential(2);};
  if(!layers.length){$('c_prop_tab').innerHTML='<p class="help">No units are present at this time step.</p>';}
  else{$('c_prop_tab').innerHTML=tableHTML({columns:['Unit','Top (m)','Base (m)','T (°C)','Porosity','Mechanical porosity','Quartz cement','k (mD)','Effective stress (MPa)','Pore pressure (MPa)'],data:layers.map(l=>[l.unit,Math.round(l.top_m),Math.round(l.base_m),num(l.temperature_C,0),num(l.porosity,3),num(l.phi_mechanical,3),num(l.quartz_cement,3),kmd(l.permeability_mD),num(l.effective_stress_mpa,1),num(l.pore_pressure_mpa,1)])});}
 }else{
  $('c_'+k+'_layers').innerHTML=tableHTML({columns:['#','Unit','Top (m)','Base (m)','Decompacted thickness (m)'],data:f.layers.map(l=>[l.unit_index+1,l.unit,...[l.top_m,l.base_m,l.thickness_m].map(v=>Math.round(v))])});
 }
}
function installColumnFrames(k,frames){
 columnFrames[k]=frames||[];if(!columnFrames[k].length)return;
 const range=$('c_'+k+'_time');range.min=-frames[0].age_Ma;range.max=-frames.at(-1).age_Ma;
 $('c_'+k+'_nodes').replaceChildren(...frames.map(f=>new Option(f.age_Ma+' Ma',String(-f.age_Ma))));
 $('c_'+k+'_box').hidden=false;showColumnFrame(k,frames.length-1);
}
for(const k of ['geo','thermo','prop']){
 $('c_'+k+'_time').oninput=()=>{stopColumnPlayback();const frames=columnFrames[k],v=+$('c_'+k+'_time').value;if(!frames.length)return;let i=0;for(let j=1;j<frames.length;j++)if(Math.abs(-frames[j].age_Ma-v)<=Math.abs(-frames[i].age_Ma-v))i=j;showColumnFrame(k,i);};
 $('c_'+k+'_time').onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();stopColumnPlayback();showColumnFrame(k,e.key==='Home'?0:e.key==='End'?columnFrames[k].length-1:columnIndices[k]+(e.key==='ArrowRight'?1:-1));}};
 $('c_'+k+'_prev').onclick=()=>{stopColumnPlayback();showColumnFrame(k,columnIndices[k]-1);};
 $('c_'+k+'_next').onclick=()=>{stopColumnPlayback();showColumnFrame(k,columnIndices[k]+1);};
 $('c_'+k+'_play').onclick=()=>{if(columnTimer){stopColumnPlayback();return;}if(!columnFrames[k].length)return;showColumnFrame(k,0);$('c_'+k+'_play').textContent='Pause timesteps';columnTimer=setInterval(()=>{if(columnIndices[k]>=columnFrames[k].length-1){stopColumnPlayback();return;}showColumnFrame(k,columnIndices[k]+1);},1000);};
}
function stopGrowthPlayback(){clearInterval(growthTimer);growthTimer=null;$("s_growth_play").textContent='Play timesteps';}
function showGrowthFrame(){const f=growthFrames[+$("s_growth_time").value];if(!f)return;image('s_growth_fig',$("s_fault_numbers").checked?f.figure:f.figure_without_numbers);$("s_growth_age").textContent=f.label;$("s_growth_time").setAttribute('aria-valuetext',f.label);}
$("s_growth_time").oninput=()=>{stopGrowthPlayback();showGrowthFrame();};$("s_fault_numbers").onchange=showGrowthFrame;
$("s_growth_play").onclick=()=>{if(growthTimer){stopGrowthPlayback();return;}if(!growthFrames.length)return;$("s_growth_play").textContent='Pause timesteps';growthTimer=setInterval(()=>{$("s_growth_time").value=(+$("s_growth_time").value+1)%growthFrames.length;showGrowthFrame();},1000);};
$("s_growth").onclick=()=>action('s',4,'/api/wizard/section/growth',{result_id:states.s.run,assume_zero:$("s_zero").checked,contact_placement:$("s_contact").value},r=>{states.s.last=r.growth_id;states.s.thermal=null;sectionThermalFrames=[];growthFrames=r.frames;$("s_growth_time").max=growthFrames.length-1;$("s_growth_time").value=1;$("s_growth_oldest").textContent=growthFrames.at(-1).age_Ma+' Ma';$("s_growth_status").textContent=r.status;$("s_growth_box").hidden=false;showGrowthFrame();states.s.figures.growth=growthFrames[7].figure;});
// Editing any earlier step invalidates its dependent snapshots. No hidden recompute.
for(const [m,level,ids] of [['c',1,['c_input_text','c_ws','c_ero']],['s',1,['s_hz_text','s_un_text','s_fl_text','s_cut']],['c',2,['c_curves','c_points','c_fit_mode']],['s',2,['s_curves','s_points','s_fit_mode']],['c',3,['c_n','c_mix']],['s',3,['s_method','s_te','s_hd','s_mc']],['c',4,['c_hf','c_ts','c_geothermal']],['s',4,['s_zero','s_contact']]])for(const id of ids)$(id).addEventListener('input',()=>invalidate(m,level));
let sectionThermalFrames=[],sectionThermalTimer=null;
function stopSectionThermalPlayback(){clearInterval(sectionThermalTimer);sectionThermalTimer=null;$("s_thermal_play").textContent='Play timesteps';}
function showSectionThermalFrame(){const f=sectionThermalFrames[+$("s_thermal_time").value];if(!f)return;image('s_thermal_fig',f.figure);$("s_thermal_age").textContent=f.label;$("s_thermal_time").setAttribute('aria-valuetext',f.label);}
$("s_thermal_time").oninput=()=>{stopSectionThermalPlayback();showSectionThermalFrame();};
$("s_thermal_play").onclick=()=>{if(sectionThermalTimer){stopSectionThermalPlayback();return;}if(!sectionThermalFrames.length)return;$("s_thermal_play").textContent='Pause timesteps';sectionThermalTimer=setInterval(()=>{$("s_thermal_time").value=(+$("s_thermal_time").value+1)%sectionThermalFrames.length;showSectionThermalFrame();},1000);};
$("s_thermal").onclick=()=>action('s',5,'/api/wizard/section/thermal',{growth_id:states.s.last,heat_flow_mW:+$("s_heat").value,T_s:+$("s_ts").value},r=>{states.s.thermal=r.thermal_id;sectionThermalFrames=r.frames;$("s_thermal_time").max=r.frames.length-1;$("s_thermal_time").value=0;$("s_thermal_oldest").textContent=r.frames.at(-1).age_Ma+' Ma';$("s_thermal_note").textContent=r.note;showSectionThermalFrame();});
for(const id of ['s_heat','s_ts'])$(id).oninput=()=>invalidate('s',5);
$("c_prop").onclick=()=>action('c',5,'/api/wizard/column/properties',{result_id:states.c.run,thermal_id:states.c.last,heat_flow_mW_m2:+$("c_hf").value,surface_temperature_C:+$("c_ts").value},r=>{states.c.properties=r.properties_id;installColumnFrames('prop',r.frames);if(r.frames&&r.frames.length)states.c.figures.properties=r.frames.at(-1).figure;});
let sectionPropFrames={}, sectionPropTimer=null;
function stopSectionPropPlayback(){clearInterval(sectionPropTimer);sectionPropTimer=null;if($("s_prop_play"))$("s_prop_play").textContent='Play timesteps';}
function showSectionPropFrame(){const frames=sectionPropFrames[$("s_prop_var").value]||[];const f=frames[+$("s_prop_time").value];if(!f)return;image('s_prop_fig',f.figure);$("s_prop_age").textContent=f.label;$("s_prop_time").setAttribute('aria-valuetext',f.label);}
$("s_prop_time").oninput=()=>{stopSectionPropPlayback();showSectionPropFrame();};
$("s_prop_var").onchange=()=>{stopSectionPropPlayback();const frames=sectionPropFrames[$("s_prop_var").value]||[];if(frames.length){$("s_prop_time").max=frames.length-1;if(+$("s_prop_time").value>frames.length-1)$("s_prop_time").value=frames.length-1;}showSectionPropFrame();};
$("s_prop_play").onclick=()=>{if(sectionPropTimer){stopSectionPropPlayback();return;}const frames=sectionPropFrames[$("s_prop_var").value]||[];if(!frames.length)return;$("s_prop_play").textContent='Pause timesteps';sectionPropTimer=setInterval(()=>{$("s_prop_time").value=(+$("s_prop_time").value+1)%frames.length;showSectionPropFrame();},1000);};
$("s_prop").onclick=()=>action('s',6,'/api/wizard/section/properties',{growth_id:states.s.last,thermal_id:states.s.thermal,heat_flow_mW:+$("s_heat").value,T_s:+$("s_ts").value},r=>{states.s.properties=r.properties_id;sectionPropFrames=r.frames_by_variable||{};const frames=sectionPropFrames[$("s_prop_var").value]||sectionPropFrames.porosity||[];$("s_prop_time").max=Math.max(0,frames.length-1);$("s_prop_time").value=0;if(frames.length)$("s_prop_oldest").textContent=frames.at(-1).age_Ma+' Ma';if(frames.length)states.s.figures.properties=frames[0].figure;showSectionPropFrame();});

function renderExport(m){const st=states[m],ready=!!st.run;$(m+'_result'+titles[m].length).hidden=!ready;if(!ready)return;
 const names=m==='c'?['Opened column CSV','Applied compaction parameters / real points if supplied','Input and compaction figures','Original geohistory PNG + data CSV', 'Node timestep PNGs + geometry JSON; optional geohistory GIF',...(st.last?['Original thermal PNG + temperature / maturity histories + conditions', 'Thermal + two-panel maturity node PNGs; optional thermal-window and maturity GIFs']:['Thermal not current — excluded']),...(st.properties?['Fixed-layout property-log frames + selected-age layer table; optional property-log GIF']:['Property evolution not current — excluded'])]:['Opened horizons / units / sticks CSV','Applied compaction parameters / real points if supplied','Input / model / computed-section figures','Restoration diagnostics CSV',...(st.last?['Eleven fixed-shape timestep PNGs','Growth / spatial rate tables + prototype metadata','Optional GIF timelapse — rendered from this cached result',...(st.thermal?['Optional cached thermal PNGs + GIF; Easy%Ro fraction CSV']:['Thermal not current — excluded']),...(st.properties?['Section property frames (porosity, permeability, effective stress) + optional porosity GIF']:['Property evolution not current — excluded'])]:['Growth not current — excluded'])];
 $(m+'_manifest').innerHTML='<h3>Current computed output</h3><ul class="export-list">'+names.map(n=>'<li>'+n+'</li>').join('')+'</ul>';
 const keys=['input','compaction','result',...(st.last?[m==='c'?'thermal':'growth']:[]),...(st.properties?['properties']:[])];$(m+'_previews').innerHTML='';for(const k of keys){if(st.figures[k]){const im=document.createElement('img');im.alt=k+' export preview';im.src='data:image/png;base64,'+(m==='s'&&k==='growth'&&growthFrames[7]?($("s_fault_numbers").checked?growthFrames[7].figure:growthFrames[7].figure_without_numbers):st.figures[k]);$(m+'_previews').append(im);}}
 if(m==='s'){$("s_export_animation").disabled=!st.last;$("s_export_thermal").disabled=!st.thermal;$("s_export_properties").disabled=!st.properties;}
}
for(const m of ['c','s'])$(m+'_export').onclick=async()=>{const st=states[m],id=st.properties||st.thermal||st.last||st.run;if(!id)return;const version=st.version;setStatus(m+'_status'+titles[m].length,'Preparing current snapshot ZIP…');try{const query=m==='c'?'?animation='+$("c_export_animation").checked:(st.last?'?animation='+$("s_export_animation").checked+'&fault_numbers='+$("s_fault_numbers").checked+'&section_thermal='+$("s_export_thermal").checked+'&section_properties='+$("s_export_properties").checked:'');const response=await api('/api/wizard/export/'+id+query);const blob=await response.blob();if(st.version!==version){setStatus(m+'_status'+titles[m].length,'Inputs changed; obsolete export discarded.');return;}const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='pybasinsubs_'+(m==='c'?'column':'section')+'_results.zip';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),3000);st.stale.delete(titles[m].length);renderNav(m);setStatus(m+'_status'+titles[m].length,'Export ready: figures, inputs and computed data from this snapshot.');}catch(e){setStatus(m+'_status'+titles[m].length,'Error: '+e.message,true);}};
(async function init(){PB.opts=await apiJson('/api/options');$("c_dataset").replaceChildren(...PB.opts.datasets.map((d,i)=>new Option(d.label,String(i))));$("s_dataset").replaceChildren(...PB.opts.section_datasets.map(d=>new Option(d.label,d.id)));for(const m of ['c','s'])renderCurvePicker(m,PB.opts.porosity_curves,[]);$("s_method").innerHTML=PB.opts.methods.map((m,i)=>`<option${i===1?' selected':''}>${m}</option>`).join('');for(const m of ['c','s']){for(let i=1;i<=titles[m].length;i++)setStatus(m+'_status'+i,'Open/apply the earlier inputs, then compute this step explicitly.');renderNav(m);}const h=location.hash.slice(1);showPage(['column','section','faults'].includes(h)?h:'home');if(PB.initFaults)PB.initFaults();})();
window.PB=PB;window.$=$;window.api=api;window.apiJson=apiJson;window.toast=toast;window.setStatus=setStatus;window.busy=busy;
