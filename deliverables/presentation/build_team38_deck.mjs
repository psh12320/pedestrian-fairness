import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
const { Presentation, PresentationFile } = await import(pathToFileURL("C:/Users/Shricharan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs").href);

const workspaceDir = process.cwd();
const SKILL_DIR = "C:/Users/Shricharan/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const TMP_DIR = path.join(workspaceDir, "deliverables", "presentation", ".build");
const RUNTIME_PYTHON = "C:/Users/Shricharan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

await fs.mkdir(TMP_DIR, { recursive: true });

const P = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  ink: "#102C3A", paper: "#F8FAF8", sea: "#057D72", soft: "#426273",
  pale: "#D5E9E6", white: "#FFFFFF", amber: "#AD5A24", grey: "#65757B",
};
const font = "Arial";

function text(slide, value, x, y, w, h, size=28, color=C.ink, bold=false, align="left") {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: {left:x, top:y, width:w, height:h},
    fill: "none", line: {fill:"none", width:0},
  });
  box.text = value;
  box.text.style = {typeface:font, fontSize:size, color, bold, alignment:align, autoFit:"none"};
  return box;
}
function base(title, subtitle="") {
  const s=P.slides.add();
  s.background.fill=C.paper;
  text(s,title,68,42,1145,82,44,C.ink,true);
  if(subtitle) text(s,subtitle,70,128,1130,50,22,C.soft);
  return s;
}
function notes(slide,value) { slide.speakerNotes.textFrame.setText(value); }
function row(slide,label,value,y) {
  text(slide,label,72,y,275,42,22,C.soft,true);
  text(slide,value,350,y,850,110,27,C.ink);
}

function parseCsv(source) {
  const records=[];
  let record=[], field="", quoted=false;
  for(let i=0;i<source.length;i++) {
    const char=source[i];
    if(char==='"') {
      if(quoted && source[i+1]==='"') { field+='"'; i++; }
      else quoted=!quoted;
    } else if(char===',' && !quoted) {
      record.push(field); field="";
    } else if((char==='\n' || char==='\r') && !quoted) {
      if(char==='\r' && source[i+1]==='\n') i++;
      record.push(field); field="";
      if(record.some(value=>value!=="")) records.push(record);
      record=[];
    } else field+=char;
  }
  if(field!=="" || record.length) { record.push(field); records.push(record); }
  const [header,...body]=records;
  if(!header) return [];
  return body.map(values=>Object.fromEntries(header.map((key,index)=>[key,values[index]??""])));
}

const finalStudyDir=process.env.TEAM38_PRESENTATION_DATA_DIR ||
  path.join(workspaceDir,"results","final-study");
async function loadFinalStudy() {
  try {
    const result=JSON.parse(await fs.readFile(path.join(finalStudyDir,"summary.json"),"utf8"));
    const crossings=parseCsv(await fs.readFile(path.join(finalStudyDir,"crossing_comparison.csv"),"utf8"));
    const decision=JSON.parse(await fs.readFile(path.join(finalStudyDir,"extension_decision.json"),"utf8"));
    const settings=JSON.parse(await fs.readFile(path.join(workspaceDir,"config","evaluation.json"),"utf8"));
    const expectedSeeds=settings.seeds;
    const expectedScenarios=[
      ...Object.keys(settings.vehicle_levels).flatMap(vehicle=>
        Object.keys(settings.pedestrian_levels).map(pedestrian=>`v${vehicle}-p${pedestrian}`)),
      "vmedium-pmedium-slow",...Object.keys(settings.extra_scenarios),
    ];
    const expectedControllers=["actuated","fixed","pressure",
      ...["ippo","mappo"].flatMap(method=>[0,3,10].map(weight=>`${method}-w${weight}`))];
    const manifest=result.training_manifest;
    const modelKeys=(manifest?.models??[]).map(model=>
      `${model.method}-w${model.reward_weight}-r${model.training_run}`);
    const expectedModels=["ippo","mappo"].flatMap(method=>
      [0,3,10].flatMap(weight=>Array.from({length:5},(_,run)=>`${method}-w${weight}-r${run}`)));
    const sameSet=(actual,expected)=>actual.length===expected.length &&
      new Set(actual).size===expected.length &&
      expected.every(item=>actual.includes(item));
    const selectedBudget=decision?.selected_training_episodes;
    const budgetMatches=(typeof decision?.extend_all_to_300==="boolean" &&
      selectedBudget===(decision.extend_all_to_300?300:150) &&
      manifest?.episodes_requested===selectedBudget &&
      manifest?.models?.every(model=>model.episodes_run===selectedBudget));
    const complete=manifest?.complete_protocol===true &&
      budgetMatches &&
      manifest?.settings?.training_run_seeds?.length===5 &&
      manifest?.models?.every(model=>model.episodes_run>=150) &&
      sameSet(modelKeys,expectedModels) &&
      JSON.stringify(result.evaluation_seeds)===JSON.stringify(expectedSeeds) &&
      JSON.stringify(result.settings?.seeds)===JSON.stringify(expectedSeeds) &&
      sameSet(result.scenarios??[],expectedScenarios) &&
      sameSet(result.controllers??[],expectedControllers) &&
      result.settings?.primary_scenario==="vmedium-pmedium" &&
      result.settings?.expected_primary_crossings===12 &&
      expectedScenarios.every(scenario=>expectedControllers.every(controller=>
        result.summary?.[scenario]?.[controller]?.vehicle_time_loss?.n===10)) &&
      expectedControllers.filter(controller=>controller!=="actuated").every(controller=>
        crossings.filter(item=>item.scenario==="vmedium-pmedium" &&
          item.controller===controller).length===12);
    if(!complete) return {ready:false,reason:"The saved study does not pass the 30-replica, 10-seed, 13-scenario completeness gate."};
    const label="ippo-w3";
    const primary="vmedium-pmedium";
    const selected=crossings.filter(item=>item.scenario===primary && item.controller===label)
      .sort((a,b)=>a.crossing.localeCompare(b.crossing,undefined,{numeric:true}));
    if(new Set(selected.map(item=>item.crossing)).size!==12)
      return {ready:false,reason:"The selected controller lacks 12 distinct crossing estimates."};
    return {ready:true,result,selected,label,primary,decision};
  } catch(error) {
    if(error?.code==="ENOENT") return {ready:false,reason:"Final evaluation outputs are not available yet."};
    return {ready:false,reason:`Final evaluation outputs could not be read: ${error.message}`};
  }
}
const finalStudy=await loadFinalStudy();
let transferStudy=null;
try {
  const transfer=JSON.parse(await fs.readFile(
    path.join(workspaceDir,"results","singapore-transfer","summary.json"),"utf8"));
  if(finalStudy.ready && transfer.smoke===false && transfer.run_count===330 &&
    transfer.replicas_per_learned_configuration===5 && transfer.seeds?.length===10 &&
    transfer.selected_training_episodes===finalStudy.decision.selected_training_episodes)
    transferStudy=transfer;
} catch(error) { if(error?.code!=="ENOENT") throw error; }
const FINAL_PPTX=process.env.TEAM38_PRESENTATION_OUTPUT ||
  path.join(workspaceDir,"deliverables","presentation","output",
    finalStudy.ready?"team38-research-talk-v2f-final.pptx":
      "team38-research-talk-v2f-preview.pptx");
await fs.mkdir(path.dirname(FINAL_PPTX),{recursive:true});

function signed(value,digits=1) {
  if(value===null || value===undefined || value==="") return "NA";
  const number=Number(value);
  return Number.isFinite(number)?`${number>0?"+":""}${number.toFixed(digits)}`:"NA";
}
function interval(item,digits=1) {
  if(!item || item.mean===null || item.mean===undefined) return "NA";
  const low=item.ci95_low, high=item.ci95_high;
  return `${signed(item.mean,digits)} [${signed(low,digits)}, ${signed(high,digits)}]`;
}
function plainNumber(value,digits=1) {
  if(value===null || value===undefined || value==="") return "NA";
  const number=Number(value);
  return Number.isFinite(number)?number.toFixed(digits):"NA";
}
function rule(slide,left,top,width,height,color=C.grey) {
  slide.shapes.add({geometry:"rect",position:{left,top,width,height},fill:color,
    line:{fill:"none",width:0}});
}

// 1. Cover
{
  const s=P.slides.add(); s.background.fill=C.ink;
  text(s,"Pedestrian-aware traffic signal control",76,132,1100,158,58,C.white,true);
  text(s,"When can multi-agent PPO save vehicle time without delaying people at crossings?",78,344,1040,100,30,C.pale);
  text(s,"CS4246 Team 38  ·  15-minute presentation",79,573,1020,40,22,C.white);
  notes(s,"The project studies simulation outcomes only. Team 38 built the synthetic SUMO environments and policies from scratch. Do not describe results as deployed-signal improvements.");
}
// 2. Research question and novelty
{
  const s=base("Research question");
  text(s,"Can vehicle delay fall while every crossing keeps its long waits under control?",76,205,1100,170,37,C.sea,true);
  text(s,"The comparison is against a pedestrian-responsive controller on the same demand seeds.",77,418,1095,110,29,C.ink);
  text(s,"Contribution: crossing-by-crossing tail waits, service counts, and careful failure analysis.",77,575,1090,74,23,C.soft);
  notes(s,"Pedestrian-aware traffic RL already exists, so this is a careful trade-off study. The proposal's strict crossing condition motivates the analysis. Related work and citations are in docs/literature.md. For centralized-critic MAPPO see Yu et al., https://arxiv.org/abs/2103.01955.");
}
// 3. Original matched-demand SUMO GUI stills
{
  const s=base("Three-junction SUMO corridor","Original synthetic roads, sidewalks, crossings, and seeded trips");
  const assetDir=path.join(workspaceDir,"deliverables","presentation","assets","final-protocol");
  const fixedPath=path.join(assetDir,"sumo-fixed-seed-6001-t147.png");
  const ippoPath=path.join(assetDir,"sumo-ippo-pedestrian-seed-6001-t147.png");
  text(s,"Fixed time",76,181,490,42,24,C.sea,true);
  text(s,"IPPO weight 3, replica 1 of 5",679,181,500,42,24,C.sea,true);
  s.images.add({blob:new Uint8Array(await fs.readFile(fixedPath)),contentType:"image/png",alt:"Original SUMO GUI view of the fixed-time signal at junction C1",fit:"contain",position:{left:68,top:225,width:540,height:360}});
  s.images.add({blob:new Uint8Array(await fs.readFile(ippoPath)),contentType:"image/png",alt:"Original SUMO GUI view of the IPPO pedestrian weight 3 policy at junction C1",fit:"contain",position:{left:673,top:225,width:540,height:360}});
  text(s,"Same seed 6001 and scheduled trips at 147 s  ·  C1 of 3 signals  ·  12 protected crossings total",76,610,1120,67,23,C.soft);
  notes(s,"Original SUMO GUI screenshots use recorded phase replays, exploratory seed 6001 and the same scheduled demand at t=147 s. Fixed time is left. IPPO weight 3, training replica zero, is right, selected as the illustration before inspecting final results. Both use the final shared safety mechanism. The matched moving comparison is deliverables/presentation/assets/final-protocol/sumo-fixed-vs-ippo-pedestrian-seed-6001.mp4. These single-replica visualizations accompany the five-replica measurements on slide 10. Source stills are in deliverables/presentation/assets/final-protocol/. This is the synthetic corridor.");
}
// 4. Controllers
{
  const s=base("Controller set","Three conventional rules and two PPO methods across three reward weights");
  row(s,"Conventional","Fixed time, pedestrian actuated, pressure inspired",204);
  row(s,"IPPO","A local actor and a local value estimator at each signal",322);
  row(s,"MAPPO","Local actors; per-junction value estimators see the network during training",442);
  text(s,"PPO pedestrian penalty weights: 0, 3, 10",74,596,1100,55,28,C.sea,true);
  notes(s,"The pressure baseline is inspired by queue pressure; it is not an exact reproduction of published max-pressure control. There are 2 PPO methods x 3 reward weights x 5 training seeds = 30 learned replicas in the final protocol. Local actors have the same observation and per-junction reward scope; only MAPPO value estimators see all junction observations during training. Source: README.md, config/training.json, src/traffic_control/ppo.py.");
}
// 5. Safety
{
  const s=base("One safety mechanism for every policy");
  text(s,"Legal phase groups",74,170,450,53,27,C.sea,true);
  text(s,"East–west vehicles\nNorth–south vehicles\nExclusive pedestrian crossing",75,235,520,180,27,C.ink);
  text(s,"Required timing",675,170,450,53,27,C.sea,true);
  text(s,"At least 12 s vehicle green\n3 s yellow, then 2 s all-red\n12 s pedestrian green",676,235,525,180,27,C.ink);
  text(s,"Pedestrian clearance uses the longest crossing at 0.8 m/s, plus a buffer and three empty crossing steps.",74,500,1110,135,26,C.ink);
  notes(s,"The project verifies SUMO conflict foes against every phase and checks observed states. Clearance uses a 3 s buffer and remains all-red if a person is still on a crossing. These experimental constraints do not reproduce official Singapore timing plans. Source: README.md, src/traffic_control/signals.py, docs/measurement_safety_audit.md.");
}
// 6. Measurement
{
  const s=base("Pedestrian wait comes from crossing events");
  row(s,"Arrival","First observed near the crossing entrance, or at the start of a stopped approach queue",187);
  row(s,"Entry","First simulation step on the crossing edge",315);
  row(s,"Reported wait","Entry time minus arrival time",431);
  text(s,"At the horizon, keep requests that never entered. Report mean, p95, entries, and unserved by crossing.",73,556,1120,98,25,C.sea,true);
  notes(s,"Approach detection uses a 1.2 m endpoint threshold. Stopped queue detection handles persons farther upstream. Direct entries are flagged with zero observed wait. A development audit downsampled one shared 0.1-second trajectory to one-second observations: 30 entered events and two pending requests, mean absolute wait difference 0.49 s, maximum 1.30 s. This is an observed sampling error on one fixture, not a universal error bound or physical ground truth. TraCI person waitingTime is only a queue trigger. For vehicles report completed-trip timeLoss beside scheduled, completed, unfinished and uninserted trips. Official SUMO documentation: https://sumo.dlr.de/docs/Simulation/Pedestrians.html and https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html. Project audit: docs/measurement_safety_audit.md and results/measurement-resolution/summary.json.");
}
// 7. Final protocol
{
  const s=base("Final comparison protocol");
  text(s,"30",75,194,262,105,76,C.sea,true);
  text(s,"independent trained replicas",76,304,410,76,27,C.ink);
  text(s,"10",538,194,260,105,76,C.sea,true);
  text(s,"locked test seeds",540,304,340,76,27,C.ink);
  text(s,"13",950,194,220,105,76,C.sea,true);
  text(s,"demand cases",951,304,255,76,27,C.ink);
  text(s,"Paired runs share scheduled demand. Validation seeds select checkpoints. Final seeds are untouched during tuning.",75,490,1100,130,27,C.ink);
  notes(s,`The 13 cases are nine low/medium/high vehicle x pedestrian combinations, plus slow walkers, concentrated crossing burst, overload, and short signal spacing. ${finalStudy.ready?`The prespecified validation-only gate selected ${finalStudy.decision.selected_training_episodes} episodes per PPO replica.`:"Each PPO replica starts with 150 episodes; the prespecified validation-only gate decides whether all configurations extend to 300."} The locked final demand seeds are recorded in config/evaluation.json. The earlier 7001–7010 safety run and partial 8001–8010 run are excluded; policies were retrained under the revised common phase rule. Source: config/training.json, config/evaluation.json, runs/training-v3/extension_decision.json, docs/safety_revision_v3.md.`);
}
// 8. Decision rule
{
  const s=base("The all-crossing decision rule");
  text(s,"A descriptive pass needs all three",75,172,1100,75,31,C.sea,true);
  row(s,"Vehicle","Lower completed-trip time loss on paired demand",277);
  row(s,"Pedestrian","No increase in p95 wait at any of 12 crossings",393);
  row(s,"Service","No fewer completed trips; no more unfinished, uninserted, or unserved pedestrian requests",512);
  notes(s,"Report the largest crossing-level p95 difference, all 12 crossing estimates with counts, and requests still unserved at the horizon. A stronger statistical claim also needs paired confidence intervals supporting both vehicle improvement and the worst-crossing result, with adequate crossing entries. Sparse p95 values are inconclusive. Source: user-approved implementation plan and README.md.");
}
// 9. Exploratory data, explicitly separated
{
  const s=base("Exploratory gain and crossing cost","Earlier protocol: one training run per method, five test seeds, stopped-approach wait definition");
  text(s,"29.3 s",76,228,490,110,64,C.ink,true);
  text(s,"Actuated vehicle time loss",77,344,515,52,25,C.soft);
  text(s,"24.9 s",673,228,490,110,64,C.sea,true);
  text(s,"Pedestrian-aware IPPO",674,344,505,52,25,C.soft);
  text(s,"6 of 12 crossings had higher p95 wait under IPPO.",77,482,1095,70,33,C.amber,true);
  text(s,"These estimates do not meet the proposed every-crossing target.",77,569,1096,65,24,C.ink);
  notes(s,"These values come only from results/REPORT.md and results/crossing_summary.csv, the old exploratory experiment. They use the earlier stopped-approach crossing-wait measurement and should never be pooled with the new event tracker or presented as final v2 evidence. Several crossing p95 estimates had only 5–10 entries across five runs. All 192 medium-demand scheduled vehicles completed in both methods. Source: results/REPORT.md and user-approved plan.");
}
// 10. Final paired outcome; IPPO-w3 is fixed in advance as the presentation example.
{
  if(!finalStudy.ready) {
    const s=base("Final paired result","Awaiting the complete 30-replica, 10-seed, 13-scenario study");
    text(s,"Vehicle outcome",76,214,505,57,31,C.sea,true);
    text(s,"Paired time-loss difference\nCompleted, unfinished, uninserted",77,290,495,118,27,C.ink);
    text(s,"Pedestrian outcome",677,214,505,57,31,C.sea,true);
    text(s,"All 12 p95 differences\nEntry counts and unserved requests",678,290,500,118,27,C.ink);
    text(s,"Final evaluation incomplete. No final numerical result is shown.",77,518,1090,105,30,C.amber,true);
    notes(s,`${finalStudy.reason} The slide will populate automatically only after the complete final protocol is saved in results/final-study/summary.json and crossing_comparison.csv. Slide 9 remains exploratory and uses a different wait definition. Do not present it as final evidence.`);
  } else {
    const {result,selected,label,primary}=finalStudy;
    const baseline=result.summary[primary].actuated;
    const model=result.summary[primary][label];
    const differences=result.paired_vs_actuated[primary][label];
    const criterion=result.primary_crossing_criterion[label];
    const maximum=criterion.maximum_crossing_p95_difference;
    const higher=criterion.crossings_with_higher_p95.length;
    const adequate=maximum.all_crossings_adequate;
    const s=base("Final paired result: IPPO weight 3",
      "Primary synthetic case · compared with pedestrian actuated · 5 training runs and 10 locked seeds");
    text(s,"Vehicle time loss difference (s)",73,179,530,40,22,C.sea,true);
    text(s,interval(differences.vehicle_time_loss),74,216,540,54,29,C.ink,true);
    text(s,"Worst crossing p95 difference (s)",73,294,535,40,22,C.sea,true);
    text(s,interval(maximum),74,331,540,54,29,
      Number(maximum.mean)>0?C.amber:C.ink,true);
    text(s,`${higher} of 12 crossings had higher p95 wait.`,73,407,535,43,23,
      higher?C.amber:C.sea,true);
    const targetStatus=criterion.descriptive_criterion_met?"PASS":
      maximum.all_crossings_measured?"FAIL":"INCONCLUSIVE";
    text(s,`All-crossing target: ${targetStatus}`,
      73,448,535,34,22,criterion.descriptive_criterion_met?C.sea:C.amber,true);
    text(s,`Statistical support: ${criterion.statistically_supported?"YES":"NO"}`,
      73,479,535,34,21,criterion.statistically_supported?C.sea:C.amber,true);
    text(s,"Service per run: IPPO vs actuated",73,518,535,30,19,C.soft,true);
    for(const [index,[name,labelText]] of [
      ["completed","Completed trips"],["unfinished","Unfinished trips"],
      ["uninserted","Uninserted trips"],["unserved_requests","Unserved requests"],
    ].entries()) {
      const modelMean=model[name].mean;
      const baselineMean=baseline[name].mean;
      text(s,`${labelText}: ${plainNumber(modelMean)} vs ${plainNumber(baselineMean)}`,
        74,546+index*27,533,27,18,C.ink);
    }
    text(s,"Crossing p95 change (s)",642,180,550,35,22,C.sea,true);
    text(s,"Entries",1038,206,171,27,16,C.soft,true);
    const finite=selected.flatMap(item=>["p95_difference_seconds","ci95_low","ci95_high"]
      .map(key=>item[key]===""?null:Number(item[key]))
      .filter(value=>value!==null && Number.isFinite(value)));
    const maxAbs=Math.max(1,...finite.map(Math.abs));
    const zero=881, halfWidth=118;
    rule(s,zero,237,2,339,"#B6C6C8");
    for(const [index,item] of selected.entries()) {
      const y=237+index*28;
      const delta=item.p95_difference_seconds===""?null:Number(item.p95_difference_seconds);
      const low=item.ci95_low===""?null:Number(item.ci95_low);
      const high=item.ci95_high===""?null:Number(item.ci95_high);
      const cx=value=>zero+Math.max(-maxAbs,Math.min(maxAbs,value))*halfWidth/maxAbs;
      text(s,item.crossing,643,y-3,101,26,16,C.ink);
      if(delta!==null && Number.isFinite(delta)) {
        if(low!==null && high!==null && Number.isFinite(low) && Number.isFinite(high))
          rule(s,cx(low),y+11,Math.max(2,cx(high)-cx(low)),2,"#7D969D");
        const point=cx(delta);
        rule(s,Math.min(zero,point),y+6,Math.max(3,Math.abs(point-zero)),11,
          delta>0?C.amber:C.sea);
        rule(s,point-2,y+3,4,17,C.ink);
      } else text(s,"NA",840,y-3,80,25,16,C.amber);
      const entries=`${item.actuated_entries}/${item.minimum_model_entries_per_training_run}`;
      text(s,entries,1038,y-3,171,26,16,
        item.adequate_entries==="True"?C.soft:C.amber);
    }
    text(s,signed(-maxAbs,1),751,584,90,25,15,C.soft);
    text(s,"0",876,584,35,25,15,C.soft);
    text(s,signed(maxAbs,1),980,584,70,25,15,C.soft);
    text(s,adequate?"All 12 crossing entry counts meet the threshold.":
      "Sparse entries: interpret crossing p95 estimates cautiously.",
      640,622,570,48,17,adequate?C.soft:C.amber);
    text(s,"Bars include paired 95% intervals. Entries are actuated / smallest IPPO replica. All settings are in REPORT.md.",
      74,670,1135,28,16,C.soft);
    const serviceNotes=["completed","unfinished","uninserted","unserved_requests"]
      .map(name=>`${name}: IPPO ${interval(model[name])}, actuated ${interval(baseline[name])}, paired difference ${interval(differences[name])}`)
      .join("; ");
    notes(s,`Final study outputs: results/final-study/summary.json and crossing_comparison.csv. IPPO weight 3 was fixed in the slide builder as the illustration; all six PPO settings and three conventional controllers appear in the report. Negative vehicle difference means less completed-trip time loss. Positive crossing difference means longer p95 wait. The crossing panel shows each point and its paired 95% interval; entry labels give baseline pooled entries and the minimum entries in any IPPO training replica. Vehicle paired interval: ${interval(differences.vehicle_time_loss)} seconds. Max-across-crossings paired interval: ${interval(maximum)} seconds. Descriptive target ${targetStatus.toLowerCase()}; statistical support ${criterion.statistically_supported?"present":"absent"}. Crossing count adequate: ${adequate}. ${serviceNotes}. All claims concern this synthetic SUMO scenario, not deployed Singapore signals.`);
  }
}
// 11. Singapore pilot
{
  const derived=JSON.parse(await fs.readFile(
    path.join(workspaceDir,"config","singapore_derived.json"),"utf8"));
  const centroids=derived.junction_site_centroids_m;
  const gaps=derived.junction_spacings_m;
  if(centroids?.length!==3 || centroids.some(point=>point?.length!==2) || gaps?.length!==2)
    throw new Error("Singapore pilot slide requires three centroids and two gaps");
  const s=base("NUS-area virtual corridor","Three approximate OSM signal-site centroids");
  // Preserve the relative easting/northing positions; all drawn roads are synthetic.
  const eastings=centroids.map(point=>point[0]);
  const northings=centroids.map(point=>point[1]);
  const minE=Math.min(...eastings), maxE=Math.max(...eastings);
  const minN=Math.min(...northings), maxN=Math.max(...northings);
  const points=centroids.map(([e,n])=>({
    x:160+(e-minE)/(maxE-minE)*520,
    y:520-(n-minN)/(maxN-minN)*260,
  }));
  const line=(a,b,width,color)=>s.shapes.add({
    geometry:"line",
    position:{left:Math.min(a.x,b.x),top:Math.min(a.y,b.y),
      width:Math.abs(a.x-b.x),height:Math.abs(a.y-b.y),
      verticalFlip:(a.x-b.x)*(a.y-b.y)<0},
    fill:"none",line:{style:"solid",fill:color,width},
  });
  for(const p of points) {
    line({x:p.x-37,y:p.y+67},{x:p.x+37,y:p.y-67},7,"#A5B5B8");
  }
  line(points[2],points[1],9,C.sea);
  line(points[1],points[0],9,C.sea);
  for(const p of points) {
    s.shapes.add({geometry:"ellipse",
      position:{left:p.x-15,top:p.y-15,width:30,height:30},
      fill:C.paper,line:{style:"solid",fill:C.sea,width:4}});
  }
  text(s,"C2",points[2].x-80,points[2].y-42,70,35,24,C.ink,true);
  text(s,"C1",points[1].x+22,points[1].y-15,70,35,24,C.ink,true);
  text(s,"C0",points[0].x+22,points[0].y+5,70,35,24,C.ink,true);
  text(s,`${gaps[1]} m`,270,276,130,44,31,C.sea,true);
  text(s,`${gaps[0]} m`,524,410,130,44,31,C.sea,true);
  text(s,"Straight synthetic links\nand side arms",820,224,370,95,26,C.ink);
  text(s,"Synthetic vehicle and\npedestrian demand",820,350,370,95,26,C.ink);
  text(s,"Site identities unverified",820,478,370,45,24,C.amber,true);
  text(s,"No calibrated claim",820,534,370,54,29,C.amber,true);
  text(s,transferStudy?"Separate geometry check: 330 runs, 10 matched demand seeds, all 30 PPO replicas.":
    "Separate repeated geometry check awaiting the final trained replicas.",
    75,624,1120,42,19,C.soft);
  text(s,"Synthetic demand and assumed road links limit this to a geometry sensitivity check.",
    75,662,1120,38,17,C.amber);
  notes(s,`Centroids and straight-line gap labels come from config/singapore_derived.json: ${JSON.stringify(centroids)}, ${JSON.stringify(gaps)} m. The diagram preserves approximate relative centroid coordinates only. Links, side arms, pedestrian crossings, phases, and demand are synthetic. ${transferStudy?`The separate 330-run transfer uses ten seeds 9101–9110, all 30 selected-budget replicas and three conventional controls. IPPO w3 paired vehicle loss: ${interval(transferStudy.paired_vs_actuated["ippo-w3"].vehicle_time_loss)} s. Descriptive crossing target: ${transferStudy.crossing_criterion_vs_actuated["ippo-w3"].descriptive_criterion_met?"met":"not met or inconclusive"}. All crossing tails, counts, uncertainty and unserved requests are in results/singapore-transfer/REPORT.md and its tables.`:"Repeated transfer outputs are still incomplete."} The earlier seed-61 v2-model interface check is excluded. This is not a Singapore performance estimate. The candidate site grouping, site identities, actual road connectivity, junction geometry, observed junction turning counts, crossing-arrival counts, and actual LTA timing plans remain unverified. The raw OSM import had no signal-controlled crossing edges. See docs/singapore.md, scenario/singapore provenance and LTA catalogue: https://datamall.lta.gov.sg/content/datamall/en/static-data.html. SUMO OSM import caveat: https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html.`);
}
// 12. Interpretation
{
  const s=base("What the evidence can support");
  text(s,finalStudy.ready?"Earlier exploratory outcome":"Current conclusion",75,183,1050,60,29,C.sea,true);
  text(s,"The exploratory vehicle gain did not satisfy the all-crossing target.",76,252,1080,101,34,C.ink,true);
  if(finalStudy.ready) {
    const criteria=finalStudy.result.primary_crossing_criterion;
    const learned=["ippo-w0","ippo-w3","ippo-w10","mappo-w0","mappo-w3","mappo-w10"];
    const passed=learned.filter(label=>criteria[label]?.descriptive_criterion_met).length;
    const supported=learned.filter(label=>criteria[label]?.statistically_supported).length;
    text(s,`${passed} of 6 PPO settings met the descriptive all-crossing target; ${supported} had statistical support.`,
      76,407,1082,110,29,C.ink);
  } else {
    text(s,"The repeated-seed study will test whether this persists under crowding, bursts, slow walkers, and short signal spacing.",76,407,1082,110,29,C.ink);
  }
  text(s,"Any claim remains about simulation, not deployed lights.",76,582,1070,60,25,C.soft);
  notes(s,`The first sentence refers to the earlier exploratory result and its earlier crossing wait definition. ${finalStudy.ready?"The second sentence gives the completed final protocol's count of descriptive and statistically supported PPO settings.":"The final repeated-seed evaluation is still incomplete."} The final short-spacing log contains 28 pedestrian-jam warnings. SUMO jam recovery ignores obstacles; a replay reproduced saved outcomes and held vehicle links red throughout sampled crossing occupancy. Pre-entry waiting excludes traversal jams. See results/reproducibility/pedestrian_jam_audit.json and https://sumo.dlr.de/docs/Simulation/Pedestrians.html#jamming. Do not promise an A grade or a publishable result. Discuss the null case if MAPPO does not change decisions under coupling. For a wider paper, independent observed-demand calibration and multiple sites would be required. Source: user-approved plan, results/REPORT.md, README.md${finalStudy.ready?", results/final-study/summary.json":""}.`);
}
// 13. Provisional individual contributions, disclosure, Q&A
{
  const s=base("Team contributions and Q&A");
  const assignments=[
    ["Jason","Network and vehicle routes"],
    ["Jarrell","Baseline signals and GUI"],
    ["Shricharan","PPO training pipeline"],
    ["Jewi","Reward and policy checks"],
    ["Reet","Crossing event analysis"],
    ["Huan Kiat","Uncertainty and figures"],
  ];
  for(const [index,[name,work]] of assignments.entries()) {
    const y=179+index*65;
    text(s,name,77,y,195,34,24,C.sea,true);
    text(s,work,270,y,385,34,21,C.ink);
  }
  text(s,"Provisional: each member must verify actual work before submission.",
    77,613,598,60,18,C.amber,true);
  text(s,"OpenAI Codex generated and revised code, ran simulations and tests, and drafted analysis and text. The team must independently verify the work.",
    718,230,455,232,26,C.ink);
  text(s,"Questions",676,552,470,78,42,C.sea,true);
  notes(s,"The named tasks are provisional draft assignments based on the role areas in AGENTS.md and the project description, not verified claims about actual individual work. The user requested placeholders to edit; replace every line with factual contributions before submission. Everyone contributes to writing and presentation. Timing: roughly 1 minute per slide, keeping 5 minutes for Q&A. Likely questions: (1) How does the measured arrival event differ from SUMO waitingTime? (2) Why are sparse crossing p95 values inconclusive? (3) Why does the Singapore pilot not constitute field validation? (4) Does MAPPO really differ from IPPO under short spacing? The team must verify and disclose exact LLM assistance in the final submitted materials.");
}

const candidatePath=path.join(TMP_DIR,"candidate.pptx");
await (await PresentationFile.exportPptx(P)).save(candidatePath);
const output=await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath:FINAL_PPTX,
  pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-heading-fit"],
  fontPolicy:{basis:"design",families:[font]},
  explicitTotalSlideCount:13,
  requiredNativeTableOwnerSlides:[],
  requiredNativeChartOwnerSlides:[],
  verifyArtifactToolImport:true,
  receiptPath:path.join(TMP_DIR,`${path.basename(FINAL_PPTX)}.validation.json`),
});
console.log(JSON.stringify({output,finalPath:FINAL_PPTX}));
