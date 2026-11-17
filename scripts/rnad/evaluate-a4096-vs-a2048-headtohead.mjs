// Research-only real four-win Match comparison, changing ONLY selection.
// Draft and battle use the identical built-in strategy/basic handlers.
// This is NOT an AQ-Base victory claim and is never a Champion gate.
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {runPhaseMatch} from '../dev/simulator/phase-match.js';
import {createBuiltinPhaseAgents} from '../dev/simulator/phase-agents.js';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {encodeRichSelectionActor}
  from './export-selection-rich.mjs';
import {loadVerifiedASelection,checkASelectionParity,selectFrozenAAction}
  from './selection-a-public-infer.mjs';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const matches=512,defaultStart=2500001;
function createAgent(p,label){
  const agents=createBuiltinPhaseAgents({
    draft:'strategy',selection:'strategy',battle:'basic'});
  agents.selection={id:label,select({view,seat}){
    const observation=projectRnadObservation(view,{seat});
    if(observation.phase!=='selection')
      throw new Error('A actor received non-selection observation');
    const legal=rnadLegalActions(view);
    const actor=encodeRichSelectionActor(observation,legal);
    const selected=selectFrozenAAction(p,actor);
    if(!Number.isInteger(selected.index)||selected.index<0||
       selected.index>=legal.length)
      throw new Error('A returned a non-authoritative candidate');
    return legal[selected.index];
  }};
  return agents;
}
function evaluateSeed(oldModel,newModel,seed){
  const games=[];
  for(const newerSeat of [0,1]){
    const agents=[0,1].map(seat=>createAgent(
      seat===newerSeat?newModel:oldModel,
      seat===newerSeat?'selection-A-new':'selection-A-frozen'));
    const result=runPhaseMatch({seed,agents});
    if(result.schema!=='p-rl1024-phase-match-v1'||
       !result.completed||
       result.score[result.winner]!==4||
       ![0,1].includes(result.winner)||
       result.rounds<4||result.rounds>64)
      throw new Error('Invalid authority-accepted full four-win Match');
    games.push({seed,newerSeat,newerWon:result.winner===newerSeat,
      rounds:result.rounds,score:result.score});
  }
  return games;
}
function bootstrap(rows){
  if(rows.length!==matches||new Set(rows.map(r=>r.seed)).size!==matches||
     rows.some(r=>r.games.length!==2))
    throw new Error('Missing exactly 512 independent complete Match seed pairs');
  const samples=rows.map(r=>r.games.filter(g=>g.newerWon).length/2);
  let seed=907183; // independent deterministic bootstrap RNG
  function rand(){
    seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;
    return (seed>>>0)/4294967296;
  }
  const boot=[];
  for(let b=0;b<10000;b++){
    let sum=0;
    for(let j=0;j<samples.length;j++)
      sum+=samples[Math.floor(rand()*samples.length)]/samples.length;
    boot.push(sum);
  }
  boot.sort((a,b)=>a-b);
  const winRate=samples.reduce((a,b)=>a+b,0)/samples.length;
  return {independent_seed_clusters:matches,
    complete_matches:2*matches,win_rate:winRate,
    approximate_two_sided_95pct_cluster_bootstrap_ci:[
      boot[250],boot[9749]],
    statistically_supported_headtohead_superiority:boot[250]>.5};
}

// Smoke is a legality/completion contract test only. Two Matches cannot
// support any comparison of gameplay strength or numerical win-rate claim.
export function smokeFrozenASelectionHeadToHead({
  modelPath,expectedSha,seed=2490001}={}){
  if(typeof modelPath!=='string'||typeof expectedSha!=='string'||
     !Number.isSafeInteger(seed)||seed<1)
    throw new TypeError('Invalid fixed A Match smoke arguments');
  const p=loadVerifiedASelection(readFileSync(modelPath),expectedSha);
  const parityError=checkASelectionParity(p);
  const games=evaluateSeed(p,p,seed);
  if(games.length!==2||games.some(g=>g.seed!==seed||
     !Number.isInteger(g.rounds)||g.rounds<4||g.rounds>64))
    throw new Error('A real authority Match smoke did not complete');
  return {schema:'rnad-i5-frozen-A-selection-real-match-contract-smoke-v1',
    seed,complete_four_win_matches:2,parityError,
    source_model_sha256:p.source_model_sha256,
    strength_established:false,league_started:false,champion_promoted:false};
}

export function evaluateFrozenASelectionHeadToHead({
  oldModel,oldSha,newModel,newSha,seed=defaultStart}={}){
  if(typeof oldModel!=='string'||typeof newModel!=='string'||
     typeof oldSha!=='string'||typeof newSha!=='string'||
     !Number.isSafeInteger(seed)||seed<1||seed+matches>0xffffffff)
    throw new TypeError('Need fixed SHA and 512 independent Match seeds');
  const old=loadVerifiedASelection(readFileSync(oldModel),oldSha);
  const newer=loadVerifiedASelection(readFileSync(newModel),newSha);
  if(old.source_sha!==newer.source_sha||
     old.engine_sha!==newer.engine_sha||
     old.public_actor_vocabulary.sha256!==newer.public_actor_vocabulary.sha256||
     old.source_sha!=='a7224b05eeef49650794b24546576119dffed951'||
     old.engine_sha!==provenance.sourceCommit)
    throw new Error('Incompatible frozen teacher/engine/public vocab provenance');
  const oldParity=checkASelectionParity(old);
  const newParity=checkASelectionParity(newer);
  const rows=[];
  for(let i=0;i<matches;i++){
    const s=seed+i,games=evaluateSeed(old,newer,s);
    rows.push({seed:s,games});
    if((i+1)%64===0)
      console.log(JSON.stringify({processed_match_clusters:i+1,
        wins:rows.reduce((acc,r)=>acc+
          r.games.filter(g=>g.newerWon).length,0)}));
  }
  const stats=bootstrap(rows);
  const report={
    schema:'rnad-i5-frozen-A4096-vs-A2048-selection-only-headtohead-v1',
    seed_start:seed,
    new_model_sha256:newer.source_model_sha256,
    old_model_sha256:old.source_model_sha256,
    old_js_sha256:oldSha,new_js_sha256:newSha,
    python_js_parity_max_errors:{old:oldParity,new:newParity},
    shared_source_sha:old.source_sha,engine_sha:old.engine_sha,
    settings:{draft:'strategy',selection:['frozen-A2048','new-A4096'],
              battle:'basic'},
    effect_is_selection_only:true,
    win_rate_vs_frozen_A2048:stats,
    real_four_win_match_result:true,
    does_not_establish_AQ_base_strength:true,
    selection_teacher_top1_not_used_for_win_rate:true,
    three_phase_integrated_league_gate_passed:false,
    league_training_started:false,
    champion_promoted:false,
    rows,
  };
  return report;
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  if(process.argv[2]==='--smoke'){
    if(process.argv.length!==8||process.argv[4]!=='--sha256'||process.argv[6]!=='--seed')
      throw new Error('Smoke usage --smoke MODEL --sha256 PINNED --seed SEED');
    console.log(JSON.stringify(smokeFrozenASelectionHeadToHead({
      modelPath:process.argv[3],expectedSha:process.argv[5],
      seed:Number(process.argv[7])})));
    process.exit(0);
  }
  const args=process.argv.slice(2),o=new Map();
  if(args.length!==12)throw new Error('Expected --old --old-sha --new --new-sha --seed --out');
  for(let i=0;i<args.length;i+=2){
    if(!['--old','--old-sha','--new','--new-sha','--seed','--out'].includes(args[i])||
       o.has(args[i])||!args[i+1])
      throw new Error('Wrong or repeated A evaluation flag');
    o.set(args[i],args[i+1]);
  }
  if(o.size!==6)throw new Error('Incomplete A evaluation flags');
  const r=evaluateFrozenASelectionHeadToHead({
    oldModel:o.get('--old'),oldSha:o.get('--old-sha'),
    newModel:o.get('--new'),newSha:o.get('--new-sha'),
    seed:Number(o.get('--seed'))});
  writeFileSync(o.get('--out'),JSON.stringify(r,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({old:r.old_model_sha256,new:r.new_model_sha256,
    winrate:r.win_rate_vs_frozen_A2048}));
}
