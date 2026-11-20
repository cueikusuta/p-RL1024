// Research-only, frozen source-pinned G48 Draft/Selection switch experiment.
// Added to the source-locked checkout as an SHA-verified overlay, not a trainer.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {chooseStrategyTeam} from '../../aq-runtime/src/ai/team-assessment.js';
import {chooseTacticalSelection} from '../../aq-runtime/src/ai/selection-tactical.js';
import {playableSets} from '../../aq-runtime/src/production-runtime.js';
import data from '../../aq-runtime/data/catalog/battle-data.json' with {type:'json'};
import {createIntegratedAQSession} from './evaluate-shared-g012-vs-aq.mjs';
import {session} from './shared_league_actor.mjs';
import {projectRnadObservation,rnadLegalActions} from '../../src/ml/legal-action-adapter.js';
import {createOpponent,weightsSha256 as AQ_SHA} from '../../models/aq-base.mjs';
import {pairedClusterCI,pairedSignSensitivity} from './paired-league-stats.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export const SCHEMA='rnad-i5-g48-phase-ablation-shard-v1';
const PLAN_SCHEMA='rnad-i5-g48-phase-ablation-plan-v1';
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const assert=(yes,message)=>{if(!yes)throw Error('Frozen phase ablation: '+message);};
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const MODES=['g48','aq-draft','aq-selection','aq-both'];
const FROZEN_G48_SHA='c06ccc66ead1ae83e4a39054f72a59222b2461ed818823489e7eb025cba1cb46';
const FROZEN_SOURCE='01501f3a1e12511140bf772d435d39c0c07319bc';
const ENGINE='ea81165993110ee9fb3a09dcb97c6708c0d7397b';
const N=2048, FIRST=10760001;

function validatePlan(plan,modelDir) {
  assert(plan?.schema===PLAN_SCHEMA&&plan.seed_start===FIRST&&
    plan.seed_blocks===N&&same(plan.modes,MODES)&&
    plan.comparisons===3&&plan.familywise_alpha===.05&&
    plan.minimum_practical_gain===.02&&plan.fixed_sample_size===true&&
    plan.model_manifest_sha256===FROZEN_G48_SHA&&
    plan.source_sha===FROZEN_SOURCE&&plan.engine_sha===ENGINE&&
    plan.aq_sha256===AQ_SHA&&plan.champion_promoted===false,
    'Unregistered study plan');
  assert(verifiedRunningSourceSha()===FROZEN_SOURCE&&
    provenance.sourceCommit===ENGINE,'Unpinned source / engine');
  const m=JSON.parse(fs.readFileSync(path.join(modelDir,'manifest.json')));
  assert(sha(path.join(modelDir,'manifest.json'))===FROZEN_G48_SHA&&
    m.generation===49&&m.source_sha===FROZEN_SOURCE&&
    m.engine_sha===ENGINE&&
    same(Object.keys(m.files).sort(),['battle','draft','selection','values']),
    'Unregistered frozen post-G48 policy bundle');
  for(const [head,digest] of Object.entries(m.files))assert(
    sha(path.join(modelDir,head+'.json'))===digest,'Head SHA drift '+head);
  assert(createOpponent().modelSha256===AQ_SHA,'Frozen AQ changed');
}

export function mixedSession({modelDir,seat,mode}) {
  assert(MODES.includes(mode)&&[0,1].includes(seat),'Invalid mode or seat');
  const frozen=session({modelDir,magnetDir:modelDir,epsilon:.02,greedy:true});
  let round=null,pending=null,accepted=0;
  const byPhase={draft:0,selection:0,battle:0};
  const history={lastKind:'none',consecutiveSwitches:0,lastSwitchTarget:-1,lastActive:-1};
  function select(view) {
    assert(pending===null,'Uncommitted phase proposal');
    const obs=projectRnadObservation(view,{seat});
    if(obs.phase==='draft'){
      assert(Number.isSafeInteger(obs.round)&&obs.round>=1&&obs.round<=64&&
        (round===null||obs.round===round+1),'Round out of sequence');
      round=obs.round;
    }
    assert(Number.isSafeInteger(round)&&round>0,'No accepted round');
    obs.acceptedHistory={...history};
    const legal=rnadLegalActions(view);
    const external=(obs.phase==='draft'&&['aq-draft','aq-both'].includes(mode))||
      (obs.phase==='selection'&&['aq-selection','aq-both'].includes(mode));
    const chosen=external?
      obs.phase==='draft'?chooseStrategyTeam(view,{data,fastDraftValidation:true}):
        chooseTacticalSelection(view,{data,sets:playableSets}):
      frozen.func({observation:obs,legalActions:legal,seat,round}).action;
    assert(legal.some(a=>same(a,chosen)),'Invalid authoritative candidate');
    pending={action:structuredClone(chosen),phase:obs.phase,
      requestType:obs.requestType??null,active:obs.self?.active??-1,
      round,external};
    return chosen;
  }
  function commit() {
    assert(pending!==null,'Missing accepted phase proposal');
    const {action,phase,requestType,active,round:r,external}=pending;
    if(!external)frozen.func.accepted({seat,round:r,phase,action});
    if(phase==='battle'&&requestType==='turn'){
      history.lastActive=Number.isSafeInteger(active)?active:-1;
      if(action?.kind==='switch'){
        history.lastKind='switch';
        history.consecutiveSwitches=Math.min(3000,history.consecutiveSwitches+1);
        history.lastSwitchTarget=Number.isSafeInteger(action.index)?action.index:-1;
      }else{
        history.lastKind='move';
        history.consecutiveSwitches=0;
        history.lastSwitchTarget=-1;
      }
    }else{
      history.lastKind=phase==='draft'?'draft':
        phase==='selection'?'selection':'none';
      history.consecutiveSwitches=0;
      history.lastSwitchTarget=-1;
      history.lastActive=-1;
    }
    byPhase[phase]++;accepted++;pending=null;
  }
  const stats=()=>({schema:'rnad-i5-authority-accepted-g48-aq-phase-ablation-session-v1',
    accepted_decisions:accepted,accepted_by_phase:{...byPhase},pending:pending!==null,
    updates:0,train:false,mode,phase_mode:'frozen-greedy'});
  function finish(reward) {
    assert(pending===null&&[-1,1].includes(reward)&&
      Object.values(byPhase).every(n=>n>0),'Invalid four-win phase coverage');
    return {...stats(),terminal_reward:reward};
  }
  return {select,commit,stats,finish};
}

function play({seed,mode,seat,modelDir,aq}) {
  const candidate=mixedSession({modelDir,seat,mode});
  const result=runStrategyMatch({
    seed,policies:[0,1].map(s=>s===seat?'pure-rl-linear':'aq-base'),
    pureRlSessions:[0,1].map(s=>s===seat?candidate:null),
    pureRlAllPhases:true,strategyMode:'integrated',
    v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq,
  });
  const stats=result.pureRlStats?.[seat];
  assert(result.completed===true&&[0,1].includes(result.winner)&&
    Array.isArray(result.score)&&result.score.length===2&&
    result.score[result.winner]===4&&result.score[1-result.winner]<4&&
    result.rounds>=4&&result.rounds<=64&&
    stats?.updates===0&&stats?.pending===false&&
    stats?.accepted_decisions===result.actionCounts[seat]+2*result.rounds,
    'Incomplete/bad authoritative four-win Match');
  return {seed,candidateSeat:seat,won:result.winner===seat,completed:true,
    score:result.score,rounds:result.rounds,accepted_decisions:stats.accepted_decisions};
}
export function smoke({modelDir}) {
  const aq=createOpponent();
  for(let seed=10759998;seed<=10759999;seed++)
    for(const seat of [0,1]){
      const reference=createIntegratedAQSession({modelDir,seat});
      const original=runStrategyMatch({
        seed,policies:[0,1].map(s=>s===seat?'pure-rl-linear':'aq-base'),
        pureRlSessions:[0,1].map(s=>s===seat?reference:null),
        pureRlAllPhases:true,strategyMode:'integrated',
        v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq});
      const comparison=play({seed,mode:'g48',seat,modelDir,aq});
      assert(original.completed&&original.winner===(comparison.won?seat:1-seat)&&
        same(original.score,comparison.score)&&
        original.rounds===comparison.rounds&&
        original.pureRlStats[seat].accepted_decisions===comparison.accepted_decisions,
        'Original immutable pure G48 parity failed');
      for(const mode of ['aq-draft','aq-selection','aq-both'])
        play({seed,mode,seat,modelDir,aq});
    }
  return {schema:'rnad-i5-phase-ablation-smoke-v1',
    independent_pure_g48_baseline_parity:true,smoke_seeds:2,
    smoke_strength_claim:false,model_sha256:FROZEN_G48_SHA};
}
export function evaluate({plan,modelDir,mode,part,parts=8}) {
  validatePlan(plan,modelDir);
  assert(MODES.includes(mode)&&Number.isInteger(part)&&part>=0&&part<parts&&
    parts===8&&N%parts===0,'Unregistered phase-shard partition');
  const aq=createOpponent(),first=FIRST+part*(N/parts),rows=[];
  for(let seed=first;seed<first+N/parts;seed++)
    for(const seat of [0,1])
      rows.push(play({seed,mode,seat,modelDir,aq}));
  return {schema:SCHEMA,mode,part,source_sha:FROZEN_SOURCE,
    engine_sha:ENGINE,model_manifest_sha256:FROZEN_G48_SHA,
    aq_sha256:AQ_SHA,seed_start:first,seed_blocks:N/parts,
    matches:rows.length,wins:rows.filter(x=>x.won).length,
    policy:'source-frozen-G48-greedy-verified-AQ-phase-swap',
    training_during_evaluation:false,champion_promoted:false,rows};
}
export function summarize({plan,reports}) {
  assert(plan?.schema===PLAN_SCHEMA&&plan.seed_blocks===N&&
    same(plan.modes,MODES)&&plan.familywise_alpha===.05&&
    plan.comparisons===3,'Bad final plan');
  assert(Array.isArray(reports)&&reports.length===32,
    'Incomplete 4 mode by 8 part study');
  const byMode=Object.fromEntries(MODES.map(k=>[k,new Map()]));
  const parts=Object.fromEntries(MODES.map(k=>[k,new Set()]));
  for(const r of reports){
    assert(r.schema===SCHEMA&&MODES.includes(r.mode)&&
      Number.isInteger(r.part)&&r.part>=0&&r.part<8&&
      !parts[r.mode].has(r.part)&&r.seed_blocks===256&&
      r.seed_start===FIRST+r.part*256&&r.matches===512&&
      r.rows?.length===512&&r.wins===r.rows.filter(x=>x.won).length&&
      r.source_sha===FROZEN_SOURCE&&r.engine_sha===ENGINE&&
      r.model_manifest_sha256===FROZEN_G48_SHA&&r.aq_sha256===AQ_SHA&&
      r.training_during_evaluation===false&&r.champion_promoted===false,
      'Invalid/duplicate phase shard');
    parts[r.mode].add(r.part);
    for(const row of r.rows){
      const key=row.seed+':'+row.candidateSeat;
      assert(row.seed>=r.seed_start&&row.seed<r.seed_start+256&&
        [0,1].includes(row.candidateSeat)&&
        row.completed===true&&typeof row.won==='boolean'&&
        row.score?.[row.won?row.candidateSeat:1-row.candidateSeat]===4&&
        row.score?.[row.won?1-row.candidateSeat:row.candidateSeat]<4&&
        Number.isInteger(row.accepted_decisions)&&row.accepted_decisions>0&&
        !byMode[r.mode].has(key),'Invalid or repeated accepted decision');
      byMode[r.mode].set(key,row);
    }
  }
  for(const mode of MODES)assert(parts[mode].size===8&&
    byMode[mode].size===2*N,'Incomplete four-arm seed/seat coverage');
  const scores={};
  for(const mode of MODES){
    scores[mode]=[];
    for(let seed=FIRST;seed<FIRST+N;seed++){
      const a=byMode[mode].get(seed+':0'),b=byMode[mode].get(seed+':1');
      assert(a&&b,'Missing paired seed/seat '+mode+':'+seed);
      scores[mode].push((Number(a.won)+Number(b.won))/2);
    }
  }
  const comparisons=['aq-draft','aq-selection','aq-both'].map(mode=>{
    const blocks=scores[mode].map((x,i)=>x-scores.g48[i]);
    const stat=pairedClusterCI(blocks,{
      familywiseAlpha:.05,comparisons:3,range:[-1,1],nullValue:0});
    return {candidate:mode,baseline:'g48',...stat,
      practical_gain_gt_2pt:stat.confidence_interval[0]>.02,
      practical_harm_gt_2pt:stat.confidence_interval[1]<-.02,
      exploratory_sign_test:pairedSignSensitivity(blocks,{nullValue:0})};
  });
  return {schema:'rnad-i5-g48-phase-ablation-final-v1',
    study_kind:'causal-phase-policy-substitution-with-battle-fixed',
    seed_start:FIRST,seed_end:FIRST+N-1,independent_seed_blocks:N,
    complete_four_win_matches:4*2*N,familywise_alpha:.05,
    comparisons,roles:Object.fromEntries(MODES.map(mode=>[mode,{
      wins:scores[mode].reduce((a,b)=>a+2*b,0),
      matches:2*N,win_rate:scores[mode].reduce((a,b)=>a+b,0)/N
    }])),source_sha:FROZEN_SOURCE,engine_sha:ENGINE,
    model_manifest_sha256:FROZEN_G48_SHA,aq_sha256:AQ_SHA,
    not_direct_g48_h2h:true,training_performed:false,
    independent_review_completed:false,champion_promoted:false,
    main_merged:false};
}

function argsOf(argv){
  const out={};
  for(let i=0;i<argv.length;i+=2){
    assert(argv[i]?.startsWith('--')&&argv[i+1]!==undefined&&!Object.hasOwn(out,argv[i]),
      'Bad command line argument');out[argv[i]]=argv[i+1];
  }
  return out;
}
const argv=process.argv.slice(2);
if(argv.length){
  const opts=argsOf(argv),kind=opts['--task'];
  if(kind==='smoke'){
    const r=smoke({modelDir:opts['--model-dir']});
    fs.writeFileSync(opts['--out'],JSON.stringify(r,null,2)+'\n',{flag:'wx'});
  }else if(kind==='evaluate'){
    const r=evaluate({plan:JSON.parse(fs.readFileSync(opts['--plan'],'utf8')),
      modelDir:opts['--model-dir'],mode:opts['--mode'],part:Number(opts['--part'])});
    fs.writeFileSync(opts['--out'],JSON.stringify(r)+'\n',{flag:'wx'});
    console.log(JSON.stringify({mode:r.mode,part:r.part,wins:r.wins,matches:r.matches}));
  }else if(kind==='summarize'){
    const files=JSON.parse(fs.readFileSync(opts['--index'],'utf8'));
    const r=summarize({plan:JSON.parse(fs.readFileSync(opts['--plan'],'utf8')),
      reports:files.map(file=>JSON.parse(fs.readFileSync(file,'utf8')))});
    fs.writeFileSync(opts['--out'],JSON.stringify(r,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({n:r.complete_four_win_matches,
      comparisons:r.comparisons.map(c=>({candidate:c.candidate,estimate:c.estimate}))}));
  }else assert(false,'Unrecognized research task');
}
