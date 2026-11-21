// Non-scoring diagnostic for a frozen p-RL1024 engine source.
// Instrument only actor-proposed battle actions. No modification to Battle,
// legal actions, opponent information, rollout, or the hard 3,000-decision cap.
// A proposal in a failed Match is not a verified authority-accepted action.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const args=process.argv.slice(2);
if(args.length!==8)throw new Error('Usage --seed N --teacher BASE128 --candidate C005 --out FILE');
const input=new Map();
for(let i=0;i<args.length;i+=2) {
  if(!['--seed','--teacher','--candidate','--out'].includes(args[i])||
     input.has(args[i]))throw new Error('Unexpected or duplicate switch probe option');
  input.set(args[i],args[i+1]);
}
const seed=Number(input.get('--seed'));
if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
   [...['--teacher','--candidate','--out']].some(k=>!input.get(k)))
  throw new Error('Invalid switch audit inputs');

const base=resolve(input.get('--teacher'));
const candidate=resolve(input.get('--candidate'));
const sha=x=>createHash('sha256').update(readFileSync(x)).digest('hex');
const manifest=JSON.parse(readFileSync(resolve('../models/manifest.json'),'utf8'));
if(manifest.schema!=='rnad-i5-queued-evaluation-models-v1'||
   manifest.source_sha!=='f08a9587d6a7033546eab00ffa6479399ca71c2f'||
   manifest.engine_sha!=='ea81165993110ee9fb3a09dcb97c6708c0d7397b'||
   manifest.models.base128!==sha(base)||
   manifest.models.c005!==sha(candidate)||
   manifest.is_champion!==false)
  throw new Error('Mismatched immutable source/models');

const imported=async p=>import(pathToFileURL(resolve(p)).href);
const {RNADMixedPolicy,PersistentRNADInference}=await imported('scripts/rnad/infer.mjs');
const {directPairedNeuralEval}=await imported('scripts/rnad/evaluate-direct.mjs');
const {runPhaseMatch}=await imported('scripts/dev/simulator/phase-match.js');
const {createBuiltinPhaseAgents}=await imported('scripts/dev/simulator/phase-agents.js');
const modelSignatures=Object.entries({base128:base,c005:candidate}).map(([name,file])=>{
  const model=new PersistentRNADInference(file);
  return {name,head:model.params['policy_out.weight'].slice(0,8)};
});
const idOf=model=>{
  const head=model.params['policy_out.weight'];
  const hits=modelSignatures.filter(item=>item.head.every((n,i)=>head[i]===n));
  if(hits.length!==1)throw new Error('Unexpected instrumented model fingerprint');
  return hits[0].name;
};
const proposals=[];let game=-1;
const originalChoose=RNADMixedPolicy.prototype.choose;
RNADMixedPolicy.prototype.choose=function(ctx) {
  const result=originalChoose.call(this,ctx);
  const model=idOf(this.model);
  const obs=ctx.observation;
  if(obs.phase==='draft'&&ctx.seat===0)game++;
  if(obs.phase==='battle'){
    const action=result.action;
    proposals.push({
      game,model,seat:ctx.seat,round:obs.context?.round??null,
      turn:obs.turn,requestType:obs.requestType,
      from:obs.self.active,
      action:action.kind,
      to:action.kind==='switch'?action.index:null,
      forced:obs.requestType!=='turn',
      ownHp:obs.self.team?.[obs.self.active]?.hp??null
    });
  }
  return result;
};
let outcome=null,issue=null;
try {
  outcome=directPairedNeuralEval({
    teacherPath:base,rnadPath:candidate,seed,pairs:1,selectionMode:'greedy'
  });
} catch(e) {
  if(!/^Decision cap exceeded in round [1-9][0-9]*$/.test(e.message))
    throw e;
  issue=e.message;
} finally {
  RNADMixedPolicy.prototype.choose=originalChoose;
}
function summarize(events) {
  const groups={};
  const last={};
  const recentSwitches={};
  for(const row of events){
    const key=`${row.game}:${row.model}:${row.seat}`;
    const x=groups[key]??={game:row.game,model:row.model,seat:row.seat,
      normalTurns:0,voluntarySwitches:0,forcedSwitches:0,
      normalMoves:0,forcedMoves:0,reversals:0,
      repeatedVoluntarySwitches:0,mostRecentSwitchPairs:[]};
    const forced=row.forced;
    if(!forced)x.normalTurns++;
    if(row.action==='switch'){
      if(forced)x.forcedSwitches++;else x.voluntarySwitches++;
      const prev=recentSwitches[key];
      if(prev&&row.round===prev.round&&
         row.from===prev.to&&row.to===prev.from){
        x.reversals++;
      }
      x.mostRecentSwitchPairs.push([row.from,row.to,row.turn]);
      if(x.mostRecentSwitchPairs.length>12)x.mostRecentSwitchPairs.shift();
      recentSwitches[key]=row;
    } else if(row.action==='move'){
      if(forced)x.forcedMoves++;else x.normalMoves++;
    }
    if(row.action==='switch'&&!forced&&last[key]?.action==='switch'&&
       !last[key].forced)x.repeatedVoluntarySwitches++;
    last[key]=row;
  }
  for(const x of Object.values(groups))
    x.voluntarySwitchRate=x.normalTurns?
      x.voluntarySwitches/x.normalTurns:null;
  return Object.values(groups).sort((a,b)=>a.game-b.game||a.seat-b.seat);
}
function referenceAgents(battleName,role){
  const wrapped=createBuiltinPhaseAgents({battle:battleName});
  const original=wrapped.battle;
  wrapped.battle={
    id:original.id,
    select:ctx=>original.select(ctx),
    accepted:({action,view})=>{
      baselineActions.push({role,policy:battleName,action:action.kind,
        forced:view.requestType!=='turn',
        turn:view.turn,from:view.self.active,
        to:action.kind==='switch'?action.index:null});
    }
  };
  return wrapped;
}
const baselineActions=[];let baselineError=null,baselineOutcome=null;
try{
  const result=runPhaseMatch({seed,agents:[
    referenceAgents('basic','seat0'),referenceAgents('intermediate','seat1')
  ]});
  baselineOutcome={completed:result.completed,score:result.score,rounds:result.rounds};
}catch(e){
  if(!/^Decision cap exceeded in round [1-9][0-9]*$/.test(e.message))throw e;
  baselineError=e.message;
}
const benchmark=Object.fromEntries(['basic','intermediate'].map(p=>{
  const rows=baselineActions.filter(x=>x.policy===p);
  const normal=rows.filter(x=>!x.forced);
  return [p,{normalDecisions:normal.length,
    voluntarySwitches:normal.filter(x=>x.action==='switch').length,
    voluntarySwitchRate:normal.length?
      normal.filter(x=>x.action==='switch').length/normal.length:null,
    forcedSwitches:rows.filter(x=>x.forced&&x.action==='switch').length}];
}));
const result={
  schema:'rnad-003-switch-proposal-sequence-diagnostic-v1',
  source_sha:manifest.source_sha,engine_sha:manifest.engine_sha,
  model_sha256:{base128:sha(base),c005:sha(candidate)},
  seed,match_completed:issue===null&&outcome?.matches===2,
  decision_cap_error:issue,
  match_score:outcome?{wins:outcome.wins,matches:outcome.matches}:null,
  proposed_battle_decisions:proposals.length,
  model_and_seat_breakdown:summarize(proposals),
  last_120_proposed_battle_decisions:proposals.slice(-120),
  heuristic_reference:{seed,basic_vs_intermediate:benchmark,
    completed:baselineOutcome,decision_cap_error:baselineError},
  // No stored opponent private roster, unseen moves or game checkpoint.
  // Selection proposals are NOT full authority receipts in a failed Match.
  proposal_trace_is_not_authority_receipt:true,
  is_champion:false
};
writeFileSync(resolve(input.get('--out')),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({
  seed,issue,proposals:proposals.length,groups:result.model_and_seat_breakdown,
  builtin:result.heuristic_reference
}));
