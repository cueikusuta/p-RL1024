// Outcome-aligned selection ranking probe. This is a BOUNDED exploratory
// intervention, not a causal Q oracle nor a Champion evidence gate.
// Frozen v3 controls draft, battle and all later rounds. Only the first
// authority-legal selection differs across matched seed/seat scenarios.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as AQ_SHA}
  from '../../models/aq-base.mjs';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {encodeRnadThreePhaseDecisionV5}
  from '../../src/ml/rnad-phase-features-v5.js';
import {PersistentRNADInference}
  from './infer.mjs';
import {RnadSwitchLoopGuard} from './switch-loop-guard.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

const sha=x=>createHash('sha256').update(x).digest('hex');
const actionKey=a=>JSON.stringify(a);
const history=()=>({lastKind:'none',consecutiveSwitches:0,
  lastSwitchTarget:-1,lastActive:-1});
export function createSelectionRankSession({model,seat,rank=0}={}){
  if(!(model instanceof PersistentRNADInference)||
     model.featureSchema!=='rnad-three-phase-features-v3'||
     ![0,1].includes(seat)||!Number.isSafeInteger(rank)||rank<0||rank>11)
    throw new TypeError('Expected frozen v3 and bounded selection rank');
  const accepted={draft:0,selection:0,battle:0};
  let round=0,pending=null,publicHistory=history(),first=null;
  const policy={
    choose({observation,legalActions,seat:s}){
      const result=model.inferProjected(observation,legalActions,s);
      let order=result.logits.map((score,i)=>({score,i}));
      order.sort((a,b)=>b.score-a.score||a.i-b.i);
      const firstSelection=observation.phase==='selection'&&accepted.selection===0;
      const index=firstSelection?order[Math.min(rank,order.length-1)].i:order[0].i;
      if(firstSelection){
        const feature=encodeRnadThreePhaseDecisionV5(observation,legalActions);
        const signature=sha(JSON.stringify({
          draft:accepted.draft,own:observation.self.team.map(p=>p.instanceId),
          publicSpecies:observation.opponent.species,
          legalActions}));
        first={selected:actionKey(legalActions[index]),rank,signature,
          candidate_features:feature.actions[index],
          state_features:feature.state,
          selection_candidate_count:legalActions.length};
      }
      return {action:legalActions[index],
        behaviorLogProbability:0,referenceLogProbability:0,
        predictedValue:result.value};
    }
  };
  const guard=new RnadSwitchLoopGuard(policy);
  function select(view){
    if(pending)throw new Error('Unaccepted selection-outcome proposal');
    const obs=projectRnadObservation(view,{seat});
    if(obs.phase==='draft'){
      if(!Number.isSafeInteger(obs.round)||obs.round!==round+1)
        throw new Error('Invalid accepted Match round order');
      round=obs.round;
    }
    if(!round)throw new Error('Missing accepted draft round');
    obs.acceptedHistory={...publicHistory};
    const legal=rnadLegalActions(view);
    const proposal=guard.choose({observation:obs,legalActions:legal,seat,round});
    if(!legal.some(a=>actionKey(a)===actionKey(proposal.action)))
      throw new Error('Outcome probe returned illegal action');
    pending={phase:obs.phase,round,requestType:obs.requestType??null,
      active:obs.self?.active??-1,action:structuredClone(proposal.action)};
    return proposal.action;
  }
  function commit(){
    if(!pending)throw new Error('Missing accepted outcome decision');
    const p=pending;
    guard.accepted({seat,round:p.round,phase:p.phase,action:p.action});
    if(p.phase==='battle'&&p.requestType==='turn'){
      publicHistory.lastActive=Number.isSafeInteger(p.active)?p.active:-1;
      if(p.action?.kind==='switch'){
        publicHistory.lastKind='switch';
        publicHistory.consecutiveSwitches=Math.min(3000,
          publicHistory.consecutiveSwitches+1);
        publicHistory.lastSwitchTarget=p.action.index;
      }else{
        publicHistory.lastKind='move';
        publicHistory.consecutiveSwitches=0;publicHistory.lastSwitchTarget=-1;
      }
    }else{
      publicHistory=history();
      publicHistory.lastKind=p.phase==='draft'?'draft':
        p.phase==='selection'?'selection':'none';
    }
    accepted[p.phase]++;
    pending=null;
  }
  const stats=()=>({schema:'rnad-i5-selection-rank-session-v1',
    accepted_decisions:Object.values(accepted).reduce((a,b)=>a+b,0),
    accepted_by_phase:{...accepted},pending:pending!==null,updates:0,train:false,
    first_selection:first});
  const finish=reward=>{
    if(pending||![-1,1].includes(reward)||!first||
       Object.values(accepted).some(n=>n<1))
      throw new Error('Incomplete accepted selection outcome probe');
    return {...stats(),terminal_reward:reward};
  };
  return {select,commit,stats,finish};
}
export function evaluateSelectionRankOutcomes({
  v3Path,seed=2240001,pairs=4,ranks=4}={}){
  if(typeof v3Path!=='string'||!v3Path||!Number.isSafeInteger(seed)||seed<1||
     !Number.isSafeInteger(pairs)||pairs<1||pairs>32||
     !Number.isSafeInteger(ranks)||ranks<2||ranks>12||
     seed+pairs-1>0xffffffff)
    throw new TypeError('Invalid bounded outcome experiment parameters');
  const bytes=readFileSync(v3Path),source=verifiedRunningSourceSha(),aq=createOpponent();
  const digest=sha(bytes),rows=[];
  for(let i=0;i<pairs;i++)for(const seat of [0,1]){
    const scenarios=[];
    for(let rank=0;rank<ranks;rank++){
      const actor=createSelectionRankSession({
        model:new PersistentRNADInference(v3Path),seat,rank});
      const result=runStrategyMatch({
        seed:seed+i,policies:[0,1].map(s=>s===seat?'pure-rl-linear':'aq-base'),
        pureRlSessions:[0,1].map(s=>s===seat?actor:null),
        pureRlAllPhases:true,strategyMode:'integrated',
        v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq});
      const stat=result.pureRlStats?.[seat];
      if(!result.completed||result.score?.[result.winner]!==4||
         !stat?.first_selection||stat.pending||stat.updates!==0||
         stat.accepted_decisions!==result.actionCounts[seat]+2*result.rounds)
        throw new Error('Incomplete or untrusted four-win outcome Match');
      scenarios.push({rank,won:result.winner===seat,
        selected:stat.first_selection.selected,
        signature:stat.first_selection.signature,
        features:stat.first_selection.candidate_features,
        state_features:stat.first_selection.state_features,
        legal_count:stat.first_selection.selection_candidate_count,
        score:result.score});
    }
    if(new Set(scenarios.map(x=>x.signature)).size!==1||
       scenarios.some(x=>x.legal_count!==60||x.features.length!==48||
         x.state_features.length!==96))
      throw new Error('Intervention changes first-selection public state/support');
    rows.push({seed:seed+i,seat,scenarios});
  }
  const baselineWins=rows.filter(r=>r.scenarios[0].won).length;
  const alternatives=Array.from({length:ranks-1},(_,i)=>{
    const rank=i+1,wins=rows.filter(r=>r.scenarios[rank].won).length;
    const delta=(wins-baselineWins)/rows.length;
    // Matched SEED clusters; each seed contributes both seats.
    const half=Math.sqrt(2*Math.log(40*ranks)/pairs);
    return {rank,wins,paired_delta:delta,
      conservative_familywise_95pct_delta_bounds:[
        Math.max(-1,delta-half),Math.min(1,delta+half)],
      improvement_confirmed:delta-half>0};
  });
  return {schema:'rnad-i5-v3-first-selection-outcome-probe-v1',
    source_sha:source,engine_sha:provenance.sourceCommit,
    v3_model_sha256:digest,aq_base_model_sha256:AQ_SHA,
    seed,pairs,ranks,complete_matches:rows.length*ranks,
    baseline_wins:baselineWins,alternatives,
    rows,exploration_only:true,strength_established:false,
    champion_promoted:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const a=process.argv.slice(2),m=new Map();
  if(a.length!==10)throw new Error('Expected --v3 --seed --pairs --ranks --out');
  for(let i=0;i<a.length;i+=2){
    if(!['--v3','--seed','--pairs','--ranks','--out'].includes(a[i])||
       m.has(a[i])||!a[i+1])throw new Error('Invalid outcome CLI flags');
    m.set(a[i],a[i+1]);
  }
  const report=evaluateSelectionRankOutcomes({
    v3Path:m.get('--v3'),seed:Number(m.get('--seed')),
    pairs:Number(m.get('--pairs')),ranks:Number(m.get('--ranks'))});
  writeFileSync(m.get('--out'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({pairs:report.pairs,ranks:report.ranks,
    complete_matches:report.complete_matches,baseline_wins:report.baseline_wins,
    alternatives:report.alternatives}));
}
