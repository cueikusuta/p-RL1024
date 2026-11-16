// Exact AQ-Base-001 head-to-head, not AQ public-teacher imitation.
// Uses the ORIGINAL frozen AQ full-Match strategy runner for the opponent;
// inserts a seat-local R-NaD deployment via its existing accepted-action
// session adapter, without exposing opponent hidden sets or altering weights.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as AQ_BASE_SHA}
  from '../../models/aq-base.mjs';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {PersistentRNADInference,RNADMixedPolicy} from './infer.mjs';
import {RnadSwitchLoopGuard,RNAD_SWITCH_GUARD}
  from './switch-loop-guard.mjs';
import {createPublicAQTeacher} from './aq-teacher-policy.mjs';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA}
  from '../../src/ml/rnad-phase-features.js';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA_V2}
  from '../../src/ml/rnad-phase-features-v2.js';
import {RNAD_THREE_PHASE_FEATURE_SCHEMA_V3}
  from '../../src/ml/rnad-phase-features-v3.js';
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export const FULL_AQ_EVALUATION_SCHEMA='rnad-vs-original-aq-base-full-match-paired-v1';
const MODES=['neural-all','aq-structure-neural-battle'];
const frozenV2E=Object.freeze({blend:.28,exposure:.75,continuation:.4});
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');

export function createRnadAQMatchSession({modelPath,seat,phaseMode='neural-all',
  switchGuard=RNAD_SWITCH_GUARD}={}){
  if(typeof modelPath!=='string'||!modelPath||
     ![0,1].includes(seat)||!MODES.includes(phaseMode)||
     !['none',RNAD_SWITCH_GUARD].includes(switchGuard))
    throw new TypeError('Invalid bounded full-AQ R-NaD adapter');
  const model=new PersistentRNADInference(modelPath);
  if(![RNAD_THREE_PHASE_FEATURE_SCHEMA,RNAD_THREE_PHASE_FEATURE_SCHEMA_V2,
        RNAD_THREE_PHASE_FEATURE_SCHEMA_V3].includes(model.featureSchema))
    throw new Error('Full AQ comparison requires a three-phase model');
  const historyNeeded=model.featureSchema===RNAD_THREE_PHASE_FEATURE_SCHEMA_V3;
  const neural=new RNADMixedPolicy({model,
    magnet:new PersistentRNADInference(modelPath),epsilon:0,
    selectionMode:'greedy'});
  const guarded=switchGuard===RNAD_SWITCH_GUARD?
    new RnadSwitchLoopGuard(neural):null;
  const structural=phaseMode==='aq-structure-neural-battle'?
    createPublicAQTeacher():null;
  let round=null,pending=null,accepted=0;
  const phaseCounts={draft:0,selection:0,battle:0};
  const ownHistory={lastKind:'none',consecutiveSwitches:0,
    lastSwitchTarget:-1,lastActive:-1};
  function select(view){
    if(pending)throw new Error('Uncommitted R-NaD action across authority boundary');
    const obs=projectRnadObservation(view,{seat});
    if(obs.phase==='draft'){
      if(!Number.isSafeInteger(obs.round)||obs.round<1||obs.round>64||
         (round!==null&&obs.round!==round+1))
        throw new Error('Out-of-order full-AQ Match round');
      round=obs.round;
    }
    if(!Number.isSafeInteger(round)||round<1)
      throw new Error('Missing accepted current draft round');
    const legalActions=rnadLegalActions(view);
    if(historyNeeded)obs.acceptedHistory={...ownHistory};
    const delegated=structural&&obs.phase!=='battle';
    const choice=delegated?
      structural({observation:obs,legalActions,seat,round}):
      guarded?guarded.choose({observation:obs,legalActions,seat,round}):
      neural.choose({observation:obs,legalActions,seat,round});
    if(!choice||!legalActions.some(a=>same(a,choice.action)))
      throw new Error('Full AQ R-NaD proposed action outside authoritative legal set');
    pending={action:structuredClone(choice.action),phase:obs.phase,
      requestType:obs.requestType??null,active:obs.self?.active??-1,
      round,delegated};
    return choice.action;
  }
  function commit(){
    if(!pending)throw new Error('Missing accepted full-AQ Match action');
    const {action,phase,requestType,active,delegated,round:r}=pending;
    // Called solely AFTER runStrategyMatch.confirmTeam/confirmSelection/
    // battleAction accepted the exact action. Rejected proposals abort.
    if(delegated)structural.accepted({seat,round:r,phase,action});
    else guarded?.accepted({seat,round:r,phase,action});
    if(historyNeeded){
      if(phase==='battle'&&requestType==='turn'){
        ownHistory.lastActive=Number.isSafeInteger(active)?active:-1;
        if(action?.kind==='switch'){
          ownHistory.lastKind='switch';
          ownHistory.consecutiveSwitches=
            Math.min(3000,ownHistory.consecutiveSwitches+1);
          ownHistory.lastSwitchTarget=Number.isSafeInteger(action.index)?
            action.index:-1;
        }else{
          ownHistory.lastKind='move';
          ownHistory.consecutiveSwitches=0;
          ownHistory.lastSwitchTarget=-1;
        }
      }else{
        ownHistory.lastKind=phase==='draft'?'draft':
          phase==='selection'?'selection':'none';
        ownHistory.consecutiveSwitches=0;
        ownHistory.lastSwitchTarget=-1;
        ownHistory.lastActive=-1;
      }
    }
    pending=null;
    accepted++;
    phaseCounts[phase]++;
  }
  function stats(){
    return {schema:'rnad-full-aq-seat-local-accepted-session-v1',
      accepted_decisions:accepted,accepted_by_phase:{...phaseCounts},
      pending:pending!==null,updates:0,train:false,
      model_feature_schema:model.featureSchema,
      phase_mode:phaseMode,switch_guard:switchGuard};
  }
  function finish(reward){
    if(pending||![-1,1].includes(reward)||
       Object.values(phaseCounts).some(n=>n<1))
      throw new Error('Incomplete accepted full-AQ actor session');
    return {...stats(),terminal_reward:reward};
  }
  return {select,commit,stats,finish};
}

export function evaluateAgainstOriginalAQ({modelPath,seed=1970001,
  pairs=8,phaseMode='neural-all',switchGuard=RNAD_SWITCH_GUARD}={}){
  if(typeof modelPath!=='string'||!modelPath||
     !Number.isSafeInteger(seed)||seed<1||
     !Number.isSafeInteger(pairs)||pairs<1||pairs>100||
     seed+pairs-1>0xffffffff||
     !MODES.includes(phaseMode)||
     !['none',RNAD_SWITCH_GUARD].includes(switchGuard))
    throw new TypeError('Invalid bounded original-AQ paired evaluation');
  const modelSha256=sha(modelPath);
  // The exact frozen AQ weights SHA is independently enforced by
  // createOpponent(); this is NOT a report against a proxy AQ teacher.
  const aq=createOpponent();
  if(aq.modelSha256!==AQ_BASE_SHA)
    throw new Error('Unexpected original AQ benchmark identity');
  const rows=[];let wins=0;
  const source=verifiedRunningSourceSha();
  for(let i=0;i<pairs;i++){
    const matchSeed=seed+i;
    for(const candidateSeat of [0,1]){
      const actor=createRnadAQMatchSession({
        modelPath,seat:candidateSeat,phaseMode,switchGuard});
      const result=runStrategyMatch({
        seed:matchSeed,
        policies:[0,1].map(s=>s===candidateSeat?'pure-rl-linear':'aq-base'),
        pureRlSessions:[0,1].map(s=>s===candidateSeat?actor:null),
        pureRlAllPhases:true,
        strategyMode:'integrated',v2Weights:frozenV2E,
        aqBase:aq});
      if(!result.completed||![0,1].includes(result.winner)||
         !Array.isArray(result.score)||result.score.length!==2||
         result.score[result.winner]!==4||
         result.score[1-result.winner]>=4||
         result.pureRlStats?.[candidateSeat]?.pending||
         result.pureRlStats?.[candidateSeat]?.updates!==0||
         result.pureRlStats?.[candidateSeat]?.accepted_decisions!==
           result.actionCounts[candidateSeat]+2*result.rounds)
        throw new Error('Incomplete or non-authoritative original AQ 4-win Match');
      const won=result.winner===candidateSeat;
      wins+=Number(won);
      rows.push({seed:matchSeed,candidateSeat,won,
        score:result.score,rounds:result.rounds,
        accepted_decisions:result.pureRlStats[candidateSeat].accepted_decisions});
    }
  }
  const total=2*pairs,ci95=pairedHoeffding95(wins,pairs);
  return {schema:FULL_AQ_EVALUATION_SCHEMA,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    aq_base_model_sha256:AQ_BASE_SHA,
    model_sha256:modelSha256,
    source_sha:source,engine_sha:provenance.sourceCommit,
    phase_mode:phaseMode,switch_guard:switchGuard,
    evaluation_policy:'greedy-no-training',
    seed,pairs,matches:total,wins,
    win_rate:wins/total,
    paired_hoeffding_95pct:ci95,
    improvement_statistically_established:ci95[0]>.5,
    champion_promoted:false,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const argv=process.argv.slice(2);
  if(argv.length!==12)throw new Error('Expected six original AQ evaluation options');
  const keys=new Set(['--model','--seed','--pairs','--phase-mode',
    '--switch-guard','--out']),opt=new Map();
  for(let i=0;i<argv.length;i+=2){
    if(!keys.has(argv[i])||opt.has(argv[i])||!argv[i+1])
      throw new Error('Invalid or duplicate original AQ evaluation option');
    opt.set(argv[i],argv[i+1]);
  }
  if(opt.size!==6)throw new Error('Missing required AQ comparison option');
  const result=evaluateAgainstOriginalAQ({
    modelPath:opt.get('--model'),seed:Number(opt.get('--seed')),
    pairs:Number(opt.get('--pairs')),
    phaseMode:opt.get('--phase-mode'),
    switchGuard:opt.get('--switch-guard')});
  writeFileSync(opt.get('--out'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({matches:result.matches,wins:result.wins,
    win_rate:result.win_rate,ci95:result.paired_hoeffding_95pct,
    mode:result.phase_mode,significant:result.improvement_statistically_established}));
}
