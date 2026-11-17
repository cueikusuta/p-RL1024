// Opt-in two-model actor: independently trained v4 phase policy + frozen v3-U4 battle.
// Both GRU memories see the SAME accepted seat-local public observations,
// and neither model ever observes the original AQ opponent's hidden sets.
// Exact AQ-Base-001 is provided by the unmodified strategy Match runner.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runStrategyMatch} from '../../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as AQ_BASE_SHA}
  from '../../models/aq-base.mjs';
import {projectRnadObservation,rnadLegalActions}
  from '../../src/ml/legal-action-adapter.js';
import {PersistentRNADInference,softmaxRnad}
  from './infer.mjs';
import {RnadSwitchLoopGuard} from './switch-loop-guard.mjs';
import {pairedHoeffding95} from './aggregate-eval.mjs';
import {verifiedRunningSourceSha} from '../dev/simulator/rnad-match.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

export const COMPOSITE_SCHEMA='rnad-v4-v3-composite-vs-original-aq-paired-v1';
export const COMPOSITE_MODES=Object.freeze({
  'all-v3':{draft:1,selection:1,battle:1},
  'all-v4':{draft:0,selection:0,battle:0},
  'draft-v4':{draft:0,selection:1,battle:1},
  'selection-v4':{draft:1,selection:0,battle:1},
  'selection-quarter-v4':{draft:1,selection:.75,battle:1},
  'selection-half-v4':{draft:1,selection:.5,battle:1},
  'selection-threequarter-v4':{draft:1,selection:.25,battle:1},
  'both-v4':{draft:0,selection:0,battle:1}
});
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
export function createV4CompositeAQSession({v4Path,v3Path,seat,mode,
  expectedPhaseSchema='rnad-three-phase-features-v4',draftPolicy=null}={}){
  if(typeof v4Path!=='string'||!v4Path||
     typeof v3Path!=='string'||!v3Path||
     ![0,1].includes(seat)||!Object.hasOwn(COMPOSITE_MODES,mode)||
     (draftPolicy!==null&&(mode!=='all-v3'||
       draftPolicy?.schema!=='rnad-draft-deepsets-inference-v1'||
       typeof draftPolicy.choose!=='function')))
    throw new TypeError('Invalid two-model AQ comparison actor');
  const v4=new PersistentRNADInference(v4Path);
  const v3=new PersistentRNADInference(v3Path);
  if(!['rnad-three-phase-features-v4',
        'rnad-three-phase-features-v5'].includes(expectedPhaseSchema)||
     v4.featureSchema!==expectedPhaseSchema||
     v3.featureSchema!=='rnad-three-phase-features-v3')
    throw new Error('Mixed or unsupported v4/v3 model observation schemas');
  let round=0,pending=null,accepted=0;
  const counts={draft:0,selection:0,battle:0};
  const history={lastKind:'none',consecutiveSwitches:0,
    lastSwitchTarget:-1,lastActive:-1};
  const combined={
    choose({observation,legalActions,seat:s}){
      // Feed both GRUs on every accepted decision, even when a phase uses
      // only one model; otherwise the inactive model's history is stale.
      const a=v4.inferProjected(observation,legalActions,s);
      const b=v3.inferProjected(observation,legalActions,s);
      const pa=softmaxRnad(a.logits),pb=softmaxRnad(b.logits);
      const alpha=COMPOSITE_MODES[mode][observation.phase];
      if(!Number.isFinite(alpha)||pa.length!==pb.length||
         pa.length!==legalActions.length)
        throw new Error('Inconsistent source-bounded composite policy support');
      // Optional research-only draft replacement: both ordinary GRU memories
      // still receive the same authority-accepted seat-local decisions.
      // All other phases remain frozen v3 and the switch guard is unchanged.
      if(draftPolicy!==null&&observation.phase==='draft'){
        const proposal=draftPolicy.choose(observation,legalActions);
        if(!Number.isSafeInteger(proposal?.index)||
           proposal.index<0||proposal.index>=legalActions.length||
           !Number.isFinite(proposal.logProbability))
          throw new Error('Invalid Deep Sets draft support/index');
        return {action:legalActions[proposal.index],
          behaviorLogProbability:0,
          referenceLogProbability:proposal.logProbability,
          predictedValue:b.value};
      }
      const scores=pa.map((x,i)=>(1-alpha)*x+alpha*pb[i]);
      let best=0;
      for(let i=1;i<scores.length;i++)
        if(scores[i]>scores[best])best=i;
      return {action:legalActions[best],
        behaviorLogProbability:0,referenceLogProbability:Math.log(scores[best]),
        predictedValue:(1-alpha)*a.value+alpha*b.value};
    }
  };
  const guard=new RnadSwitchLoopGuard(combined);
  function select(view){
    if(pending)throw new Error('Unaccepted composite Match decision');
    const obs=projectRnadObservation(view,{seat});
    if(obs.phase==='draft'){
      if(!Number.isSafeInteger(obs.round)||obs.round!==round+1)
        throw new Error('Out-of-order composite draft round');
      round=obs.round;
    }
    if(!round)throw new Error('Composite actor missing accepted draft round');
    obs.acceptedHistory={...history};
    const legal=rnadLegalActions(view);
    const chosen=guard.choose({observation:obs,legalActions:legal,
      seat,round}).action;
    if(!legal.some(a=>same(a,chosen)))
      throw new Error('Composite action omitted from authority support');
    pending={action:structuredClone(chosen),phase:obs.phase,
      requestType:obs.requestType??null,
      active:obs.self?.active??-1,round};
    return chosen;
  }
  function commit(){
    if(!pending)throw new Error('Missing accepted composite decision');
    const {phase,requestType,active,action,round:r}=pending;
    guard.accepted({seat,round:r,phase,action});
    // Only the authority acceptance callback can advance own history.
    if(phase==='battle'&&requestType==='turn'){
      history.lastActive=Number.isSafeInteger(active)?active:-1;
      if(action?.kind==='switch'){
        history.lastKind='switch';
        history.consecutiveSwitches=Math.min(3000,history.consecutiveSwitches+1);
        history.lastSwitchTarget=Number.isSafeInteger(action.index)?
          action.index:-1;
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
    pending=null;accepted++;counts[phase]++;
  }
  function stats(){
    return {schema:'rnad-v4-v3-composite-authority-session-v1',
      accepted_decisions:accepted,accepted_by_phase:{...counts},
      pending:pending!==null,updates:0,train:false,mode,
      ...(draftPolicy?{draft_model_schema:draftPolicy.schema,
        draft_model_sha256:draftPolicy.modelSha256}: {})};
  }
  function finish(reward){
    if(pending||![-1,1].includes(reward)||
       Object.values(counts).some(x=>x<1))
      throw new Error('Incomplete original AQ composite Match');
    return {...stats(),terminal_reward:reward};
  }
  return {select,commit,stats,finish};
}
export function evaluateV4CompositeAgainstOriginalAQ({v4Path,v3Path,
  seed=1980001,pairs=24,mode='all-v3'}={}){
  if(typeof v4Path!=='string'||!v4Path||
     typeof v3Path!=='string'||!v3Path||
     !Number.isSafeInteger(seed)||seed<1||
     !Number.isSafeInteger(pairs)||pairs<1||pairs>100||
     seed+pairs-1>0xffffffff||
     !Object.hasOwn(COMPOSITE_MODES,mode))
    throw new TypeError('Invalid bounded composite vs original AQ evaluation');
  const shaV4=sha(v4Path),shaV3=sha(v3Path);
  const aq=createOpponent(),source=verifiedRunningSourceSha();
  const rows=[];let wins=0;
  for(let n=0;n<pairs;n++)for(const candidateSeat of [0,1]){
    const matchSeed=seed+n;
    const actor=createV4CompositeAQSession({v4Path,v3Path,seat:candidateSeat,mode});
    const result=runStrategyMatch({
      seed:matchSeed,
      policies:[0,1].map(s=>s===candidateSeat?'pure-rl-linear':'aq-base'),
      pureRlSessions:[0,1].map(s=>s===candidateSeat?actor:null),
      pureRlAllPhases:true,strategyMode:'integrated',
      v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:aq});
    if(!result.completed||![0,1].includes(result.winner)||
       result.score[result.winner]!==4||
       result.pureRlStats?.[candidateSeat]?.pending||
       result.pureRlStats?.[candidateSeat]?.updates!==0||
       result.pureRlStats?.[candidateSeat]?.accepted_decisions!==
          result.actionCounts[candidateSeat]+2*result.rounds)
      throw new Error('Invalid composite authority-accepted four-win Match');
    const won=result.winner===candidateSeat;
    wins+=Number(won);
    rows.push({seed:matchSeed,candidateSeat,won,
      score:result.score,rounds:result.rounds,
      accepted_decisions:result.pureRlStats[candidateSeat].accepted_decisions});
  }
  const ci95=pairedHoeffding95(wins,pairs);
  return {schema:COMPOSITE_SCHEMA,source_sha:source,
    engine_sha:provenance.sourceCommit,
    benchmark:'AQ-Base-001 exact integrated strategy-match',
    aq_base_model_sha256:AQ_BASE_SHA,
    v4_model_sha256:shaV4,v3_model_sha256:shaV3,
    mode,policy:'frozen-greedy-v4-v3-anti-loop-v1',
    seed,pairs,matches:2*pairs,wins,win_rate:wins/(2*pairs),
    paired_hoeffding_95pct:ci95,
    statistically_beats_aq:ci95[0]>.5,
    champion_promoted:false,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const argv=process.argv.slice(2),opts=new Map(),
    allowed=new Set(['--v4','--v3','--seed','--pairs','--mode','--out']);
  if(argv.length!==12)throw new Error('Expected six composite AQ argument pairs');
  for(let i=0;i<argv.length;i+=2){
    if(!allowed.has(argv[i])||opts.has(argv[i])||!argv[i+1])
      throw new Error('Invalid or duplicate composite AQ argument');
    opts.set(argv[i],argv[i+1]);
  }
  if(opts.size!==6)throw new Error('Incomplete composite AQ model arguments');
  const r=evaluateV4CompositeAgainstOriginalAQ({
    v4Path:opts.get('--v4'),v3Path:opts.get('--v3'),
    seed:Number(opts.get('--seed')),pairs:Number(opts.get('--pairs')),
    mode:opts.get('--mode')});
  writeFileSync(opts.get('--out'),JSON.stringify(r,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({mode:r.mode,matches:r.matches,wins:r.wins,
    win_rate:r.win_rate,ci95:r.paired_hoeffding_95pct}));
}
