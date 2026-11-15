// Opt-in RNAD-001 acceptance probe. This is NOT an R-NaD learner.
// Only this trusted runner owns collectors; user-supplied policies receive
// actor projections and legal candidates, never a commit/accepted callback.
import {runPhaseMatch} from './phase-match.js';
import {execFileSync} from 'node:child_process';
import {createBuiltinPhaseAgents} from './phase-agents.js';
import {rnadLegalActions,projectRnadObservation} from '../../../src/ml/legal-action-adapter.js';
import {createRnadTrajectoryCollector} from '../../../src/ml/trajectory-collector.js';
import battleData from '../../../data/catalog/battle-data.json' with {type:'json'};
import provenance from '../../../provenance/manifest.json' with {type:'json'};

const hexSha=x=>typeof x==='string'&&/^[0-9a-f]{40,64}$/i.test(x);
const positiveSeed=x=>Number.isSafeInteger(x)&&x>=1&&x<=0xffffffff;
export function verifiedRunningSourceSha(){
  const sha=execFileSync('git',['rev-parse','HEAD'],{
    encoding:'utf8',timeout:5000}).trim();
  if(!/^[a-f0-9]{40}$/.test(sha))throw new Error('Not running a verified Git checkout');
  return sha;
}

// Deterministic untrained contract policy; never use these records as
// a trained R-NaD policy or a performance claim. Selection uses normal Basic.
export function stage0FixedPolicy({observation,legalActions}){
  const active=observation.self.team?.[observation.self.active];
  const attacks=legalActions.filter(action=>action.kind==='move'&&
    Number.isInteger(action.index));
  // A deterministic nonlearning smoke policy: prefer a valid damaging move.
  // Only own moves and static public catalog are inspected.
  const power=action=>{
    const moveId=active?.moves?.[action.index]?.id;
    const def=battleData.moves?.[moveId];
    return def&&def.category!=='Status'?
      Math.max(0,def.basePower??0)*Math.max(.1,(def.accuracy??100)/100):0;
  };
  const attack=attacks.length?attacks.reduce((best,a)=>
    power(a)>power(best)?a:best,attacks[0]):
    legalActions.find(action=>action.kind==='move');
  return {action:attack??legalActions[0],
    behaviorLogProbability:0,
    referenceLogProbability:-Math.log(legalActions.length),
    predictedValue:0};
}

// This function is the trustworthy integration boundary, not a generic
// trajectoryCollector.commit() call supplied by the Actor.
// Existing runPhaseMatch submits Match.confirmTeam/confirmSelection/battleAction
// BEFORE invoking the agent's accepted callback.
export function runRnadAcceptanceSmoke({
  seed=22003,matchId='stage0-smoke',experimentId='rnad-stage0',
  sourceSha=verifiedRunningSourceSha(),engineSnapshotSha,policy=stage0FixedPolicy,
  phasePolicy=null,phasePolicies=null,
  policyVersion='rnad-stage0-fixed-untrained',opponentPolicyVersion=policyVersion,
  policyVersions=[policyVersion,policyVersion],
  maxRounds=64,maxDecisionsPerRound=3000
}={}){
  if(!positiveSeed(seed)||!hexSha(sourceSha)||!hexSha(engineSnapshotSha)||
    typeof matchId!=='string'||!matchId||typeof experimentId!=='string'||
    !experimentId||typeof policy!=='function'||
    (phasePolicy!==null&&typeof phasePolicy!=='function')||
    (phasePolicies!==null&&
      (phasePolicy!==null||!Array.isArray(phasePolicies)||
       phasePolicies.length!==2||
       phasePolicies.some(x=>x!==null&&typeof x!=='function')))||
    engineSnapshotSha!==provenance.sourceCommit||sourceSha!==verifiedRunningSourceSha()||
    typeof policyVersion!=='string'||!policyVersion||
    typeof opponentPolicyVersion!=='string'||!opponentPolicyVersion||
    !Array.isArray(policyVersions)||policyVersions.length!==2||
    policyVersions.some(x=>typeof x!=='string'||!x))
    throw new TypeError('Invalid RNAD smoke provenance or policy');
  const collectors=[0,1].map(seat=>createRnadTrajectoryCollector({
    experimentId,matchId,seed,seat,
    policyVersion:policyVersions[seat],opponentPolicyVersion:policyVersions[1-seat],
    sourceSha,engineSnapshotSha
  }));
  const timeline=[]; // Actual acceptance order; never Actor-visible.
  const agents=[0,1].map(seat=>{
    const collector=collectors[seat];
    const seatPhasePolicy=phasePolicies?.[seat]??phasePolicy;
    // A single per-seat collector follows accepted draft -> selection ->
    // Battle in order.  The other seat's pending action is never visible.
    function actor(stage){
      return {
        id:seatPhasePolicy?'rnad-three-phase-'+stage:'rnad-stage0-actor',
        select({view,context,round,rng}){
          const observation=projectRnadObservation(view,{seat,context});
          const legalActions=rnadLegalActions(view);
          const choose=seatPhasePolicy??policy;
          const proposed=choose({observation,legalActions,round,seat,rng});
          if(!proposed||typeof proposed!=='object'||
            !Object.hasOwn(proposed,'action'))
            throw new TypeError('Policy must return an action and logged probabilities');
          collector.propose({
            phase:stage,round,observation,legalActions,
            action:proposed.action,
            behaviorLogProbability:proposed.behaviorLogProbability,
            referenceLogProbability:proposed.referenceLogProbability,
            predictedValue:proposed.predictedValue
          });
          return proposed.action;
        },
        accepted({action,authorityReceipt,round}){
          const index=collector.commit(action,authorityReceipt)-1;
          timeline.push({seat,index});
          // Optional trusted policy memory may update only after the
          // one-use authoritative Match receipt has been consumed.
          seatPhasePolicy?.accepted?.({seat,round,action});
        },
        ...(stage==='battle'?{
          finish({result,authorityReceipt}){
            if(!result.completed)return collector.stats();
            return collector.finish(result,authorityReceipt);
          }
        }:{})
      };
    }
    return createBuiltinPhaseAgents({
      draft:seatPhasePolicy?actor('draft'):'basic',
      selection:seatPhasePolicy?actor('selection'):'basic',
      battle:actor('battle')
    });
  });
  const result=runPhaseMatch({seed,agents,maxRounds,maxDecisionsPerRound});
  // Only completed runs produce complete trajectories. Aborted / rejected
  // proposals cannot yield a publishable learned dataset.
  const trajectories=result.completed?
    result.agentStats.map(agent=>agent.battle):null;
  if(trajectories){
    const counts=trajectories.map(t=>t.accepted_decisions);
    if(counts.some((n,seat)=>n!==result.actionCounts[seat]+
      ((phasePolicies?.[seat]??phasePolicy)?2*result.rounds:0))||
      timeline.length!==counts[0]+counts[1])
      throw new Error('Authority trajectory order/count mismatch');
  }
  return {...result,trajectories,
    jointTimeline:trajectories?timeline:null};
}
