// RNAD-001 trusted two-seat full-Match rollout exporter. Only Match-issued
// accepted receipts can enter the training file; Actor receives no opponent
// hidden state and never sees the opponent's uncommitted simultaneous action.
import {writeFileSync,readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {runRnadAcceptanceSmoke,verifiedRunningSourceSha}
  from '../dev/simulator/rnad-match.js';
import {PersistentRNADInference,RNADMixedPolicy} from './infer.mjs';
import {RnadSwitchLoopGuard,RNAD_SWITCH_GUARD} from './switch-loop-guard.mjs';
import {shapeSwitchOpportunity,SWITCH_OPPORTUNITY_OBJECTIVE}
  from './switch-opportunity.mjs';
import {encodeRnadDecision,RNAD_FEATURE_SCHEMA} from '../../src/ml/rnad-features.js';
import {encodeRnadThreePhaseDecision,RNAD_THREE_PHASE_FEATURE_SCHEMA}
  from '../../src/ml/rnad-phase-features.js';
import {encodeRnadThreePhaseDecisionV2,RNAD_THREE_PHASE_FEATURE_SCHEMA_V2}
  from '../../src/ml/rnad-phase-features-v2.js';
import {encodeRnadThreePhaseDecisionV3,RNAD_THREE_PHASE_FEATURE_SCHEMA_V3}
  from '../../src/ml/rnad-phase-features-v3.js';
import {encodeRnadThreePhaseDecisionV4,RNAD_THREE_PHASE_FEATURE_SCHEMA_V4}
  from '../../src/ml/rnad-phase-features-v4.js';
import provenance from '../../provenance/manifest.json' with {type:'json'};

function argsMap(argv){
  const r={};
  for(let i=0;i<argv.length;i+=2){
    const k=argv[i],v=argv[i+1];
    if(!k?.startsWith('--')||v===undefined)throw new Error('Invalid RNAD CLI argument');
    r[k.slice(2)]=v;
  }
  return r;
}
export function createLearningTrajectory({seed,modelPath,magnetPath,
  matchId='rnad-match-'+seed,epsilon=.02,threePhase=false,
  featuresV2=false,featuresV3=false,featuresV4=false,
  switchGuard='none',switchCost='none',
  opponentModelPath=null,learnerSeat=null}={}){
  if(!Number.isSafeInteger(seed)||seed<1||seed>0xffffffff||
     !['none',RNAD_SWITCH_GUARD].includes(switchGuard)||
     !['none',SWITCH_OPPORTUNITY_OBJECTIVE].includes(switchCost)||
     (featuresV4&&(featuresV2||featuresV3||!threePhase))||
     (featuresV3&&(featuresV2||!threePhase))||
     (switchCost!== 'none'&&
       (switchGuard!==RNAD_SWITCH_GUARD||!threePhase||
        !(featuresV2||featuresV3||featuresV4))))
    throw new TypeError('Invalid bounded RNAD seed or guarded policy');
  const league=opponentModelPath!==null;
  if(league&&(!featuresV3||switchGuard!==RNAD_SWITCH_GUARD||
     ![0,1].includes(learnerSeat)||
     typeof opponentModelPath!=='string'||!opponentModelPath))
    throw new TypeError('League opponent requires v3 and seat-bound guard');
  if(!league&&learnerSeat!==null)
    throw new TypeError('League learner seat without frozen opponent');
  const opponentSha=league?createHash('sha256').update(
    readFileSync(opponentModelPath)).digest('hex'):null;
  const neural=new RNADMixedPolicy({
    model:new PersistentRNADInference(modelPath),
    magnet:new PersistentRNADInference(magnetPath),epsilon
  });
  const featureSchema=featuresV4?RNAD_THREE_PHASE_FEATURE_SCHEMA_V4:
    featuresV3?RNAD_THREE_PHASE_FEATURE_SCHEMA_V3:
    featuresV2?RNAD_THREE_PHASE_FEATURE_SCHEMA_V2:
    threePhase?RNAD_THREE_PHASE_FEATURE_SCHEMA:RNAD_FEATURE_SCHEMA;
  if(typeof threePhase!=='boolean'||typeof featuresV2!=='boolean'||
    typeof featuresV3!=='boolean'||typeof featuresV4!=='boolean'||
    ((featuresV2||featuresV3||featuresV4)&&!threePhase)||
    neural.model.featureSchema!==featureSchema)
    throw new TypeError('RNAD model and rollout feature schemas disagree');
  if(switchGuard===RNAD_SWITCH_GUARD&&!threePhase)
    throw new TypeError('Anti-loop learning requires true three-phase actor');
  const guarded=switchGuard===RNAD_SWITCH_GUARD;
  const opportunity=switchCost===SWITCH_OPPORTUNITY_OBJECTIVE;
  const objectiveSuffix=(opportunity?'-'+SWITCH_OPPORTUNITY_OBJECTIVE:'')+
    (league?'-league-v1':'');
  const guardedSeatPolicies=guarded?[0,1].map(seat=>{
    const useOpponent=league&&seat!==learnerSeat;
    const actorPath=useOpponent?opponentModelPath:modelPath;
    const referencePath=useOpponent?opponentModelPath:magnetPath;
    const guard=new RnadSwitchLoopGuard(new RNADMixedPolicy({
      model:new PersistentRNADInference(actorPath),
      magnet:new PersistentRNADInference(referencePath),epsilon
    }));
    const policy=ctx=>guard.choose(ctx);
    policy.accepted=ctx=>guard.accepted(ctx);
    return policy;
  }):null;
  const result=runRnadAcceptanceSmoke({
    seed,matchId,experimentId:(featuresV4?'RNAD-005-3phase-v4':
      featuresV3?'RNAD-004-3phase-v3':
      featuresV2?'RNAD-003-3phase-v2':
      threePhase?'RNAD-002-3phase':'RNAD-001')+
      (guarded?'-'+RNAD_SWITCH_GUARD:'')+objectiveSuffix,
    sourceSha:verifiedRunningSourceSha(),
    engineSnapshotSha:provenance.sourceCommit,
    opponentPolicyVersion:(featuresV4?'RNAD-005-three-phase-v4':
      featuresV3?'RNAD-004-three-phase-v3':
      featuresV2?'RNAD-003-three-phase-v2':
      threePhase?'RNAD-002-three-phase':'RNAD-001-neural-learner')+
      (guarded?'-'+RNAD_SWITCH_GUARD:'')+objectiveSuffix,
    policyVersion:(featuresV4?'RNAD-005-three-phase-v4':
      featuresV3?'RNAD-004-three-phase-v3':
      featuresV2?'RNAD-003-three-phase-v2':
      threePhase?'RNAD-002-three-phase':'RNAD-001-neural-learner')+
      (guarded?'-'+RNAD_SWITCH_GUARD:'')+objectiveSuffix,
    includeAcceptedHistory:featuresV3||featuresV4,
    ...(league?{policyVersions:[0,1].map(seat=>
      (featuresV3?'RNAD-004-three-phase-v3':'RNAD-003-three-phase-v2')+
      '-'+RNAD_SWITCH_GUARD+(seat===learnerSeat?
        '-learner-league-v1':'-frozen-league-v1-'+opponentSha.slice(0,8)))}:{}),
    phasePolicies:guarded?guardedSeatPolicies:null,
    phasePolicy:!guarded&&threePhase?(args=>neural.choose(args)):null,
    policy:guarded?(ctx=>guardedSeatPolicies[ctx.seat](ctx)):
      (args=>neural.choose(args))
  });
  if(!result.completed||!result.jointTimeline||
    result.jointTimeline.length===0||result.trajectories.length!==2)
    throw new Error('Only completed accepted Matches may enter learning');
  const acceptedRows=result.jointTimeline.map(({seat,index})=>
    result.trajectories[seat].decisions[index]);
  const events=result.jointTimeline.map(({seat,index})=>{
    const row=result.trajectories[seat].decisions[index];
    if(row.player_seat!==seat||
      !['draft','selection','battle'].includes(row.phase)||
      (!threePhase&&row.phase!=='battle'))
      throw new Error('Invalid authority timeline/seat boundary');
    // Policy support was narrowed only after legal authority supplied ALL
    // candidates. The old full set is retained on the receipt-backed record;
    // training softmax/magnet must use exactly the support seen by the Actor.
    if(guarded&&row.phase==='battle'&&!row.policy_eligible_actions)
      throw new Error('Missing guarded Actor support in accepted battle');
    if(!guarded&&row.policy_eligible_actions)
      throw new Error('Unexpected guarded Actor support in unguarded learning');
    const support=row.policy_eligible_actions??row.legal_actions;
    const encoded=featuresV4?
      encodeRnadThreePhaseDecisionV4(row.observation,support):
      featuresV3?
      encodeRnadThreePhaseDecisionV3(row.observation,support):
      featuresV2?
      encodeRnadThreePhaseDecisionV2(row.observation,support):
      threePhase?encodeRnadThreePhaseDecision(row.observation,support):
      encodeRnadDecision(row.observation,support);
    const actionIndex=support.findIndex(a=>
      JSON.stringify(a)===JSON.stringify(row.accepted_action));
    if(actionIndex<0)throw new Error('Missing accepted legal action');
    return {actor:seat,phase:row.phase,round:row.round,
      turn:row.observation.turn??null,
      state:encoded.state,actions:encoded.actions,action_index:actionIndex,
      ...(guarded?{
        authoritative_legal_action_count:row.legal_actions.length,
        policy_eligible_action_count:support.length
      }:{}),
      behavior_prob:Math.exp(row.behavior_log_probability),
      magnet_log_prob:row.reference_log_probability,
      reward:[0,0]};
  });
  if(threePhase){
    const counts={draft:0,selection:0,battle:0};
    for(const e of events)counts[e.phase]++;
    if(counts.draft!==2*result.rounds||
       counts.selection!==2*result.rounds||
       counts.battle!==result.actionCounts[0]+result.actionCounts[1])
      throw new Error('Incomplete three-phase accepted Match timeline');
  }
  const shaping=opportunity?
    shapeSwitchOpportunity(events,acceptedRows):null;
  const payoff0=result.winner===0?1:-1;
  events.at(-1).reward[0]+=payoff0;
  events.at(-1).reward[1]-=payoff0;
  return {schema:featuresV4?
      'rnad-accepted-three-phase-v4-training-match-v1':
      featuresV3?
      'rnad-accepted-three-phase-v3-training-match-v1':
      opportunity?
      'rnad-accepted-three-phase-v2-opportunity-training-match-v1':
      guarded?'rnad-accepted-three-phase-v2-anti-loop-training-match-v1':
      featuresV2?'rnad-accepted-three-phase-v2-training-match-v1':
      threePhase?'rnad-accepted-three-phase-training-match-v1':
      'rnad-accepted-training-match-v1',
    feature_schema:featureSchema,
    ...(league?{league:{learner_seat:learnerSeat,
      frozen_opponent_sha256:opponentSha,
      frozen_opponent_feature_schema:featureSchema}}:{}),
    ...(guarded?{deployment_policy:{switch_guard:RNAD_SWITCH_GUARD}}:{}),
    ...(opportunity?{reward_objective:SWITCH_OPPORTUNITY_OBJECTIVE,
       reward_shaping:shaping}:{}),
    match_id:matchId,
    source_sha:verifiedRunningSourceSha(),
    engine_sha:provenance.sourceCommit,seed,completed:true,
    winner:result.winner,score:result.score,rounds:result.rounds,
    action_counts:result.actionCounts,events};
}
if(process.argv[1]?.endsWith('/export-trajectory.mjs')){
  const a=argsMap(process.argv.slice(2));
  if(!a.model||!a.magnet||!a.out||!a.seed)
    throw new Error('Usage: --model MODEL.json --magnet REF.json --seed N --out FILE');
  const seed=Number(a.seed);
  if((a['three-phase']!==undefined&&a['three-phase']!=='true')||
     (a['features-v2']!==undefined&&a['features-v2']!=='true')||
     (a['features-v3']!==undefined&&a['features-v3']!=='true')||
     (a['features-v4']!==undefined&&a['features-v4']!=='true')||
     (a['switch-guard']!==undefined&&a['switch-guard']!==RNAD_SWITCH_GUARD)||
     (a['switch-cost']!==undefined&&
      a['switch-cost']!==SWITCH_OPPORTUNITY_OBJECTIVE)||
     (a['opponent-model']!==undefined&&
      (typeof a['opponent-model']!=='string'||!a['opponent-model']))||
     (a['learner-seat']!==undefined&&!['0','1'].includes(a['learner-seat'])))
    throw new TypeError('Invalid three-phase/v2/guard flag');
  const match=createLearningTrajectory({seed,modelPath:a.model,magnetPath:a.magnet,
    threePhase:a['three-phase']==='true',
    featuresV2:a['features-v2']==='true',
    featuresV3:a['features-v3']==='true',
    featuresV4:a['features-v4']==='true',
    switchGuard:a['switch-guard']??'none',
    switchCost:a['switch-cost']??'none',
    opponentModelPath:a['opponent-model']??null,
    learnerSeat:a['learner-seat']===undefined?null:Number(a['learner-seat'])});
  writeFileSync(a.out,JSON.stringify(match)+'\n',{flag:'wx'});
  process.stderr.write(JSON.stringify({seed,winner:match.winner,score:match.score,
    rounds:match.rounds,events:match.events.length})+'\n');
}
