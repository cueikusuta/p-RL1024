// RNAD-001 Stage 0. Never record a proposed action as accepted.
// Call commit() only *after* authoritative Match/Battle acceptance.
import {RNAD_OBSERVATION_SCHEMA,validateRnadLegalActions}
  from './legal-action-adapter.js';
import {consumePhaseAuthorityReceipt} from '../../scripts/dev/simulator/phase-match.js';

export const RNAD_TRAJECTORY_SCHEMA='rnad-trajectory-v1';
const phases=new Set(['draft','selection','battle']);
const isObject=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
const clone=x=>structuredClone(x);
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const positiveInt=n=>Number.isSafeInteger(n)&&n>0;
const key=value=>JSON.stringify(value);
const label=x=>typeof x==='string'&&x.length>0&&x.length<=160;
export function createRnadTrajectoryCollector({
  experimentId,matchId,seed,seat,policyVersion,opponentPolicyVersion,
  sourceSha,engineSnapshotSha
}={}){
  if(![0,1].includes(seat)||!positiveInt(seed)||seed>0xffffffff||
    ![experimentId,matchId,policyVersion,opponentPolicyVersion,
      sourceSha,engineSnapshotSha].every(label))
    throw new TypeError('Invalid trajectory provenance');
  const meta={experiment_id:experimentId,match_id:matchId,seed,
    player_seat:seat,policy_version:policyVersion,
    opponent_policy_version:opponentPolicyVersion,
    source_sha:sourceSha,engine_snapshot_sha:engineSnapshotSha};
  const records=[];
  let pending=null,finished=false,lastRound=0;
  function propose({phase,round,observation,legalActions,action,
    behaviorLogProbability,referenceLogProbability,predictedValue}={}){
    if(finished||pending||!phases.has(phase)||!positiveInt(round)||
      round>64||round<lastRound||!isObject(observation)||
      observation.schema!==RNAD_OBSERVATION_SCHEMA||
      observation.seat!==seat||observation.phase!==phase)
      throw new TypeError('Invalid proposed RNAD decision');
    const legal=validateRnadLegalActions(legalActions);
    const target=key(action);
    if(!legal.some(candidate=>key(candidate)===target))
      throw new TypeError('Proposed action not in the exact legal set');
    if(![behaviorLogProbability,referenceLogProbability,predictedValue]
      .every(finite)||behaviorLogProbability>0||referenceLogProbability>0||
      Math.abs(predictedValue)>1e6)
      throw new TypeError('Invalid behavior/reference/value');
    pending={...meta,round,phase,observation_schema_version:RNAD_OBSERVATION_SCHEMA,
      observation:clone(observation),legal_actions:legal,
      accepted_action:clone(action),
      behavior_log_probability:behaviorLogProbability,
      reference_log_probability:referenceLogProbability,
      predicted_value:predictedValue};
  }
  function commit(acceptedAction,authorityReceipt){
    if(finished||!pending||key(acceptedAction)!==key(pending.accepted_action))
      throw new Error('No matching authoritative accepted action');
    consumePhaseAuthorityReceipt(authorityReceipt,{kind:pending.phase,seat,
      round:pending.round,action:acceptedAction});
    records.push(pending);
    lastRound=pending.round;
    pending=null;
    return records.length;
  }
  function reject(){
    if(finished||!pending)throw new Error('No pending RNAD action');
    pending=null;
  }
  function finish({completed,winner,score}={},authorityReceipt){
    if(finished||pending||completed!==true||![0,1].includes(winner)||
      !Array.isArray(score)||score.length!==2||score[winner]!==4||!score.every(x=>Number.isInteger(x)&&x>=0&&x<=4)||score[1-winner]>=4)
      throw new Error('Expected completed authoritative four-win Match');
    consumePhaseAuthorityReceipt(authorityReceipt,{kind:'finish',seat,
      round:Math.max(...records.map(r=>r.round),0),action:{winner,score}});
    finished=true;
    const terminal=winner===seat?1:-1;
    return {schema:RNAD_TRAJECTORY_SCHEMA,...meta,completed:true,
      winner,score:clone(score),accepted_decisions:records.length,
      decisions:records.map(record=>({...clone(record),terminal_reward:terminal}))};
  }
  function stats(){return {seat,accepted_decisions:records.length,
    pending:pending!==null,finished};}
  return {propose,commit,reject,finish,stats};
}
