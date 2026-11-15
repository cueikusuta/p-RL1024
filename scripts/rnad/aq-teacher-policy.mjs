// Opt-in AQ-Base-001 imitation teacher. Uses frozen, SHA-validated AQ weights
// and the pinned public V2E ranking with public belief and tactical context.
// Not an exact reproduction of full strategy-match AQ: its score/turn history
// is intentionally not sent through the RNAD seat projection.
import {chooseStrategyTeam} from '../../src/ai/team-assessment.js';
import {chooseTacticalSelection} from '../../src/ai/selection-tactical.js';
import {rankStrategyV2Actions,commitStrategyV2Choice}
  from '../../aq-runtime/src/ai/strategy-v2.js';
import {assessCompactMatchup,assessMatchup}
  from '../../src/ai/matchup-assessment.js';
import {createI877AntiStall}
  from '../../aq-runtime/src/ai/v3-research-anti-stall.js';
import {createModel as createAQ,weightsSha256 as AQ_WEIGHTS_SHA256}
  from '../../models/aq-base.mjs';
import {chooseFrozenAQBase}
  from '../../aq-runtime/scripts/dev/simulator/rl-linear-aq-base-opponent.js';
import {playableSets} from '../../src/production-runtime.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};
import {RNAD_OBSERVATION_SCHEMA} from '../../src/ml/legal-action-adapter.js';

export const AQ_TEACHER_ID='rnad-aq-public-belief-v2';
export {AQ_WEIGHTS_SHA256};
const sign=x=>JSON.stringify(x);
const v2Weights=Object.freeze({blend:.28,exposure:.75,continuation:.4});
export function createPublicAQTeacher(){
  const model=createAQ();
  const seats=[0,1].map(()=>({
    round:0,belief:null,memory:{},selected:[],guard:createI877AntiStall({maxNonAttacks:5}),
    pending:null
  }));
  function choose({observation,legalActions,seat=observation?.seat,round=1}){
    if(observation?.schema!==RNAD_OBSERVATION_SCHEMA||
      ![0,1].includes(seat)||seat!==observation.seat||
      !Number.isInteger(round)||round<1||round>64||
      !['draft','selection','battle'].includes(observation.phase)||
      !Array.isArray(legalActions)||!legalActions.length)
      throw new TypeError('AQ teacher requires seat-local projected observation and legal actions');
    const own=seats[seat];
    if(round!==own.round){
      if(own.pending)throw new Error('Unaccepted AQ plan crossed round boundary');
      own.round=round;own.belief=null;own.memory={};own.selected=[];
      own.guard=createI877AntiStall({maxNonAttacks:5});
    }
    if(own.pending)throw new Error('Unaccepted AQ plan at next decision');
    let proposed;
    if(observation.phase==='draft'){
      proposed=chooseStrategyTeam(observation,{data,fastDraftValidation:true});
    }else if(observation.phase==='selection'){
      proposed=chooseTacticalSelection(observation,{data,sets:playableSets});
      own.pending={round,phase:'selection',action:proposed,
        roster:observation.self.team};
    }else{
      const active=observation.self?.team?.[observation.self.active];
      if(!active)throw new TypeError('No own active Pokémon in AQ public observation');
      const ownSet=own.selected[observation.self.active];
      if(!ownSet?.setId)throw new TypeError('Missing accepted own selected set in AQ public teacher');
      const evaluated=assessCompactMatchup(observation,{
        ownSet,catalog:playableSets,data,previousBelief:own.belief});
      own.belief=evaluated.beliefState;
      const ownMatchups=observation.self.team.map((member,index)=>{
        if(!(member.hp>0))return null;
        if(index===observation.self.active)return evaluated.assessment;
        return assessMatchup({ownSet:own.selected[index],own:member,
          opponentBelief:evaluated.beliefState.belief,
          opponentPublic:observation.opponent,
          catalog:playableSets,data,field:observation.field});
      });
      const options={
        data,memory:own.memory,matchup:evaluated.assessment,
        ownMatchups,switchMatchups:ownMatchups,roundWins:0};
      // Original AQ-Base-001 obtains its 61D correction only on full turns.
      // Forced replacements/pivots use frozen V2E, as in original runner.
      const ranking=rankStrategyV2Actions(observation,{
        ...options,variant:'strategy-v2-e',catalog:playableSets,weights:v2Weights});
      const sorted=ranking.rows.map(row=>row.action);
      if(sorted.length<1)throw new Error('Empty public V2E AQ ranking');
      if(observation.requestType==='turn'){
        const baselineScores=ranking.rows.map(row=>row.score);
        const ranked=chooseFrozenAQBase(model,observation,sorted,baselineScores,{
          data,roundWins:0,roundLosses:0});
        proposed=sorted[own.guard.choose(
          ranked.candidates,ranked.selectedIndex).index];
      }else proposed=sorted[0];
      own.pending={round,phase:'battle',view:observation,action:proposed,
        options,ranking};
    }
    const idx=legalActions.findIndex(candidate=>sign(candidate)===sign(proposed));
    if(idx<0){
      own.pending=null;
      throw new Error('AQ teacher proposed a non-authorized action');
    }
    return {action:legalActions[idx],behaviorLogProbability:0,
      referenceLogProbability:-Math.log(legalActions.length),predictedValue:0};
  }
  choose.accepted=({seat,round,action})=>{
    if(![0,1].includes(seat)||!Number.isInteger(round))
      throw new TypeError('Invalid teacher acceptance');
    const own=seats[seat],p=own.pending;
    if(!p)return;
    if(p.round!==round||sign(p.action)!==sign(action))
      throw new Error('AQ plan not supported by accepted Match action');
    // Only authoritative accepted selection IDs may bind own private sets.
    if(p.phase==='selection'){
      const byId=new Map(p.roster.map(pokemon=>[pokemon.instanceId,pokemon]));
      if(p.action.length!==3||p.action.some(id=>!byId.has(id)))
        throw new Error('Malformed accepted teacher selection');
      own.selected=p.action.map(id=>byId.get(id));
    }else if(p.phase==='battle'){
      // Tactical memory is updated only AFTER the accepted action receipt.
      commitStrategyV2Choice(p.view,{
        ...p.options,variant:'strategy-v2-e'},p.ranking,p.action);
    }else throw new Error('Unexpected AQ teacher accepted memory event');
    own.pending=null;
  };
  return choose;
}
