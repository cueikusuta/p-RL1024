// Frozen AQ-Base public 61D correction over legal V2E action scores.
// Opt-in research teacher only: this does not modify the source AQ-Base or
// the RNAD-001 learner and never reads opponent hidden sets/true actions.
import {chooseStrategyTeam} from '../../src/ai/team-assessment.js';
import {chooseTacticalSelection} from '../../src/ai/selection-tactical.js';
import {createModel as createV2E,rankActions as rankV2E}
  from '../../models/v2e.mjs';
import {createModel as createAQ,rankActions as rankAQ,
  weightsSha256 as AQ_WEIGHTS_SHA256} from '../../models/aq-base.mjs';
import {playableSets} from '../../src/production-runtime.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};
import {RNAD_OBSERVATION_SCHEMA} from '../../src/ml/legal-action-adapter.js';

export const AQ_TEACHER_ID='rnad-aq-public-ranked-v1';
export {AQ_WEIGHTS_SHA256};
const sign=x=>JSON.stringify(x);
export function createPublicAQTeacher(){
  const model=createAQ(),v2=createV2E();
  return function choose({observation,legalActions}){
    if(observation?.schema!==RNAD_OBSERVATION_SCHEMA||
      !['draft','selection','battle'].includes(observation.phase)||
      !Array.isArray(legalActions)||!legalActions.length)
      throw new TypeError('AQ teacher requires only projected actor observation and legal actions');
    let proposed;
    if(observation.phase==='draft'){
      proposed=chooseStrategyTeam(observation,{data,fastDraftValidation:true});
    }else if(observation.phase==='selection'){
      proposed=chooseTacticalSelection(observation,{data,sets:playableSets});
    }else{
      const ranked=rankV2E(v2,observation,legalActions);
      if(ranked.length===0)throw new Error('Frozen V2E teacher returned no actions');
      const sorted=ranked.map(row=>row.action);
      const baselineScores=ranked.map(row=>row.q);
      const corrected=rankAQ(model,observation,sorted,{
        data,baselineScores,roundWins:0,roundLosses:0});
      proposed=corrected[0].action;
    }
    const idx=legalActions.findIndex(x=>sign(x)===sign(proposed));
    if(idx<0)throw new Error('AQ teacher proposed a non-authorized action');
    return {action:legalActions[idx],behaviorLogProbability:0,
      referenceLogProbability:-Math.log(legalActions.length),predictedValue:0};
  };
}
