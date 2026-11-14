// I975: exact I877 frozen 61D AQ-Base as a research-only opponent.
// V2E contributes its authorized legal candidates and baseline scores to
// AQ-Base, but the pure RL student never sees either score or AQ-Base weights.
import {createHash} from 'node:crypto';
import {validatePublicActionQ,rankPublicActionQ}
  from '../../../src/ai/v3-public-action-q.js';
import {validateV3ValueModel} from '../../../src/ai/v3-value.js';
import {buildV3BattleSample} from '../../../src/ai/v3-learned-models.js';

export const AQ_BASE_ID='AQ-Base';
export const AQ_BASE_REPORT_SHA256=
  'ce616ec641f0986aa45065bdec257625235069953f0c759ef47856149aec5f67';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');

export function readFrozenAQBase(bytes){
  if(!(bytes instanceof Uint8Array)&&typeof bytes!=='string')
    throw new TypeError('AQ-Base requires report bytes');
  if(sha(bytes)!==AQ_BASE_REPORT_SHA256)
    throw new Error('Untrusted AQ-Base artifact SHA-256');
  const source=JSON.parse(Buffer.from(bytes).toString('utf8'));
  if(source.schema!=='i877-public-action-q-twelve-feedback-v1'||
     source.cycles!==12||source.completedMatches!==556||
     source.noFinalHoldoutUsed!==true)
    throw new Error('Untrusted original AQ-Base provenance');
  const model=validatePublicActionQ(source.models?.bestLossModel);
  validateV3ValueModel(source.models?.critic);
  if(model.featureCount!==61)
    throw new Error('Expected original AQ-Base 61D');
  Object.freeze(model.weights);
  Object.freeze(model);
  return {model,reportSha256:AQ_BASE_REPORT_SHA256,
    modelSha256:sha(JSON.stringify(model))};
}

export function chooseFrozenAQBase(model,view,actions,baselineScores,{
  data,roundWins=0,roundLosses=0
}={}){
  validatePublicActionQ(model);
  if(model.featureCount!==61||!Array.isArray(actions)||!actions.length||
     !Array.isArray(baselineScores)||actions.length!==baselineScores.length)
    throw new TypeError('AQ-Base requires its own ordered legal action rows');
  const {candidates}=buildV3BattleSample(view,actions[0],actions,{
    data,roundWins,roundLosses
  });
  if(candidates.some(row=>row.length!==61))
    throw new Error('AQ-Base public feature contract is not 61D');
  const selectedIndex=rankPublicActionQ(model,candidates,baselineScores).selectedIndex;
  return {selectedIndex,candidates};
}
