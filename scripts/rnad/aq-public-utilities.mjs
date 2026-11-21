// Read-only public AQ utility audit for accepted R-NaD phase decisions.
// Scores are NOT win probabilities and are NOT opponent hidden-team facts.
// Keep this separate from full-Match authority and battle reward.
import {scoreStrategyDraftRoster,rankStrategySelections}
  from '../../src/ai/team-assessment.js';
import {mapPublicAQSelectionScores} from './selection-oracle.mjs';
import {playableSets} from '../../src/production-runtime.js';
import data from '../../data/catalog/battle-data.json' with {type:'json'};

export function aqPublicUtilityScores(observation,legalActions,chosenIndex){
  if(!['draft','selection'].includes(observation?.phase)||
     !Array.isArray(legalActions)||legalActions.length<1||
     !Number.isSafeInteger(chosenIndex)||chosenIndex<0||
     chosenIndex>=legalActions.length)
    throw new TypeError('Expected accepted public phase decision');
  let scores;
  if(observation.phase==='selection'){
    const ranked=rankStrategySelections(observation,{catalog:playableSets,data});
    scores=mapPublicAQSelectionScores(ranked,legalActions).scores;
  }else{
    const offered=[...(observation.self?.carry??[]),
                   ...(observation.self?.fresh??[])];
    const byId=new Map(offered.map(set=>[set.instanceId,set]));
    if(byId.size!==offered.length||
       !Number.isSafeInteger(observation.self?.credits))
      throw new Error('Invalid own draft offer for utility scorer');
    scores=legalActions.map(ids=>{
      if(!Array.isArray(ids)||ids.length!==6)
        throw new TypeError('Malformed authority draft candidate');
      const chosen=ids.map(id=>byId.get(id));
      if(chosen.some(x=>!x))
        throw new Error('Untrusted draft candidate not in own offer');
      return scoreStrategyDraftRoster(chosen,{
        data,credits:observation.self.credits,carry:observation.self.carry??[]});
    });
  }
  if(scores.length!==legalActions.length||
     scores.some(x=>!Number.isFinite(x)))
    throw new Error('Nonfinite or mismatched public AQ candidate utility');
  const top=Math.max(...scores);
  const selected=scores[chosenIndex];
  // The actual teacher choice MUST maximize its declared public ranking.
  // Otherwise labels and action utilities do not have the same meaning.
  if(top-selected>1e-5)
    throw new Error('Accepted public AQ teacher does not maximize its declared utility');
  return scores;
}
