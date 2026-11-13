// I737 Phase B: observational selection-difference classification.
// This consumes only the caller's OWN confirmed roster and public opponent
// species; categories are hypotheses, never causal claims about wins.
import {evaluateHybridFeatures,createHybridCoverageContext}
  from './hybrid-selection.js';

export function diagnoseSelectionDifference(view,{basic,alternative,data,sets}={}) {
  const team=view?.self?.team;
  if(!Array.isArray(team)||team.length!==6)
    throw new TypeError('Diagnosis requires six own confirmed members');
  const ownIds=new Set(team.map(p=>p.instanceId));
  for(const selection of [basic,alternative]) {
    if(!Array.isArray(selection)||selection.length!==3||
       new Set(selection).size!==3||selection.some(id=>!ownIds.has(id)))
      throw new TypeError('Diagnosis requires legal own selection IDs');
  }
  const original=new Set(basic),other=new Set(alternative);
  const trioChanged=basic.some(id=>!other.has(id));
  const leadChanged=basic[0]!==alternative[0];
  const benchChanged=basic.slice(1).join('|')!==alternative.slice(1).join('|');
  const context=createHybridCoverageContext(view,{data,sets});
  const b=evaluateHybridFeatures(basic,context);
  const a=evaluateHybridFeatures(alternative,context);
  const selectedMegaCount=selection=>selection.filter(id=>
    team.find(p=>p.instanceId===id)?.mega).length;
  const megaConflict=selectedMegaCount(alternative)>1;
  const possibleCauses=[];
  if(trioChanged)possibleCauses.push('trio-composition');
  else if(leadChanged)possibleCauses.push('lead-only');
  else if(benchChanged)possibleCauses.push('bench-order-only');
  if(b.coverage!==null&&a.coverage!==null&&a.coverage<b.coverage)
    possibleCauses.push('offensive-complement-loss');
  if(b.unfavorable!==null&&a.unfavorable!==null&&
     a.unfavorable<b.unfavorable)
    possibleCauses.push('worst-matchup-risk');
  if(megaConflict)possibleCauses.push('multi-mega-opportunity');
  if(context.unknownPublicSpecies||a.unknown||b.unknown)
    possibleCauses.push('unmodelled-public-matchup');
  return {
    trioChanged,leadChanged,benchChanged,megaConflict,
    unknownPublicSpecies:context.unknownPublicSpecies,
    evaluatedPublicFoes:a.evaluated,
    possibleCauses,
    // No opponent-private or own set IDs leave this API.
    coverageDifference:a.coverage===null||b.coverage===null?null:a.coverage-b.coverage,
    unfavorableDifference:a.unfavorable===null||b.unfavorable===null?null:
      a.unfavorable-b.unfavorable,
    warning:'Observational classification only; paired real Match interventions are needed for causality.',
  };
}
