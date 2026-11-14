// Opt-in adapters. Existing direct runStrategyMatch policies are unchanged.
import {chooseTeam,chooseSelection} from './draft-ai.js';
import {chooseBattleAction} from './battle-ai.js';
import {chooseIntermediateAction} from './intermediate-ai.js';
import {chooseStrategyTeam,chooseStrategySelection} from '../../../src/ai/team-assessment.js';
import {chooseTacticalSelection} from '../../../src/ai/selection-tactical.js';
import {playableSets} from '../../../src/production-runtime.js';
import data from '../../../data/catalog/battle-data.json' with {type:'json'};

const handlers={
  draft:{
    basic:({view,rng})=>chooseTeam(view,{data,rng}),
    strategy:({view})=>chooseStrategyTeam(view,{data,fastDraftValidation:true}),
  },
  selection:{
    basic:({view,rng})=>chooseSelection(view,{data,sets:playableSets,rng}),
    strategy:({view})=>chooseStrategySelection(view,{catalog:playableSets,data}),
    tactical:({view})=>chooseTacticalSelection(view,{data,sets:playableSets}),
  },
  battle:{
    basic:({view,rng,memory})=>chooseBattleAction(view,{data,rng,memory}),
    intermediate:({view,rng,memory})=>chooseIntermediateAction(view,{data,rng,memory}),
  },
};

// A stable, opt-in selection vocabulary. Custom handlers can use any model
// family (including RL-Linear, RL-Deep, V2E or AQ-Base) through the same API.
export const BUILTIN_PHASE_AGENTS=Object.freeze(Object.fromEntries(
  Object.entries(handlers).map(([phase,choices])=>[phase,Object.freeze(Object.keys(choices))])));
export function createBuiltinPhaseAgents({draft='basic',selection='basic',battle='basic'}={}){
  const configured={draft,selection,battle},result={};
  for(const phase of ['draft','selection','battle']){
    const requested=configured[phase];
    if(typeof requested==='string'){
      if(!Object.hasOwn(handlers[phase],requested))
        throw new TypeError('Unsupported '+phase+' AI: '+requested);
      const memory={};
      result[phase]={id:requested,select:args=>handlers[phase][requested]({...args,memory})};
    }else if(requested&&typeof requested==='object'){
      result[phase]=requested;
    }else throw new TypeError('Invalid '+phase+' AI');
  }
  return result;
}

// Reuse a model's accepted-action session without coupling its private
// training state to the generic cross-phase JSON context.
export function sessionPhaseAgent(id,session){
  if(!session||['select','commit','finish','stats'].some(key=>typeof session[key]!=='function'))
    throw new TypeError('Expected a synchronous model session');
  return {id,select:({view})=>session.select(view),
    accepted:()=>session.commit(),
    finish:({result,seat})=>result.completed?
      session.finish(result.winner===seat?1:-1):session.stats()};
}
