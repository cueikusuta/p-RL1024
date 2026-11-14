// RNAD-001 Stage 0: legal-set adapter and actor observation projection.
// Reuse the authoritative research candidates. This module never inspects
// opponent private sets, Match checkpoints or Battle internals.
import {legalPureRlTeams,legalPureRlSelections} from '../ai/pure-rl-phases.js';
import {legalPureRlActions} from '../ai/pure-rl-linear.js';

export const RNAD_OBSERVATION_SCHEMA='rnad-actor-observation-v1';
const phases=new Set(['draft','selection','battle']);
const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
const clone=x=>structuredClone(x);
const fields=(src,keys)=>object(src)?Object.fromEntries(keys
  .filter(key=>Object.hasOwn(src,key))
  .map(key=>[key,clone(src[key])])):{};
const ownKeys=['instanceId','setId','speciesId','speciesClauseId','speciesName',
  'types','baseStats','stats','cost','moves','itemId','abilityId','mega',
  'megaPreview','hp','maxHP','status','stages','statPoints','nature','pp',
  'gender','alive','canMega'];
const ownPokemon=p=>fields(p,ownKeys);
const opponentBattleKeys=['speciesId','types','hpPercent','status','remaining',
  'stages','publicObservation'];
const publicObservationKeys=['appearanceId','usedMoves','observedItemIds',
  'observedAbilityIds','moves','itemIds','abilityIds'];
const fieldKeys=['weather','terrain','trickRoomTurns','hazards','screens',
  'tailwind','gravity','room'];
const ownBattleKeys=['team','active','canSwitch','canMega','needsSwitch',
  'confirmed','remaining','side','hazards','screens'];

export function rnadLegalActions(view){
  if(!object(view))throw new TypeError('Expected phase observation');
  const phase=view.phase??(view.requestType?'battle':null);
  if(!phases.has(phase))throw new TypeError('Unsupported RNAD phase');
  const actions=phase==='draft'?legalPureRlTeams(view):
    phase==='selection'?legalPureRlSelections(view):legalPureRlActions(view);
  // Empty lists are legal only when the authority explicitly declares no decision.\n  const inactive=phase==='battle'&&(['wait','finished','confirmed'].includes(view.requestType)||view.self?.confirmed===true);\n  return validateRnadLegalActions(actions,{allowEmpty:inactive});
}
export function validateRnadLegalActions(actions,{allowEmpty=false}={}){
  if(!Array.isArray(actions)||actions.length>60000||
    (!allowEmpty&&actions.length===0))
    throw new TypeError('Invalid legal action list length');
  const signatures=new Set();
  for(const action of actions){
    if(!(Array.isArray(action)||object(action)))
      throw new TypeError('Invalid candidate action');
    const key=JSON.stringify(action);
    if(!key||signatures.has(key))throw new TypeError('Duplicate/invalid legal action');
    signatures.add(key);
  }
  return clone(actions);
}
export function projectRnadObservation(view,{seat,context=null}={}){
  if(![0,1].includes(seat)||!object(view))
    throw new TypeError('Expected public view and own seat');
  const phase=view.phase??(view.requestType?'battle':null);
  if(!phases.has(phase))throw new TypeError('Unsupported RNAD phase');
  if(!object(view.self))throw new TypeError('Missing own view');
  const result={schema:RNAD_OBSERVATION_SCHEMA,phase,seat};
  if(phase==='battle'){
    Object.assign(result,fields(view,['turn','requestType']));
    result.self=fields(view.self,ownBattleKeys);
    if(Array.isArray(view.self.team))
      result.self.team=view.self.team.map(ownPokemon);
    result.opponent=fields(view.opponent,opponentBattleKeys);
    if(Object.hasOwn(result.opponent,'publicObservation'))
      result.opponent.publicObservation=fields(
        result.opponent.publicObservation,publicObservationKeys);
    result.field=fields(view.field,fieldKeys);
  }else{
    Object.assign(result,fields(view,['round','score','budget','grant']));
    result.self=fields(view.self,['credits','teamCost','selected']);
    for(const key of ['fresh','carry','team']){
      if(Array.isArray(view.self[key]))result.self[key]=view.self[key].map(ownPokemon);
    }
    result.opponent=fields(view.opponent,['ready','teamConfirmed',
      'selectionConfirmed','species','carrySpecies','carryover']);
  }
  // Only accept history emitted by the seat-scoped, accepted-action pipeline.
  // No arbitrary external object may be smuggled into actor memory.
  if(context!==null){
    if(!object(context)||context.schema!=='p-rl1024-phase-pipeline-v1'||
      context.seat!==seat)throw new TypeError('Invalid seat-scoped context');
    // Pipeline handoff/completed/current are untrusted nested JSON; until a\n    // versioned recursive public schema exists, never expose them to actors.\n    result.context=fields(context,['round']);
  }
  return result;
}
