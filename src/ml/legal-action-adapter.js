// RNAD-001 Stage 0: legal-set adapter and actor observation projection.
// Reuse the authoritative research candidates. This module never inspects
// opponent private sets, Match checkpoints or Battle internals.
import {legalPureRlTeams,legalPureRlSelections} from '../ai/pure-rl-phases.js';
import {legalPureRlActions} from '../ai/pure-rl-linear.js';

export const RNAD_OBSERVATION_SCHEMA='rnad-actor-observation-v1';
const phases=new Set(['draft','selection','battle']);
// The real Battle uses phase='turn'/'pivot'/'replacement', not 'battle'.
const phaseOf=view=>['draft','selection'].includes(view?.phase)
  ?view.phase:(typeof view?.requestType==='string'?'battle':null);
const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
// structuredClone of primitive allowlisted fields is a no-op. Preserve
// structuredClone validation for objects, functions and symbols.
const clone=x=>x===null||
  (typeof x!=='object'&&typeof x!=='function'&&typeof x!=='symbol')?
  x:structuredClone(x);
const fields=(src,keys)=>object(src)?Object.fromEntries(keys
  .filter(key=>Object.hasOwn(src,key))
  .map(key=>[key,clone(src[key])])):{};
const ownKeys=['instanceId','setId','speciesId','speciesClauseId','speciesName',
  'types','baseStats','stats','cost','moves','itemId','abilityId','mega',
  'megaPreview','hp','maxHP','status','stages','statPoints','nature','pp',
  'gender','alive','canMega','megaCandidate','boosts'];
const ownPokemon=p=>fields(p,ownKeys);
const opponentBattleKeys=['speciesId','types','hpPercent','status','remaining',
  'stages','publicObservation'];
const publicObservationKeys=['appearanceId','usedMoves','observedItemIds',
  'observedAbilityIds'];
const fieldKeys=['weather','terrain','trickRoomTurns','sideEffects'];
const ownBattleKeys=['active','canSwitch','canMega','needsSwitch',
  'confirmed','remaining','side','hazards','screens'];

export function rnadLegalActions(view){
  if(!object(view))throw new TypeError('Expected phase observation');
  const phase=phaseOf(view);
  if(!phases.has(phase))throw new TypeError('Unsupported RNAD phase');
  const actions=phase==='draft'?legalPureRlTeams(view):
    phase==='selection'?legalPureRlSelections(view):legalPureRlActions(view);
  // Empty lists are legal only when the authority explicitly declares no decision.
  const active=view.self?.team?.[view.self?.active];
  const inactive=phase==='battle'&&(
    ['wait','finished'].includes(view.requestType)||view.self?.confirmed===true||
    (view.requestType==='replacement'&&active?.hp>0&&view.self?.needsSwitch!==true));
  return validateRnadLegalActions(actions,{allowEmpty:inactive});
}
export function validateRnadLegalActions(actions,{allowEmpty=false,includeSignatures=false}={}){
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
  // Internal receipt collector can reuse the signatures computed during
  // strict duplicate detection, without serializing every action again.
  const safeActions=clone(actions);
  return includeSignatures?{actions:safeActions,signatures}:safeActions;
}
export function projectRnadObservation(view,{seat,context=null}={}){
  if(![0,1].includes(seat)||!object(view))
    throw new TypeError('Expected public view and own seat');
  const phase=phaseOf(view);
  if(!phases.has(phase))throw new TypeError('Unsupported RNAD phase');
  if(!object(view.self))throw new TypeError('Missing own view');
  const result={schema:RNAD_OBSERVATION_SCHEMA,phase,seat};
  if(phase==='battle'){
    Object.assign(result,fields(view,['turn','requestType']));
    if(!Array.isArray(view.self.team))throw new TypeError('Missing own battle team');
    result.self=fields(view.self,ownBattleKeys);
    result.self.team=view.self.team.map(ownPokemon);
    result.opponent=fields(view.opponent,opponentBattleKeys.filter(x=>
      !['stages','publicObservation','types'].includes(x)));
    if(Object.hasOwn(result.opponent,'speciesId')&&
      typeof result.opponent.speciesId!=='string')
      throw new TypeError('Invalid public opponent species ID');
    if(Object.hasOwn(result.opponent,'status')&&
      result.opponent.status!==null&&typeof result.opponent.status!=='string')
      throw new TypeError('Invalid public opponent status');
    for(const key of ['hpPercent','remaining'])if(Object.hasOwn(result.opponent,key)&&
      !Number.isFinite(result.opponent[key]))
      throw new TypeError('Invalid public opponent scalar');
    if(view.opponent?.types!=null){
      if(!Array.isArray(view.opponent.types)||
        !view.opponent.types.every(x=>typeof x==='string'))
        throw new TypeError('Invalid public opponent types');
      result.opponent.types=[...view.opponent.types];
    }
    if(view.opponent?.publicObservation!=null){
      const raw=view.opponent.publicObservation;
      if(!object(raw))throw new TypeError('Invalid public reveal');
      const obs=fields(raw,['appearanceId']);
      if(Object.hasOwn(obs,'appearanceId')&&
        typeof obs.appearanceId!=='string')
        throw new TypeError('Invalid public appearance');
      for(const key of ['usedMoves','observedItemIds','observedAbilityIds']){
        if(Object.hasOwn(raw,key)){
          if(!Array.isArray(raw[key])||raw[key].length>256||
            !raw[key].every(x=>typeof x==='string'))
            throw new TypeError('Invalid public revealed '+key);
          obs[key]=[...raw[key]];
        }
      }
      result.opponent.publicObservation=obs;
    }
    if(view.opponent?.stages!==undefined){
      const stageNames=['atk','def','spa','spd','spe','accuracy','evasion'];
      const raw=view.opponent.stages;
      if(!object(raw))throw new TypeError('Invalid public opponent stages');
      result.opponent.stages=fields(raw,stageNames);
      if(Object.values(result.opponent.stages).some(x=>!Number.isFinite(x)))
        throw new TypeError('Invalid opponent stage value');
    }
    result.field=fields(view.field,fieldKeys.filter(key=>key!=='sideEffects'));
    // Field/weather/terrain are public ONLY as versioned primitive values.
    // Clone-based projection without type validation could carry a nested
    // object containing training-only or otherwise private data.
    for(const name of ['weather','terrain']){
      if(Object.hasOwn(result.field,name)&&result.field[name]!==null&&
        typeof result.field[name]!=='string')
        throw new TypeError('Invalid visible field '+name);
    }
    if(Object.hasOwn(result.field,'trickRoomTurns')&&
      (!Number.isSafeInteger(result.field.trickRoomTurns)||
        result.field.trickRoomTurns<0||
        result.field.trickRoomTurns>64))
      throw new TypeError('Invalid visible Trick Room turns');
    if(Object.hasOwn(view.field??{},'sideEffects')){
      const effects=view.field.sideEffects;
      if(!Array.isArray(effects)||effects.length!==2||
        effects.some(side=>!Array.isArray(side)||side.length>32))
        throw new TypeError('Invalid visible field side effects');
      result.field.sideEffects=effects.map(side=>side.map(entry=>{
        if(!object(entry)||typeof entry.id!=='string'||!entry.id)
          throw new TypeError('Invalid visible side-effect ID');
        const publicEntry=fields(entry,['id','layers','duration']);
        for(const name of ['layers','duration']){
          if(Object.hasOwn(publicEntry,name)&&
            (!Number.isSafeInteger(publicEntry[name])||
              publicEntry[name]<0||publicEntry[name]>64))
            throw new TypeError('Invalid visible side-effect '+name);
        }
        return publicEntry;
      }));
    }
  }else{
    Object.assign(result,fields(view,['round','score','budget','grant']));
    result.self=fields(view.self,['credits','teamCost','selected']);
    for(const key of ['fresh','carry','team']){
      if(Array.isArray(view.self[key]))result.self[key]=view.self[key].map(ownPokemon);
    }
    result.opponent=fields(view.opponent,['ready','teamConfirmed',
      'selectionConfirmed']);
    if(Array.isArray(view.opponent?.species)){
      if(!view.opponent.species.every(x=>typeof x==='string'))
        throw new TypeError('Invalid public opponent species');
      result.opponent.species=[...view.opponent.species];
    }
  }
  // Only accept history emitted by the seat-scoped, accepted-action pipeline.
  // No arbitrary external object may be smuggled into actor memory.
  if(context!==null){
    if(!object(context)||context.schema!=='p-rl1024-phase-pipeline-v1'||
      context.seat!==seat)throw new TypeError('Invalid seat-scoped context');
    // Pipeline handoff/completed/current are untrusted nested JSON; until a
    // versioned recursive public schema exists, never expose them to actors.
    result.context=fields(context,['round']);
  }
  return result;
}
