import { changeBoosts, whiteHerb } from './battle-moves.js';
import { requireRule } from './draft.js';
import { clearStatus, setStatus } from './battle-status.js';
import { setTerrain, setWeather, placeHazard, clearScreens, entryWeather, updateMimicry } from './battle-field.js';
import { updateHungerSwitch } from './battle-forms.js';
import { makesContact } from './battle-secondary.js';
import { abilityLabels, itemLabels } from '../data/catalog/labels-ja.js';
import { endCondition } from './battle-messages.js';

// Entry events use current active opponents. Ability copies and activation
// order are handled separately when Trace is connected.
export function entryAbility(pokemon,foe,team,field=null,totalFainted=null) {
  const result={};
  if(!pokemon.hp)return result;
  if(pokemon.abilityId==='oblivious') {endCondition(pokemon,'attract',{causeAbilityId:'oblivious'});endCondition(pokemon,'taunt',{causeAbilityId:'oblivious'});}
  if(pokemon.abilityId==='owntempo')endCondition(pokemon,'confusion');
  if(pokemon.abilityId==='thermalexchange' && pokemon.status==='brn')clearStatus(pokemon);
  if(pokemon.abilityId==='waterbubble' && pokemon.status==='brn')clearStatus(pokemon);
  if(pokemon.abilityId==='screencleaner' && field)result.clearedScreens=clearScreens(field);
  if(pokemon.abilityId==='intimidate' && foe.hp && !foe.volatiles?.substitute) {
    result.intimidate=changeBoosts(foe,{atk:-1},{source:pokemon,effect:'intimidate'});
  }
  if(pokemon.abilityId==='intrepidsword'&&!pokemon.intrepidSwordUsed){
    pokemon.intrepidSwordUsed=true;
    result.intrepidSword=changeBoosts(pokemon,{atk:1},{source:pokemon,effect:'intrepidsword'});
  }
  if(pokemon.abilityId==='supremeoverlord')pokemon.supremeOverlordFallen=Math.min(5,
    Number.isInteger(totalFainted)&&totalFainted>=0 ? totalFainted : team.filter(member=>member.hp===0).length);
  if(pokemon.abilityId==='frisk' && foe.hp && foe.itemId)result.friskedItemId=foe.itemId;
  if(['pressure','moldbreaker','vesselofruin','swordofruin','asonespectrier','teravolt','turboblaze'].includes(pokemon.abilityId))result.announcedAbilityId=pokemon.abilityId;
  // White Herb's onAnySwitchIn observes all boosts after entry events.
  whiteHerb(pokemon);whiteHerb(foe);
  return result;
}
const abilityEndVolatiles={unburden:'unburden',flashfire:'flashfire',protosynthesis:'protosynthesis',quarkdrive:'quarkdrive'};

// End removes only volatile state owned by the ability. Charge is a separate
// volatile and intentionally survives an ability change.
export function endAbility(pokemon,abilityId) {
  if(!pokemon)return false;
  const volatile=abilityEndVolatiles[abilityId];
  if(!volatile || !pokemon.volatiles?.[volatile])return false;
  delete pokemon.volatiles[volatile];
  return true;
}
export function resetAbilityState(pokemon) {
  if(!pokemon)return false;
  pokemon.proteanUsed=false;
  pokemon.supremeOverlordFallen=0;
  return true;
}
function teamOf(teams,pokemon) {
  return teams.find(team=>team.includes(pokemon)) ?? [];
}
function totalFaintedFor(pokemon,teams,totalFaintedBySide) {
  const seat=teams.findIndex(team=>team.includes(pokemon));
  const total=totalFaintedBySide?.[seat];
  if(Number.isInteger(total)&&total>=0)return total;
  return teamOf(teams,pokemon).filter(member=>member!==pokemon&&member.hp===0).length;
}
// Ability Start runs only supported onStart effects. Switch-in-only behavior
// (hazards, seeds, White Herb, entry items, Trace/Transform) stays out.
// Current-catalog onStart audit: weather/terrain surges, Intimidate, Frisk,
// Supreme Overlord, Screen Cleaner, and Mimicry run here. Forecast, Unnerve,
// and Cloud Nine are resynced by the production hit callback; Mold Breaker and
// Pressure only announce themselves, and Flash Fire's onStart belongs to its
// volatile condition.
export function startAbility(pokemon,{foe=null,team=[],field=null,active=[],message=null,publishEntryAbility=null,allowPendingFaint=false,totalFainted=null}={}) {
  const result={};
  if(!pokemon || !pokemon.hp&&!allowPendingFaint)return result;
  if(field && entryWeather(field,pokemon,active,{allowPendingFaint}))result.fieldChanged=true;
  if(field && updateMimicry(pokemon,field.terrain))result.typeChanged=true;
  if(pokemon.abilityId==='screencleaner' && field)result.clearedScreens=clearScreens(field);
  if(pokemon.abilityId==='intimidate' && foe?.hp && !foe.volatiles?.substitute) {
    result.intimidate=changeBoosts(foe,{atk:-1},{source:pokemon,effect:'intimidate'});
  }
  if(pokemon.abilityId==='intrepidsword'&&!pokemon.intrepidSwordUsed){
    pokemon.intrepidSwordUsed=true;
    result.intrepidSword=changeBoosts(pokemon,{atk:1},{source:pokemon,effect:'intrepidsword'});
  }
  if(pokemon.abilityId==='supremeoverlord')pokemon.supremeOverlordFallen=Math.min(5,
    Number.isInteger(totalFainted)&&totalFainted>=0 ? totalFainted : team.filter(member=>member!==pokemon&&member.hp===0).length);
  if(pokemon.abilityId==='frisk' && foe?.hp && foe.itemId)result.friskedItemId=foe.itemId;
  if(message) {
    const announce=id=>{
      if(!publishEntryAbility?.(pokemon,id))
        message(`${pokemon.speciesName} の ${abilityLabels[id]??'とくせい'}！`);
    };
    if(['pressure','moldbreaker','teravolt','turboblaze'].includes(pokemon.abilityId))announce(pokemon.abilityId);
    if(result.intrepidSword)announce('intrepidsword');
    if(result.intimidate && Object.keys(result.intimidate).length)announce('intimidate');
    if(result.clearedScreens)announce('screencleaner');
    if(result.friskedItemId
      && !publishEntryAbility?.(pokemon,'frisk',{friskTarget:foe,
        friskedItemId:result.friskedItemId}))
      message(`${pokemon.speciesName} の おみとおし！ ${foe.speciesName} は ${itemLabels[result.friskedItemId]??result.friskedItemId} を持っている！`);
  }
  return result;
}
// Pinned setAbility order: End, replace, reset state, announce, Start.
export function replaceAbility(pokemon,abilityId,{source=null,foe=null,teams=[],field=null,active=[],message=null,publishAbilityLifecycle=null,publishEntryAbility=null,abilityFlags={},totalFaintedBySide=null}={}) {
  if(!pokemon?.hp || !abilityId)return false;
  const oldAbilityId=pokemon.abilityId;
  if(oldAbilityId===abilityId)return false;
  if(abilityFlags[abilityId]?.cantsuppress || abilityFlags[oldAbilityId]?.cantsuppress)return false;
  endAbility(pokemon,oldAbilityId);
  pokemon.abilityId=abilityId;
  resetAbilityState(pokemon);
  if(message && !publishAbilityLifecycle?.({
    kind:'replaced',holder:pokemon,source:source??pokemon,
    oldAbilityId,newAbilityId:abilityId,confirmed:true,
  })) {
    // Compatibility-only fallback for an unhandled lifecycle effect.
    message(`${(source??pokemon).speciesName} の ${abilityLabels[abilityId] ?? 'とくせい'}！`);
    message(`${pokemon.speciesName} の ${abilityLabels[oldAbilityId] ?? 'とくせい'} は ${abilityLabels[abilityId] ?? 'とくせい'} になった！`);
  }
  startAbility(pokemon,{foe,team:teamOf(teams,pokemon),field,active,message,publishEntryAbility,totalFainted:totalFaintedFor(pokemon,teams,totalFaintedBySide)});
  return true;
}
// Pinned skillSwap order: validate, announce, End both, swap, reset both,
// then Start the old first ability on the second holder and vice versa.
export function swapAbilities(first,second,{teams=[],field=null,active=[],message=null,publishAbilityLifecycle=null,publishEntryAbility=null,abilityFlags={},activation=null,allowPendingFaint=false,totalFaintedBySide=null}={}) {
  if(!first || !second || (!first.hp || !second.hp)&&!allowPendingFaint)return false;
  const firstAbilityId=first.abilityId,secondAbilityId=second.abilityId;
  if(abilityFlags[firstAbilityId]?.failskillswap || abilityFlags[secondAbilityId]?.failskillswap)return false;
  endAbility(first,firstAbilityId);
  endAbility(second,secondAbilityId);
  first.abilityId=secondAbilityId;
  second.abilityId=firstAbilityId;
  resetAbilityState(first);resetAbilityState(second);
  // Emit at the original announcement position, after the successful swap
  // is committed and before either newly installed ability starts.
  if(message && activation && !publishAbilityLifecycle?.({
    kind:'swapped',first,second,firstAbilityId,secondAbilityId,confirmed:true,
  }))message(activation);
  startAbility(second,{foe:first,team:teamOf(teams,second),field,active,message,publishEntryAbility,allowPendingFaint,totalFainted:totalFaintedFor(second,teams,totalFaintedBySide)});
  startAbility(first,{foe:second,team:teamOf(teams,first),field,active,message,publishEntryAbility,allowPendingFaint,totalFainted:totalFaintedFor(first,teams,totalFaintedBySide)});
  return true;
}
export function updateUnnerve(active=[],{includeFainted=false}={}) {
  for(let seat=0;seat<active.length;seat++) {
    const pokemon=active[seat],foe=active[1-seat];
    if(!pokemon?.volatiles)continue;
    if(pokemon.hp>0 && (includeFainted||foe?.hp>0) && ['unnerve','asonespectrier','asoneglastrier'].includes(foe?.abilityId))pokemon.volatiles.unnerved=true;
    else delete pokemon.volatiles.unnerved;
  }
}
export function updateAbility(pokemon) {
  if(!pokemon?.hp)return false;
  const curedStatus={vitalspirit:'slp',insomnia:'slp',limber:'par',thermalexchange:'brn',waterbubble:'brn'}[pokemon.abilityId];
  if(curedStatus===pokemon.status || pokemon.abilityId==='immunity' && ['psn','tox'].includes(pokemon.status))return clearStatus(pokemon);
  if(pokemon.abilityId==='oblivious') {
    let changed=false;
    for(const id of ['attract','taunt'])if(endCondition(pokemon,id,{causeAbilityId:'oblivious'}))changed=true;
    return changed;
  }
  return false;
}
export function residualAbility(pokemon,rng,{sun=false,data=null}={}) {
  if(!pokemon.hp)return {};
  if(pokemon.abilityId==='hungerswitch')return updateHungerSwitch(pokemon,data)?{changedFormId:pokemon.speciesId}:{};
  if(pokemon.abilityId==='harvest') {
    // The residual chance precedes item eligibility in the pinned callback.
    const restored=sun || rng.integer(2)===0;
    if(pokemon.itemId || !pokemon.lastBerryId)return {};
    if(restored) {
      const itemId=pokemon.lastBerryId;
      pokemon.itemId=itemId;pokemon.lastBerryId='';
      return {restoredBerryId:itemId};
    }
    return {};
  }
  if(pokemon.abilityId==='speedboost' && pokemon.activeTurns)return changeBoosts(pokemon,{spe:1},{effect:'speedboost'});
  if(pokemon.abilityId==='moody') {
    const stats=['atk','def','spa','spd','spe'];
    const available=stats.filter(stat=>(pokemon.boosts?.[stat]??0)<6);
    const up=available.length ? available[rng.integer(available.length)] : null;
    const downCandidates=stats.filter(stat=>(pokemon.boosts?.[stat]??0)>-6 && stat!==up);
    const down=downCandidates.length ? downCandidates[rng.integer(downCandidates.length)] : null;
    const changes={};if(up)changes[up]=2;if(down)changes[down]=-1;
    return changeBoosts(pokemon,changes,{effect:'moody'});
  }
  return {};
}
export function damagingHitAbility(move,actor,target,{rng,chart,field,active,targetSeat,abilityFlags={},damage,substitute=false,weather='',terrain='',grounded=()=>true,teams=[],message=null,publishAbilityLifecycle=null,publishEntryAbility=null,totalFaintedBySide=null}={}) {
  if(typeof damage !== 'number' || substitute)return false;
  // Registration is fixed before this callback can swap either ability.
  const defenderAbility=target.abilityId;
  let changed=false;
  if(defenderAbility==='stamina' && target.hp)changed ||= Object.keys(changeBoosts(target,{def:1},{effect:'stamina'})).length>0;
  if(defenderAbility==='weakarmor' && target.hp && move.category==='Physical') {
    changed ||= Object.keys(changeBoosts(target,{def:-1,spe:2},{effect:'weakarmor'})).length>0;
  }
  if(defenderAbility==='justified' && target.hp && move.type==='Dark')changed ||= Object.keys(changeBoosts(target,{atk:1},{effect:'justified'})).length>0;
  if(defenderAbility==='electromorphosis' && target.hp) {target.volatiles??={};target.volatiles.charge=true;changed=true;}
  // DamagingHit runs before faint processing in the pinned reference, so a
  // lethal hit still lets Seed Sower leave terrain behind.
  if(defenderAbility==='seedsower')changed ||= setTerrain(field,'grassyterrain',target,active);
  if(defenderAbility==='gooey' && makesContact(move,actor) && actor.hp)changed ||= Object.keys(changeBoosts(actor,{spe:-1},{source:target,effect:'gooey'})).length>0;
  if(defenderAbility==='sandspit' && field)changed ||= setWeather(field,'sandstorm',target);
  if(defenderAbility==='toxicdebris' && move.category==='Physical')changed ||= placeHazard(field,1-targetSeat,'toxicspikes');
  if(makesContact(move,actor) && defenderAbility==='mummy' && actor.hp && actor.abilityId!=='mummy' && !abilityFlags[actor.abilityId]?.cantsuppress) {
    // Acquire the lifecycle result without short-circuiting other activations.
    changed=replaceAbility(actor,'mummy',{source:target,foe:target,teams,field,active:active??[],message,publishAbilityLifecycle,publishEntryAbility,abilityFlags,totalFaintedBySide}) || changed;
  }
  if(makesContact(move,actor) && defenderAbility==='wanderingspirit' &&
    !abilityFlags[actor.abilityId]?.failskillswap && !abilityFlags[target.abilityId]?.failskillswap) {
    const actorAbilityId=actor.abilityId,defenderAbilityId=target.abilityId;
    changed=swapAbilities(actor,target,{teams,field,active:active??[],message,publishAbilityLifecycle,publishEntryAbility,abilityFlags,allowPendingFaint:true,totalFaintedBySide,
      activation:message
        ? `${target.speciesName} の ${abilityLabels.wanderingspirit}！ ${actor.speciesName} の ${abilityLabels[actorAbilityId] ?? 'とくせい'} と ${target.speciesName} の ${abilityLabels[defenderAbilityId] ?? 'とくせい'} を入れ替えた！`
        : null}) || changed;
  }
  if(defenderAbility==='spicyspray' && actor.hp)changed ||= setStatus(actor,'brn',rng,{
    chart,source:target,sunny:weather==='sunnyday',electricTerrain:terrain==='electricterrain',
    mistyTerrain:terrain==='mistyterrain',grounded:grounded(actor),
  });
  return changed;
}
export function copyTrace(pokemon,foe,flags,rng) {
  if(!pokemon.hp || pokemon.abilityId!=='trace' || !foe.hp)return false;
  const copiedId=foe.abilityId==='noability'?'noability':foe.abilityId||foe.transformedAbilityId;
  if(!copiedId || copiedId==='noability')return false;
  const metadata=flags[copiedId];requireRule(metadata,'Missing ability flags');
  if(metadata.notrace || metadata.cantsuppress)return false;
  rng.integer(1); // fixed reference sample still draws for a single opponent
  pokemon.abilityId=copiedId;
  pokemon.proteanUsed=false;pokemon.supremeOverlordFallen=0;
  if(pokemon.abilityId==='insomnia' && pokemon.status==='slp' || pokemon.abilityId==='vitalspirit' && pokemon.status==='slp' || pokemon.abilityId==='limber' && pokemon.status==='par' || pokemon.abilityId==='waterbubble' && pokemon.status==='brn')clearStatus(pokemon);
  return true;
}
