import { requireRule } from './draft.js';

// Fixed b8f23c9: Pokemon.transformInto and clearVolatile. HP, held item,
// nature and the original draft set are never copied from the target.
const RESTORE_FIELDS=['speciesId','speciesName','stats','moves','types','originalTypes','weightkg','originalAbilityId'];
export function transformCombatant(actor,target,definitions,abilityFlags) {
  if(!actor.hp || !target.hp || actor.transformed || target.transformed || actor.illusionInstanceId || target.illusionInstanceId || target.volatiles?.substitute)return false;
  const flags=abilityFlags[target.abilityId];
  requireRule(flags,'Missing Transform ability flags: '+target.abilityId);
  const moves=target.moves.map(slot=>{
    const definition=definitions[slot.id];
    requireRule(definition && Number.isFinite(definition.pp),'Missing Transform move: '+slot.id);
    const pp=Math.min(5,definition.pp);
    return {id:slot.id,name:slot.name,pp,maxPP:pp};
  });
  actor.transformOriginal=Object.fromEntries(RESTORE_FIELDS.map(field=>[field,structuredClone(actor[field])]));
  actor.transformed=true;
  actor.speciesId=target.speciesId;actor.speciesName=target.speciesName;actor.weightkg=target.weightkg;
  actor.stats={...structuredClone(target.stats),hp:actor.stats.hp};
  actor.moves=moves;
  actor.types=[...(target.volatiles?.roost?.types??target.types)];
  actor.originalTypes=[...actor.types];
  actor.boosts=structuredClone(target.boosts);
  actor.atkStage=actor.boosts.atk??0;actor.defStage=actor.boosts.def??0;
  actor.timesAttacked=target.timesAttacked??0;
  // Reference ignoringAbility suppresses notransform abilities while transformed.
  // Keep the copied identity for presentation/Trace; effective handlers see none.
  if(flags.notransform)actor.transformedAbilityId=target.abilityId;
  actor.abilityId=flags.notransform?'':target.abilityId;
  for(const id of ['dragoncheer','focusenergy','gmaxchistrike','laserfocus']) {
    delete actor.volatiles[id];
    if(target.volatiles?.[id])actor.volatiles[id]=structuredClone(target.volatiles[id]);
  }
  return true;
}

export function clearTransform(actor) {
  if(!actor.transformed)return false;
  requireRule(actor.transformOriginal,'Missing original Transform state');
  for(const [field,value] of Object.entries(actor.transformOriginal))actor[field]=structuredClone(value);
  actor.abilityId=actor.originalAbilityId;
  delete actor.transformOriginal;delete actor.transformed;delete actor.transformedAbilityId;
  return true;
}
