// Pure battle rule used by both the attack resolver and production hit hooks.
// It must not depend on the high-level attack resolution module.
export function grounded(pokemon,{ignoreAbility=false}={}) {
  if (pokemon.itemId==='ironball')return true;
  return pokemon.itemId!=='airballoon' && !pokemon.types.includes('Flying') && (ignoreAbility || !['levitate','eelevate'].includes(pokemon.abilityId));
}
