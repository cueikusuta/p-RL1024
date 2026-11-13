// Dedicated CPU-only, 8-credit legendary set definitions. The normal
// player catalog (sets-source -> production-sets) remains unchanged.
import special from '../data/catalog/cpu-special-sets.json' with {type:'json'};
import battleData from '../data/catalog/battle-data.json' with {type:'json'};
import additions from '../data/catalog/cpu-legendary-additions.json' with {type:'json'};

export const CPU_SPECIAL_KIND='cpu-legendary-v1';
export const CPU_SPECIAL_SETS=Object.freeze([...special.sets,...additions.sets].map(set=>Object.freeze({
  ...set,cpuSpecial:CPU_SPECIAL_KIND,
})));
export const CPU_SPECIAL_SPECIES=Object.freeze({...special.species,...additions.species});
export const CPU_SPECIAL_IDS=new Set(CPU_SPECIAL_SETS.map(set=>set.setId));
export const CPU_SPECIAL_BATTLE_DATA=Object.freeze({
  ...battleData,
  species:{...battleData.species,...CPU_SPECIAL_SPECIES},
  items:{...battleData.items,...(special.items??{}),...(additions.items??{})},
  moves:{...battleData.moves,...(special.moves??{}),...(additions.moves??{})},
  abilities:{...battleData.abilities,...(special.abilities??{}),...(additions.abilities??{})},
  abilityFlags:{...battleData.abilityFlags,...Object.fromEntries(
    [...Object.entries(special.abilities??{}),...Object.entries(additions.abilities??{})].map(([id,ability])=>[id,ability.flags??{}]))},
  megaStones:{...battleData.megaStones,...(special.megaStones??{}),...(additions.megaStones??{})},
});
export function isCpuSpecialSet(set){
  return set?.cpuSpecial===CPU_SPECIAL_KIND&&
    set.cost===8&&CPU_SPECIAL_IDS.has(set.setId)&&
    CPU_SPECIAL_SETS.some(known=>known.setId===set.setId&&
      known.speciesId===set.speciesId&&known.abilityId===set.abilityId);
}

if(CPU_SPECIAL_SETS.length!==special.sets.length+additions.sets.length||
    CPU_SPECIAL_SETS.filter(s=>s.legendClass==='sublegendary').length<2||
    CPU_SPECIAL_SETS.filter(s=>s.legendClass==='legendary').length<2||
    CPU_SPECIAL_IDS.size!==CPU_SPECIAL_SETS.length||
    CPU_SPECIAL_SETS.some(s=>!CPU_SPECIAL_SPECIES[s.speciesId]))
  throw new Error('Invalid CPU special legendary set pool');
