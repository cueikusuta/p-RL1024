import RUNTIME_VERSION from '../data/catalog/runtime-version.json' with {type:'json'};
import { requirePlayableCatalog } from './catalog-support.js';
import PRODUCTION_CATALOG from '../data/catalog/production-sets.json' with {type:'json'};
import PRODUCTION_BATTLE_DATA from '../data/catalog/battle-data.json' with {type:'json'};
import { generateCatalogOffer } from './catalog-offers.js';
import {generateCpuCatalogOffer} from './cpu-offers.js';
import {resolveCpuHandicap} from './cpu-handicap.js';
import {CPU_SPECIAL_BATTLE_DATA} from './cpu-special-sets.js';
import { Battle } from './battle.js';
import { calculateChampionsStats } from './champions-stats.js';
import { productionCombatant } from './battle-production.js';
import { genderForSpeciesInstance } from './pokemon-gender.js';

// Battle-effect support controls room eligibility; M-C reference history is not an admission gate.
export const playableSets=requirePlayableCatalog(PRODUCTION_CATALOG.sets,PRODUCTION_BATTLE_DATA);

export function productionDependencies(seed,{headlessPresentation=false,cpuDifficulty=0}={}) {
  const handicap=resolveCpuHandicap(cpuDifficulty);
  const battleData=handicap.specialLegendaryCandidates?CPU_SPECIAL_BATTLE_DATA:PRODUCTION_BATTLE_DATA;
  const mix=(round,seat=0)=>((seed ^ Math.imul(round,0x9e3779b9) ^ Math.imul(seat+1,0x85ebca6b))>>>0)||1;
  const audits=new Map();
  const provider=({round,seat,carry,credits})=>{
    const params={round,seat,carry,seed:mix(round,seat),credits};
    const result=cpuDifficulty>0&&seat===1
      ?generateCpuCatalogOffer(playableSets,battleData,{...params,difficulty:cpuDifficulty})
      :generateCatalogOffer(playableSets,PRODUCTION_BATTLE_DATA,params);
    audits.set(seat,result.audit);
    return result.fresh.map(set=>({...set,
      gender:genderForSpeciesInstance(battleData.species[set.speciesId],set.instanceId),
      stats:calculateChampionsStats(set.baseStats,set.statPoints,battleData.natures[set.nature]),
      ...(set.mega?{megaPreview:{...battleData.species[set.mega.speciesId],speciesId:set.mega.speciesId,abilityId:set.mega.abilityId,stats:calculateChampionsStats(battleData.species[set.mega.speciesId].baseStats,set.statPoints,PRODUCTION_BATTLE_DATA.natures[set.nature])}}:{})}));
  };
  provider.auditFor=seat=>structuredClone(audits.get(seat));
  return {provider,
    options:{ruleset:'champions-mc-v1',catalogVersion:RUNTIME_VERSION.version,cpuDifficulty,
      battleFactory:(teams,{round})=>new Battle(teams,mix(round,2),{data:battleData,headlessPresentation}),
      battleRestore:saved=>Battle.restore(saved,{data:battleData})},
  };
}
