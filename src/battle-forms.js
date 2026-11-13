import { requireRule } from './draft.js';
import { calculateChampionsStats } from './champions-stats.js';

export const DYNAMIC_BATTLE_FORMS_BY_ABILITY=Object.freeze({
  forecast:['castformsunny','castformrainy','castformsnowy'],
  stancechange:['aegislashblade'],
  zerotohero:['palafinhero'],
  hungerswitch:['morpekohangry'],
  disguise:['mimikyubusted'],
});

export const DYNAMIC_BATTLE_FORM_NAMES_JA=Object.freeze({
  morpekohangry:'モルペコ（はらぺこもよう）',
  mimikyubusted:'ミミッキュ（ばれたすがた）',
});

export function changeBattleForm(pokemon,formId,data) {
  if(pokemon.speciesId===formId)return false;
  const form=data.battleForms?.[formId]??data.species?.[formId];
  requireRule(form?.baseStats && form.types?.length,'Missing battle form data: '+formId);
  pokemon.speciesId=formId;
  if(form.nameJa)pokemon.speciesName=form.nameJa;
  else if(formId===pokemon.baseSpeciesId)pokemon.speciesName=pokemon.baseSpeciesName??pokemon.speciesName;
  pokemon.stats=calculateChampionsStats(form.baseStats,pokemon.statPoints,data.natures[pokemon.nature]);
  pokemon.types=[...form.types];pokemon.weightkg=form.weightkg;
  return true;
}

export function updateForecast(pokemon,field,data) {
  if(pokemon?.abilityId!=='forecast' || pokemon.baseSpeciesId!=='castform' || !pokemon.hp)return false;
  const weather=field.weatherSuppressed?'':field.weather;
  const formId=weather==='sunnyday'?'castformsunny':weather==='raindance'?'castformrainy':weather==='snowscape'?'castformsnowy':'castform';
  return changeBattleForm(pokemon,formId,data);
}

export function updateHungerSwitch(pokemon,data) {
  if(pokemon?.abilityId!=='hungerswitch'||pokemon.baseSpeciesId!=='morpeko'||!pokemon.hp)return false;
  const formId=pokemon.speciesId==='morpeko'?'morpekohangry':pokemon.speciesId==='morpekohangry'?'morpeko':'';
  return formId ? changeBattleForm(pokemon,formId,data) : false;
}

export function updateDisguise(pokemon,data) {
  if(pokemon?.abilityId!=='disguise' || pokemon.baseSpeciesId!=='mimikyu' ||
    pokemon.speciesId!=='mimikyu' || !pokemon.disguiseBusted)return false;
  return changeBattleForm(pokemon,'mimikyubusted',data);
}

export function updateDynamicForms(active,field,data) {
  for(const pokemon of active)if(pokemon?.hp)updateForecast(pokemon,field,data);
}
