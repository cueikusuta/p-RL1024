// Same deterministic gender for drafts, combatants, saves and reconnects.
// Gender data is authored by the pinned Pokémon Showdown species reference.
export function genderForSpeciesInstance(species,instanceId) {
  if(['M','F','N'].includes(species?.gender))return species.gender;
  const femaleRatio=species?.genderRatio?.F ?? 0.5;
  if(femaleRatio===0)return 'M';
  if(femaleRatio===1)return 'F';
  const hash=[...String(instanceId)].reduce(
    (value,char)=>Math.imul(value^char.charCodeAt(0),16777619)>>>0,2166136261);
  return hash/4294967296 < femaleRatio?'F':'M';
}
