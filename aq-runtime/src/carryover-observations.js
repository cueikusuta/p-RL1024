// Server-only snapshot of previously disclosed facts for survivors. The
// internal instance ID is a lookup key, never a disclosure or public ID.
// Matching by public species is permitted only when unambiguous in the team.
export function mergeCarryoverObservations(rows,team,publicRows,previousMoves,previousItems,previousAbilities) {
  const speciesCounts=new Map();
  for(const member of team) {
    const name=member.publicSpeciesName??member.speciesName??member.speciesId;
    speciesCounts.set(name,(speciesCounts.get(name)??0)+1);
  }
  const byInstance={},itemsByInstance={},abilitiesByInstance={};
  for(const member of rows) {
    const name=member.publicSpeciesName??member.speciesName??member.speciesId;
    const previous=previousMoves[member.instanceId]??[];
    const identified=speciesCounts.get(name)===1
      ?publicRows.filter(row=>row.rosterSpeciesName===name):[];
    const newlyObserved=identified.flatMap(row=>Array.isArray(row.usedMoves)?row.usedMoves:[]);
    byInstance[member.instanceId]=[...new Set([...previous,...newlyObserved].filter(move=>typeof move==='string'))];
    const earlierItems=previousItems[member.instanceId]??[];
    const newlyObservedItems=identified.flatMap(row=>Array.isArray(row.observedItemIds)?row.observedItemIds:[]);
    itemsByInstance[member.instanceId]=[...new Set([...earlierItems,...newlyObservedItems].filter(id=>typeof id==='string'))];
    const earlierAbilities=previousAbilities[member.instanceId]??[];
    const newlyObservedAbilities=identified.flatMap(row=>Array.isArray(row.observedAbilityIds)?row.observedAbilityIds:[]);
    abilitiesByInstance[member.instanceId]=[...new Set([...earlierAbilities,...newlyObservedAbilities].filter(id=>typeof id==='string'))];
  }
  return {byInstance,itemsByInstance,abilitiesByInstance};
}


// Convert internal carryover lookup maps into the established client-facing
// arrays. No private instance ID or unconfirmed current loadout is serialized.
export function projectCarryoverPublicView(carry,usedMovesById,itemsById,abilitiesById) {
  const carryoverSpecies=carry.map(member=>
    member.publicSpeciesName??(member.mega?member.speciesId:member.speciesName??member.speciesId));
  const carryoverUsedMoves=carry.map(member=>[...(usedMovesById[member.instanceId]??[])]);
  const carryoverObservedItemIds=carry.map(member=>[...(itemsById[member.instanceId]??[])]);
  const carryoverObservedAbilityIds=carry.map(member=>[...(abilitiesById[member.instanceId]??[])]);
  return {
    carryoverSpecies,carryoverUsedMoves,
    ...(carryoverObservedItemIds.some(items=>items.length)?{carryoverObservedItemIds}:{}),
    ...(carryoverObservedAbilityIds.some(abilities=>abilities.length)?{carryoverObservedAbilityIds}:{}),
  };
}


// Internal storage groups each confirmed fact set by its private instance ID.
// Keep the legacy checkpoint fields as an adapter; preserving key presence
// also preserves old snapshots where only one of the three histories exists.
export function packCarryoverKnowledge(usedMoves=[{},{}],observedItemIds=[{},{}],observedAbilityIds=[{},{}]) {
  return [0,1].map(seat=>{
    const moves=usedMoves[seat]??{},items=observedItemIds[seat]??{},abilities=observedAbilityIds[seat]??{};
    const records={};
    const ids=new Set([...Object.keys(moves),...Object.keys(items),...Object.keys(abilities)]);
    for(const id of ids) {
      const record={};
      if(Object.hasOwn(moves,id))record.usedMoves=structuredClone(moves[id]);
      if(Object.hasOwn(items,id))record.observedItemIds=structuredClone(items[id]);
      if(Object.hasOwn(abilities,id))record.observedAbilityIds=structuredClone(abilities[id]);
      records[id]=record;
    }
    return records;
  });
}

export function unpackCarryoverKnowledge(knowledge) {
  const usedMoves=[{},{}],observedItemIds=[{},{}],observedAbilityIds=[{},{}];
  for(const seat of [0,1])for(const [id,record] of Object.entries(knowledge?.[seat]??{})) {
    if(Object.hasOwn(record,'usedMoves'))usedMoves[seat][id]=structuredClone(record.usedMoves);
    if(Object.hasOwn(record,'observedItemIds'))observedItemIds[seat][id]=structuredClone(record.observedItemIds);
    if(Object.hasOwn(record,'observedAbilityIds'))observedAbilityIds[seat][id]=structuredClone(record.observedAbilityIds);
  }
  return {usedMoves,observedItemIds,observedAbilityIds};
}
