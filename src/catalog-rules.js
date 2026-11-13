// This exception describes the approved fixed Ditto build, not arbitrary short sets.
export function isImposterDittoSet(set) {
  return set.speciesId === 'ditto' && !set.mega && set.abilityId === 'imposter'
    && Array.isArray(set.moves) && set.moves.length === 1 && set.moves[0] === 'transform'
    && ['hp','atk','def','spa','spd','spe'].every(stat => set.statPoints?.[stat] === (stat === 'hp' ? 32 : 0));
}

// Ditto's Champions learnset contains only Transform, so its manual Transform
// build is the one tightly scoped second exception to the four-move catalog rule.
export function isManualTransformDittoSet(set) {
  return set.speciesId === 'ditto' && !set.mega && set.abilityId === 'limber'
    && Array.isArray(set.moves) && set.moves.length === 1 && set.moves[0] === 'transform'
    && ['hp','atk','def','spa','spd','spe'].every(stat => Number.isInteger(set.statPoints?.[stat])
      && set.statPoints[stat] >= 0 && set.statPoints[stat] <= 32)
    && ['hp','atk','def','spa','spd','spe'].reduce((total, stat) => total + set.statPoints[stat], 0) === 66;
}

export function isSingleMoveDittoSet(set) {
  return isImposterDittoSet(set) || isManualTransformDittoSet(set);
}

export function catalogSummary(sets) {
  return {
    speciesCount: new Set(sets.map(set => set.speciesId)).size,
    speciesClauseGroupCount: new Set(sets.map(set => set.speciesClauseId)).size,
    setCount: sets.length,
    megaCount: sets.filter(set => set.mega).length,
    costCounts: Object.fromEntries(Array.from({length: 7}, (_, index) => [index + 1, sets.filter(set => set.cost === index + 1).length])),
  };
}
