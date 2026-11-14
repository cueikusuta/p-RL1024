// Candidate hypotheses derived ONLY from public species/observations and the
// published catalog. This module must never read the actual opposing team.
import PUBLIC_BATTLE_DATA from '../../data/catalog/battle-data.json' with {type:'json'};
const present = (value) => typeof value === 'string' && value.length > 0;
const uniqueStrings = (rows) => [...new Set(
  (Array.isArray(rows) ? rows : []).filter(present)
)].sort();

// The same public species/stone dictionary used by Battle's Mega display.
// A catalog Mega set keeps its PRE-Mega speciesNameJa, so that field alone
// cannot identify a publicly displayed Mega name.
export function megaMatchesObservation(set,observations) {
  if(!set?.mega)return false;
  const speciesId=observations?.revealedMegaSpeciesId;
  if(present(speciesId))
    return set.mega.speciesId===speciesId;
  const name=observations?.revealedMegaSpeciesName;
  if(!present(name))return false;
  return [
    set.speciesNameJa,set.displaySpeciesNameJa,
    set.mega.nameJa,set.mega.speciesNameJa,
    PUBLIC_BATTLE_DATA.megaStones?.[set.itemId]?.nameJa,
    PUBLIC_BATTLE_DATA.species?.[set.mega.speciesId]?.name,
  ].includes(name);
}

function publicNameBySpeciesId(sets) {
  const names = new Map();
  for (const set of sets) {
    if (!set.mega && present(set.speciesId) && present(set.speciesNameJa) &&
        !names.has(set.speciesId))
      names.set(set.speciesId, set.speciesNameJa);
  }
  return names;
}

// These are observed events, NOT necessarily initial equipment or an innate
// ability: Trick, Skill Swap, Trace and other effects can change them.
// Only explicitly confirmed original facts may eliminate catalog candidates.
const CONFIRMED_FIELDS = Object.freeze([
  'confirmedOriginalItemId', 'confirmedBaseAbilityId',
  'revealedMegaSpeciesName', 'revealedMegaSpeciesId',
]);
const rows = value => Array.isArray(value) ? value : [];

function observationsFrom(previous = {}, update = {}) {
  const old = previous ?? {}, next = update ?? {};
  // A contradictory "original" fact must not silently rewrite the past.
  // Preserve the first confirmed observation and keep the conflict sticky.
  const conflicts = new Set(rows(old.conflictingConfirmedFields)
    .filter(field => CONFIRMED_FIELDS.includes(field)));
  const confirmed = {};
  for (const field of CONFIRMED_FIELDS) {
    const before = present(old[field]) ? old[field] : null;
    const incoming = present(next[field]) ? next[field] : null;
    if (before !== null && incoming !== null && before !== incoming)
      conflicts.add(field);
    confirmed[field] = before ?? incoming;
  }
  return {
    usedMoves: uniqueStrings([...rows(old.usedMoves), ...rows(next.usedMoves)]),
    observedItemIds: uniqueStrings([
      ...rows(old.observedItemIds), ...rows(next.observedItemIds),
    ]),
    observedAbilityIds: uniqueStrings([
      ...rows(old.observedAbilityIds), ...rows(next.observedAbilityIds),
    ]),
    ...confirmed,
    conflictingConfirmedFields: [...conflicts].sort(),
  };
}

function compute(publicSpeciesName, sets, observations) {
  if (!present(publicSpeciesName) || !Array.isArray(sets))
    throw new TypeError('Expected public species name and catalog array');
  const names = publicNameBySpeciesId(sets);
  const all = sets.filter(set => {
    if (!present(set.setId) || !present(set.speciesNameJa)) return false;
    const publicName = set.mega
      ? names.get(set.speciesId) ?? set.displaySpeciesNameJa ?? set.speciesNameJa
      : set.speciesNameJa;
    return publicName === publicSpeciesName;
  });
  // No catalog hypothesis can simultaneously satisfy incompatible confirmed
  // original facts; preserve uncertainty instead of switching to a new set.
  const contradicted = observations.conflictingConfirmedFields.length > 0;
  // With only a display name, never pick arbitrarily among distinct public
  // Mega forms. One canonical Mega species ID must be uniquely resolvable.
  const nameOnly=observations.revealedMegaSpeciesName!==null
    &&observations.revealedMegaSpeciesId===null;
  const matchingMegaForms=new Set(nameOnly?all
    .filter(set=>megaMatchesObservation(set,observations))
    .map(set=>set.mega?.speciesId??'legacy:'+set.setId):[]);
  const ambiguousMegaName=nameOnly&&matchingMegaForms.size!==1;
  const matched = (contradicted||ambiguousMegaName) ? [] : all.filter(set => {
    if (observations.usedMoves.some(move => !(set.moves ?? []).includes(move)))
      return false;
    if (observations.confirmedOriginalItemId !== null &&
        set.itemId !== observations.confirmedOriginalItemId) return false;
    if (observations.confirmedBaseAbilityId !== null &&
        set.abilityId !== observations.confirmedBaseAbilityId) return false;
    // A public Mega species ID is authoritative. Catalog set names refer to
    // the pre-Mega roster, which may not equal the in-battle Mega display name.
    if ((observations.revealedMegaSpeciesId !== null ||
        observations.revealedMegaSpeciesName !== null) &&
        !megaMatchesObservation(set,observations))return false;
    return true;
  });
  // Uniform within the surviving catalog hypotheses is an UNCALIBRATED prior,
  // not an assertion about real draft offer/selection frequencies.
  const hypotheses = matched.map(set => ({
    setId: set.setId, weight: 1 / matched.length,
  })).sort((a, b) => a.setId.localeCompare(b.setId, 'en'));
  return {
    publicSpeciesName,
    observations,
    hypotheses,
    unknown: hypotheses.length === 0,
    unknownReason: hypotheses.length ? null :
      (contradicted ? 'contradictory-confirmed-observations'
        : all.length ? 'observations-unmatched' : 'species-not-in-catalog'),
    priorModel: 'uncalibrated-uniform-catalog',
  };
}

/** Make a hypothesis distribution from a publicly visible species name.
 *  The caller must supply the published catalog, never an actual enemy set.
 */
export function createOpponentBelief(publicSpeciesName, sets) {
  return compute(publicSpeciesName, sets, observationsFrom());
}

/** Incorporate only explicitly public events. Non-whitelisted fields (including
 *  a private set ID) are deliberately ignored. Does not mutate prior belief.
 */
export function observeOpponentBelief(previous, publicEvents, sets) {
  if (!previous || !present(previous.publicSpeciesName))
    throw new TypeError('Expected prior public opponent belief');
  return compute(previous.publicSpeciesName, sets,
    observationsFrom(previous.observations, publicEvents));
}
