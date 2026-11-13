// AI-facing, read-only capabilities inferred from current set and battle data.
// This module deliberately does NOT choose actions or modify catalog/battle rules.
const CAPABILITIES = Object.freeze([
  'attack', 'setup', 'recovery', 'teamRecovery', 'hazard', 'hazardRemoval', 'phazing',
  'yawn', 'pivot', 'screen', 'batonPass', 'trickRoom', 'tailwind',
  'statusControl', 'weatherStart', 'terrainStart', 'priorityAttack',
]);
export const CAPABILITY_IDS = CAPABILITIES;

const IDS = {
  setup: new Set('swordsdance nastyplot dragondance quiverdance calmmind bulkup shellsmash agility irondefense amnesia coil workup shiftgear honeclaws cosmicpower curse growth bellydrum rockpolish cottonguard minimize'.split(' ')),
  recovery: new Set('recover roost softboiled slackoff synthesis moonlight morningsun wish milkdrink healorder shoreup lifedew rest strengthsap painsplit'.split(' ')),
  teamRecovery: new Set('healingwish revivalblessing'.split(' ')),
  hazard: new Set('stealthrock spikes toxicspikes stickyweb'.split(' ')),
  hazardRemoval: new Set('rapidspin defog mortalspin tidyup courtchange'.split(' ')),
  phazing: new Set('roar whirlwind dragontail circlethrow'.split(' ')),
  pivot: new Set('uturn voltswitch flipturn partingshot chillyreception teleport shedtail'.split(' ')),
  screen: new Set('reflect lightscreen auroraveil'.split(' ')),
  statusControl: new Set('toxic thunderwave willowisp spore sleeppowder glare stunspore yawn taunt encore disable haze clearsmog leechseed hypnosis nuzzle torment'.split(' ')),
};
const WEATHER_MOVES = new Set('sunnyday raindance sandstorm snowscape hail chillyreception'.split(' '));
const TERRAIN_MOVES = new Set('electricterrain grassyterrain psychicterrain mistyterrain'.split(' '));
const WEATHER_ABILITIES = new Set('drought drizzle sandstream sandspit snowwarning'.split(' '));
const TERRAIN_ABILITIES = new Set('electricsurge grassysurge psychicsurge mistysurge'.split(' '));

const positiveBoost = changes => changes && Object.values(changes).some(value => Number(value) > 0);

/**
 * Extract *capabilities*, not the desirability or legal availability of a tactic.
 * A Baton Pass user, for example, needs a useful receiver before a pass plan
 * becomes viable. Evidence records exactly which known field triggered a tag.
 * Mega abilities are explicitly marked conditional and must not be treated as
 * active before Mega Evolution.
 */
export function analyzeSetCapabilities(set, battleData) {
  if (!set || typeof set.setId !== 'string' || !set.setId ||
      !Array.isArray(set.moves) || !battleData?.moves)
    throw new TypeError('Expected current set and battle definitions');
  const evidence = Object.fromEntries(CAPABILITIES.map(key => [key, []]));
  const add = (key, marker) => {
    if (!evidence[key].includes(marker)) evidence[key].push(marker);
  };
  for (const id of set.moves) {
    const move = battleData.moves[id];
    if (!move) throw new Error('Unknown battle move in ' + set.setId + ': ' + id);
    const marker = 'move:' + id;
    if (move.category !== 'Status') {
      add('attack', marker);
      if ((move.priority ?? 0) > 0) add('priorityAttack', marker);
    }
    if (IDS.setup.has(id) || positiveBoost(move.boosts) ||
        positiveBoost(move.self?.boosts)) add('setup', marker);
    if (IDS.teamRecovery.has(id)) add('teamRecovery', marker);
    // heal is also a Showdown move flag for Healing Wish/Revival Blessing:
    // these restore teammates, not the current user's HP.
    if (!IDS.teamRecovery.has(id) &&
        (IDS.recovery.has(id) || move.heal || move.flags?.heal)) add('recovery', marker);
    if (IDS.hazard.has(id) || IDS.hazard.has(move.sideCondition)) add('hazard', marker);
    if (IDS.hazardRemoval.has(id)) add('hazardRemoval', marker);
    if (IDS.phazing.has(id)) add('phazing', marker);
    if (id === 'yawn') add('yawn', marker);
    // Revival Blessing uses selfSwitch to choose a fainted teammate, not to pivot.
    if (IDS.pivot.has(id) || (move.selfSwitch && !IDS.teamRecovery.has(id))) add('pivot', marker);
    if (IDS.screen.has(id) || IDS.screen.has(move.sideCondition)) add('screen', marker);
    if (id === 'batonpass') add('batonPass', marker);
    if (id === 'trickroom' || move.pseudoWeather === 'trickroom') add('trickRoom', marker);
    if (id === 'tailwind' || move.sideCondition === 'tailwind') add('tailwind', marker);
    if (IDS.statusControl.has(id) || (move.status && move.target !== 'self') ||
        move.volatileStatus === 'leechseed') add('statusControl', marker);
    if (WEATHER_MOVES.has(id) || move.weather) add('weatherStart', marker);
    if (TERRAIN_MOVES.has(id) || move.terrain) add('terrainStart', marker);
  }
  const abilityIds = {
    base: set.abilityId ?? null,
    mega: set.mega?.abilityId ?? null,
  };
  for (const [form, id] of Object.entries(abilityIds)) {
    if (!id) continue;
    if (WEATHER_ABILITIES.has(id)) add('weatherStart', 'ability:' + id + '@' + form);
    if (TERRAIN_ABILITIES.has(id)) add('terrainStart', 'ability:' + id + '@' + form);
  }
  const tags = CAPABILITIES.filter(key => evidence[key].length > 0);
  for (const values of Object.values(evidence)) Object.freeze(values);
  return Object.freeze({
    setId: set.setId,
    itemId: set.itemId ?? null,
    abilities: Object.freeze(abilityIds),
    tags: Object.freeze(tags),
    evidence: Object.freeze(evidence),
  });
}

/** Reproducible, catalog-size-independent audit data; no writes or side effects. */
export function summarizeCatalogCapabilities(sets, battleData) {
  if (!Array.isArray(sets)) throw new TypeError('Expected catalog array');
  const bySet = Object.create(null);
  const tagCounts = Object.fromEntries(CAPABILITIES.map(id => [id, 0]));
  const untaggedSetIds = [];
  for (const set of sets) {
    if (!set?.setId || Object.hasOwn(bySet, set.setId))
      throw new Error('Missing or duplicate setId: ' + (set?.setId ?? '<missing>'));
    const capabilities = analyzeSetCapabilities(set, battleData);
    bySet[set.setId] = capabilities;
    if (!capabilities.tags.length) untaggedSetIds.push(set.setId);
    for (const tag of capabilities.tags) tagCounts[tag]++;
  }
  return {total: sets.length, tagCounts, untaggedSetIds, bySet};
}
