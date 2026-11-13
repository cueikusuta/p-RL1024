import { resolveForcedSwitch } from './battle-roar.js';

// Pinned Champions inherits Whirlwind's Gen 9 move definition: forceSwitch,
// bypasssub, but no sound or protect flag. Keep these flags separate from Roar.
const WHIRLWIND = Object.freeze({
  id: 'whirlwind',
  category: 'Status',
  flags: Object.freeze({ bypasssub: 1 }),
});

/** Resolve Whirlwind after the action pipeline selects its current foe. */
export function resolveWhirlwind({ id = 'whirlwind', actor, target, opponentTeam, rng, turn = 0,
  ignoreAbility = actor?.abilityId === 'moldbreaker' } = {}) {
  return resolveForcedSwitch({
    id, expectedId: 'whirlwind', descriptor: WHIRLWIND, actor, target, opponentTeam, rng, turn, ignoreAbility,
  });
}
