import { requireRule } from './draft.js';
import { blocksStatusMove } from './battle-hit.js';

// Roar's pinned Gen 9 definition is a status move with priority -6,
// forceSwitch, sound, and bypasssub; it has no `protect` flag.
const ROAR = Object.freeze({
  id: 'roar',
  category: 'Status',
  flags: Object.freeze({ sound: 1, bypasssub: 1 }),
});

export function resolveForcedSwitch({ id, expectedId, descriptor, actor, target, opponentTeam, rng, turn,
  ignoreAbility }) {
  if (id !== expectedId) return { success: false, reason: `not-${expectedId}` };
  if (!(actor?.hp > 0)) return { success: false, reason: 'actor-fainted' };
  if (!(target?.hp > 0)) return { success: false, reason: 'target-fainted' };

  // These TryHit abilities precede the Prankster and later DragOut checks.
  if (!ignoreAbility && descriptor.flags?.sound && target.abilityId==='soundproof')return {success:false,reason:'soundproof'};
  if (!ignoreAbility && target.abilityId === 'goodasgold') {
    return { success: false, reason: 'good-as-gold' };
  }

  if (blocksStatusMove(descriptor, actor, target, turn)) {
    return { success: false, reason: 'blocked' };
  }
  if(actor.abilityId==='prankster' && target.types?.includes('Dark'))return {success:false,reason:'prankster-dark'};
  if(target.volatiles?.ingrain || !ignoreAbility && ['suctioncups','guarddog'].includes(target.abilityId))return {success:false,reason:'drag-out-blocked'};

  requireRule(Array.isArray(opponentTeam), 'Opponent team required');
  const reserves = opponentTeam.filter(pokemon => pokemon !== target && pokemon.hp > 0);
  if (!reserves.length) return { success: false, reason: 'no-living-reserve' };
  requireRule(rng && typeof rng.integer === 'function', 'Seeded battle RNG required');
  const draw = rng.integer(reserves.length);
  requireRule(Number.isInteger(draw) && draw >= 0 && draw < reserves.length, 'Invalid random reserve index');

  return {
    success: true,
    forcedSwitch: true,
    reason: 'eligible',
    outgoing: target,
    incoming: reserves[draw],
  };
}

/** Resolve Roar after the normal action pipeline has selected its current foe. */
export function resolveRoar({ id = 'roar', actor, target, opponentTeam, rng, turn = 0,
  ignoreAbility = actor?.abilityId === 'moldbreaker' } = {}) {
  return resolveForcedSwitch({
    id, expectedId: 'roar', descriptor: ROAR, actor, target, opponentTeam, rng, turn, ignoreAbility,
  });
}
