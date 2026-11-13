import { requireRule } from './draft.js';
import { afterAttackItems, resolveAttack } from './battle-effects.js';

/**
 * Resolve Dragon Tail's hit using the common damaging-move pipeline, then
 * decide whether it can drag the current target out.  The battle owner still
 * performs the switch and entry effects.
 */
export function resolveDragonTail({
  id = 'dragontail', definition, actor, target, opponentTeam, chart, turn, rng,
  executeAttack = resolveAttack, onDamage = () => {}, onAfterMoveSecondary=()=>{}, ...attackOptions
} = {}) {
  if (id !== 'dragontail') return { success: false, reason: 'not-dragon-tail' };
  requireRule(typeof executeAttack === 'function', 'Attack resolver required');
  requireRule(Array.isArray(opponentTeam), 'Opponent team required');

  // Upstream performs the force-switch check inside spreadMoveHit, before
  // Life Orb/Shell Bell's after-move item event. Defer those two item effects
  // so the eligibility snapshot sees the same pre-item HP state.
  let afterMove;
  const attack = executeAttack({
    ...attackOptions, id, definition, actor, target, chart, turn, rng, onDamage,
    deferAfterAttackItems: true,
    onAfterMoveSecondary:(move,result)=>{afterMove=onAfterMoveSecondary(move,result);return afterMove;},
  });
  if (!(attack?.hits > 0)) {
    const reason = attack?.immune ? 'immune' : attack?.missed ? 'missed' :
      attack?.blocked === 'protect' ? 'protected' : 'move-did-not-hit';
    return { ...attack, success: false, forcedSwitch: false, reason };
  }
  let reason = 'eligible';
  let reserves = opponentTeam.filter(pokemon => pokemon !== target && pokemon.hp > 0);
  if (attack.substituteHits > 0) reason = 'hit-substitute';
  else if (!(actor?.hp > 0)) reason = 'actor-fainted';
  else if (!(target?.hp > 0)) reason = 'target-fainted';
  // DragOut is checked after hit effects. These pinned Gen 9 handlers stop
  // forced movement while leaving Dragon Tail's already-dealt damage intact.
  else if (target.volatiles?.ingrain || ['suctioncups', 'guarddog'].includes(target.abilityId)) reason = 'drag-out-blocked';
  else if (!reserves.length) reason = 'no-living-reserve';

  const forcedSwitch = reason === 'eligible';
  // The replacement draw occurs later, during dragIn after the action ends.
  // Apply deferred after-move item effects first, without undoing a force-switch
  // flag that was already set while the user was alive.
  const item = afterAttackItems(definition, actor, attack.totalDamage,{forcedSwitch:Boolean(afterMove?.actorForcedSwitch)});
  if (item < 0) onDamage({ target: actor, damage: -item, kind: 'item' });
  if (forcedSwitch) {
    requireRule(rng && typeof rng.integer === 'function', 'Seeded battle RNG required');
    reserves = opponentTeam.filter(pokemon => pokemon !== target && pokemon.hp > 0);
    if (!reserves.length) return { ...attack, success: true, forcedSwitch: true, reason, outgoing: target };
    const draw = rng.integer(reserves.length);
    requireRule(Number.isInteger(draw) && draw >= 0 && draw < reserves.length, 'Invalid random reserve index');
    return { ...attack, success: true, forcedSwitch: true, reason, outgoing: target, incoming: reserves[draw] };
  }
  return { ...attack, success: true, forcedSwitch: false, reason };
}
