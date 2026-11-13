import { requireRule } from './draft.js';

// Resolve any move whose fixed reference declares selfdestruct: "always".
// The Battle owner applies Explosion's separate winner rule only to Explosion.
export function resolveSelfdestruct({ actor, target, executeAttack, onDamage = () => {}, moveId = null }) {
  requireRule(actor && Number.isInteger(actor.hp) && actor.hp > 0, 'Living selfdestruct user required');
  requireRule(typeof executeAttack === 'function', 'Attack executor required');

  // runMove exits before selfdestruct when there is no selected target.
  if (!target) return { hits: 0, totalDamage: 0, failed: true, noTarget: true, selfdestruct: false };

  const faintHP = actor.hp;
  actor.hp = 0;
  onDamage({ target: actor, damage: faintHP, kind: 'selfdestruct', ...(moveId ? {moveId} : {}) });

  // A selected but already-fainted target yields no targets for the spread
  // hit loop; the always-faint has already happened.
  if (target.hp <= 0) return { hits: 0, totalDamage: 0, failed: true, noTarget: true, selfdestruct: true };

  const result = executeAttack({ allowFaintedSelfdestruct: true, onDamage });
  return { ...result, selfdestruct: true };
}

export const resolveExplosion=resolveSelfdestruct;

export function explosionWinner(living, actorSeat) {
  requireRule(Array.isArray(living) && living.length===2 && living.every(value=>typeof value==='boolean'), 'Invalid living teams');
  requireRule(actorSeat===0 || actorSeat===1, 'Invalid Explosion user seat');
  return !living[0] && !living[1] ? 1-actorSeat : undefined;
}
