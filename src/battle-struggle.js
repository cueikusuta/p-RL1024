import { afterAttackItems, resolveAttack } from './battle-effects.js';

// The caller owns PP and turn eligibility. This definition covers only the
// damaging move semantics shared with the production attack resolver.
export const STRUGGLE_DEFINITION = Object.freeze({
  name: 'わるあがき',
  type: '???',
  category: 'Physical',
  basePower: 50,
  accuracy: true,
  pp: 1,
  noPPBoosts: true,
  priority: 0,
  target: 'foe',
  flags: Object.freeze({ contact: 1, protect: 1 }),
  struggleRecoil: true,
});

/** Resolve production Struggle, including its direct max-HP recoil. */
export function resolveStruggle({ actor, target, chart, turn, rng, onDamage = () => {}, onAfterHit = () => {}, onAfterMoveSecondary=()=>{}, ...options }) {
  const result = resolveAttack({
    ...options,
    id: 'struggle',
    definition: STRUGGLE_DEFINITION,
    actor,
    target,
    chart,
    turn,
    rng,
    onDamage,
    onAfterHit,
    // Keep held-item identity available throughout damage and contact hooks.
    deferAfterAttackItems: true,
    deferAfterMoveSecondary: true,
  });

  // Champions applies Struggle recoil directly, so Rock Head, damage-based
  // recoil modifiers, and healing-item update events do not affect it.
  let recoilDamage = 0;
  if (result.totalDamage > 0 && actor.hp > 0) {
    recoilDamage = Math.min(actor.hp, Math.max(1, Math.round(actor.maxHP / 4)));
    actor.hp -= recoilDamage;
    if (recoilDamage) onDamage({ target: actor, damage: recoilDamage, kind: 'recoil', moveId:'struggle' });
  }

  const afterMove=onAfterMoveSecondary(STRUGGLE_DEFINITION,result);
  const itemDamage = afterAttackItems({ ...STRUGGLE_DEFINITION, id: 'struggle' }, actor, result.totalDamage,{forcedSwitch:Boolean(afterMove?.actorForcedSwitch)});
  if (itemDamage < 0) onDamage({ target: actor, damage: -itemDamage, kind: 'item', itemId:'lifeorb' });
  return { ...result, recoilDamage };
}
