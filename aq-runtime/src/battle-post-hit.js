import { transferHeldItem } from './battle-mechanics.js';
import { rapidSpinField, placeHazard, updateMimicry } from './battle-field.js';

// Mutate only already-confirmed post-hit effects. The caller preserves the
// original condition-notification boundary before and after this function.
export function applyConfirmedPostHitEffects({id,effective,event,actor,target,field,seat,getActive}) {
  // Body onAfterHit follows DamagingHit; onAfterSubDamage additionally
  // requires a surviving source for field cleanup.
  if (id === 'rapidspin' && !effective.hasSheerForce && (actor.hp || !event.substitute)) rapidSpinField(field, seat, actor);
  if (id === 'mortalspin' && !effective.hasSheerForce && (actor.hp || !event.substitute)) {
    delete actor.volatiles.leechseed;
    delete actor.volatiles.partiallytrapped;
    for (const hazard of ['stealthrock','spikes','toxicspikes','stickyweb']) delete field.sides[seat][hazard];
  }
  if (['ceaselessedge','stoneaxe'].includes(id) && !effective.hasSheerForce && (actor.hp || !event.substitute))
    placeHazard(field, 1-seat, id === 'stoneaxe' ? 'stealthrock' : 'spikes');
  if (id === 'icespinner' && (actor.hp || !event.substitute)) {
    field.terrain = '';
    field.terrainTurns = 0;
    for (const pokemon of [getActive(0), getActive(1)]) updateMimicry(pokemon, '');
  }
  if (id === 'thief' && !event.substitute && actor.hp && !actor.itemId && target.itemId &&
    target.itemId !== target.megaStoneId && (!target.hp || actor.abilityId === 'moldbreaker' || target.abilityId !== 'stickyhold')) {
    transferHeldItem(target, actor);
  }
}
