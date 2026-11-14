import { requireRule } from './draft.js';
import { notifyEffectEvent } from './battle-messages.js';

// Fixed-point half-down rounding used by the recorded Champions reference.
export function modify(value, numerator, denominator = 4096) {
  requireRule(Number.isInteger(value) && value >= 0 && Number.isInteger(numerator) && numerator >= 0 && Number.isInteger(denominator) && denominator > 0, 'Invalid modifier');
  return Math.floor((value * numerator + Math.ceil(denominator / 2) - 1) / denominator);
}

export function effectiveness(type, defenderTypes, chart) {
  if (type === '???') return 1;
  requireRule(Array.isArray(defenderTypes) && defenderTypes.length > 0, 'Missing defender types');
  let factor = 1;
  for (const defender of defenderTypes) {
    // Double Shock changes its user's type to ??? after a successful hit.
    // That type has no offensive or defensive chart interactions, so it is
    // neutral against every attacking type.
    if (defender === '???') continue;
    const row = chart[defender.toLowerCase()];
    requireRule(row && Object.hasOwn(row, type), 'Unknown battle type');
    factor *= [1, 2, 0.5, 0][row[type]];
  }
  return factor;
}

export function movePP(basePP, noPPBoosts = false) {
  requireRule(Number.isInteger(basePP) && basePP > 0, 'Invalid base PP');
  const pp = Math.min(20, basePP);
  return noPPBoosts ? pp : (pp / 5 + 1) * 4;
}
const heldItemObservers=new WeakMap();
const pendingItemEvents=new WeakMap();
export function observeHeldItemEvents(pokemon,observer) {
  if(typeof observer==='function')heldItemObservers.set(pokemon,observer);
  else heldItemObservers.delete(pokemon);
}
export function announceHeldItemEvent(pokemon,event) {
  const pending=pendingItemEvents.get(pokemon);
  if(pending)pending.push(()=>heldItemObservers.get(pokemon)?.(event));
  else heldItemObservers.get(pokemon)?.(event);
}
// Compute HP/items in the existing order; only publish their text after the
// owning hit has been announced. The queue never enters a checkpoint or view.
export function captureItemMessages(pokemon,apply) {
  const pending=[];pendingItemEvents.set(pokemon,pending);
  let value;try {value=apply();} finally {pendingItemEvents.delete(pokemon);}
  let flushed=false;
  return {value,flush:()=>{if(flushed)return;flushed=true;for(const notify of pending)notify();}};
}
export function consumeHeldItem(pokemon,{kind='activated',other=null,notify=true}={}) {
  const item=pokemon.itemId;
  if(!item)return '';
  pokemon.itemId='';
  // Removing or transferring a held item does not erase a previously consumed item.
  // Air Balloon popping also does not replace that Recycle history (fixed Showdown).
  if(kind!=='removed'&&kind!=='transfer'&&item!=='airballoon')pokemon.lastItemId=item;
  markUnburden(pokemon);
  if(notify)announceHeldItemEvent(pokemon,{item,kind,other});
  return item;
}
const SUPPORTED_BERRIES=new Set(['chartiberry','chestoberry','chopleberry','cobaberry','habanberry','kasibberry','lumberry','occaberry','passhoberry','roseliberry','shucaberry','sitrusberry','wacanberry','yacheberry']);
export function afterBerryEaten(pokemon,itemId) {
  if(!SUPPORTED_BERRIES.has(itemId))return false;
  pokemon.lastBerryId=itemId;
  if(pokemon.abilityId!=='cheekpouch'||pokemon.hp<=0||pokemon.volatiles?.healblock)return false;
  const restored=Math.max(0,Math.min(pokemon.maxHP-pokemon.hp,Math.floor(pokemon.maxHP/3)));
  pokemon.hp+=restored;
  if(restored>0) {
    const notify=()=>notifyEffectEvent(pokemon,{kind:'heal',ability:'cheekpouch'});
    const pending=pendingItemEvents.get(pokemon);if(pending)pending.push(notify);else notify();
  }
  return restored>0;
}
export function markUnburden(pokemon) {if(pokemon.abilityId==='unburden'){pokemon.volatiles??={};pokemon.volatiles.unburden=true;}}
export function isFormLockedHeldItem(pokemon) {
  return Boolean(pokemon?.itemId && (
    pokemon?.abilityId==='multitype' && /plate$/.test(pokemon.itemId) ||
    pokemon?.speciesId==='zaciancrowned' && pokemon.itemId==='rustedsword'));
}
export function transferHeldItem(from,to) {
  if(!from?.itemId || to?.itemId || from.itemId===from.megaStoneId||isFormLockedHeldItem(from))return '';
  // A transfer is not publicly confirmed until both inventories are updated.
  // Keep the same observer/event and action order, but never let a subscriber
  // observe the sender empty while the receiver still lacks the item.
  const item=consumeHeldItem(from,{kind:'transfer',other:to,notify:false});
  to.itemId=item;
  announceHeldItemEvent(from,{item,kind:'transfer',other:to});
  return item;
}

export function elementalDamage({ attack, defense, power, roll, crit = false, stab = false, effectiveness: factor = 1,
  baseModifier = 4096, weatherModifier = 4096, burned = false, finalModifier = 4096 }) {
  requireRule(Number.isInteger(roll) && roll >= 0 && roll < 16 && attack > 0 && defense > 0 && power > 0, 'Invalid damage parameters');
  requireRule([0,0.25,0.5,1,2,4].includes(factor), 'Invalid effectiveness');
  if (factor === 0) return 0;
  let damage = Math.floor(Math.floor(22 * power * attack / defense) / 50) + 2;
  damage = modify(damage, baseModifier);
  damage = modify(damage, weatherModifier);
  if (crit) damage = Math.floor(damage * 1.5);
  damage = Math.floor(damage * (100 - roll) / 100);
  if (stab) damage = modify(damage, typeof stab==='number' ? stab : 6144);
  damage = Math.floor(damage * factor);
  if (burned) damage = modify(damage, 2048);
  damage = modify(damage, finalModifier);
  return Math.max(1, damage) % 65536;
}

// These functions mutate server-owned combatants, never immutable catalog sets.
export function consumeHealingBerry(pokemon) {
  if(pokemon.volatiles?.berserkItemPending)return false;
  if (pokemon.itemId !== 'sitrusberry' || pokemon.volatiles?.unnerved || pokemon.hp <= 0 || pokemon.hp > pokemon.maxHP / 2 || pokemon.volatiles?.healblock) return false;
  const before=pokemon.hp;
  const item=consumeHeldItem(pokemon,{kind:'activated',notify:false});
  pokemon.hp = Math.min(pokemon.maxHP, pokemon.hp + Math.max(1, Math.floor(pokemon.maxHP / 4)));
  announceHeldItemEvent(pokemon,{item,kind:'healed',amount:pokemon.hp-before});
  afterBerryEaten(pokemon,item);
  return true;
}
export function applyMoveDamage(pokemon, damage, {ignoreAbility=false}={}) {
  requireRule(Number.isInteger(damage) && damage >= 0, 'Invalid move damage');
  if (pokemon.hp<=0) return 0;
  if (!ignoreAbility && pokemon.hp===pokemon.maxHP && damage>=pokemon.hp && pokemon.abilityId==='sturdy') {
    damage=pokemon.hp-1;
  }
  if (pokemon.hp === pokemon.maxHP && damage >= pokemon.hp && pokemon.itemId === 'focussash') {
      consumeHeldItem(pokemon,{kind:'activated'}); damage = pokemon.hp - 1;
  }
  const lost = Math.min(pokemon.hp, damage);
  pokemon.hp -= lost;
  consumeHealingBerry(pokemon);
  return lost;
}
export function residualItem(pokemon) {
  if (pokemon.hp <= 0 || pokemon.itemId !== 'leftovers' || pokemon.volatiles?.healblock) return 0;
  const restored = Math.min(pokemon.maxHP - pokemon.hp, Math.max(1, Math.floor(pokemon.maxHP / 16)));
  pokemon.hp += restored;
  return restored;
}
export function battleSpeed(pokemon) {
  return pokemon.itemId === 'choicescarf' ? modify(pokemon.stats.spe, 6144) : pokemon.stats.spe;
}
