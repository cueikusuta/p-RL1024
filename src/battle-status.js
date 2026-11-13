import { requireRule } from './draft.js';
import { elementalDamage, consumeHeldItem, consumeHealingBerry, afterBerryEaten } from './battle-mechanics.js';
import { changeBoosts } from './battle-moves.js';
import { notifyEffectEvent, endCondition } from './battle-messages.js';

const STATUSES = ['brn','par','slp','frz','psn','tox'];
export function clearStatus(pokemon) {
  const old = pokemon.status;
  pokemon.status = ''; pokemon.statusState = {};
  return Boolean(old);
}
export function statusBerry(pokemon) {
  if (pokemon.hp <= 0) return false;
  if (pokemon.itemId === 'lumberry' && !pokemon.volatiles?.unnerved && (pokemon.status || pokemon.volatiles?.confusion)) {
    const item=consumeHeldItem(pokemon,{kind:'activated'});clearStatus(pokemon);afterBerryEaten(pokemon,item);
    endCondition(pokemon,'confusion');
    return true;
  }
  if (pokemon.itemId === 'chestoberry' && !pokemon.volatiles?.unnerved && pokemon.status === 'slp') {
    const item=consumeHeldItem(pokemon,{kind:'activated'});clearStatus(pokemon);afterBerryEaten(pokemon,item);return true;
  }
  return false;
}
export function canSetStatus(pokemon, status, { chart, ignoreAbility = false, sunny = false, electricTerrain = false, mistyTerrain = false, grounded = true, source = null } = {}) {
  requireRule(STATUSES.includes(status), 'Unknown status');
  if (pokemon.hp <= 0 || pokemon.status) return false;
  // Ally-prefixed Flower Veil handlers include the holder itself. Self-caused
  // status (Rest/orbs) and the later sleep from an existing Yawn are exempt.
  if (!ignoreAbility && pokemon.abilityId==='flowerveil' && pokemon.types.includes('Grass') && source && source!==pokemon) return false;
  const immunity = status === 'tox' ? 'psn' : status;
  if (pokemon.types.some(type => chart?.[type.toLowerCase()]?.[immunity] === 3 && !(immunity==='psn' && ['Steel','Poison'].includes(type) && source?.abilityId==='corrosion'))) return false;
  if (status === 'frz' && sunny || status === 'slp' && electricTerrain && grounded || mistyTerrain && grounded) return false;
  if (!ignoreAbility && (status === 'slp' && ['insomnia','vitalspirit','sweetveil'].includes(pokemon.abilityId) || status === 'par' && pokemon.abilityId === 'limber' || status === 'brn' && ['thermalexchange','waterbubble'].includes(pokemon.abilityId) || ['psn','tox'].includes(status) && pokemon.abilityId==='immunity' || pokemon.abilityId==='purifyingsalt' || ['brn','par','slp','frz','psn','tox'].includes(status) && pokemon.abilityId==='leafguard' && sunny)) return false;
  return true;
}
export function setStatus(pokemon, status, rng, options = {}) {
  if (!canSetStatus(pokemon, status, options)) return false;
  pokemon.status = status;
  pokemon.statusState = status === 'slp' ? {time: [2,3,3][rng.integer(3)]} : status === 'frz' ? {time:3} : status === 'tox' ? {stage:0} : {};
  if(pokemon.abilityId==='synchronize' && options.source && options.source!==pokemon && ['brn','par','psn','tox'].includes(status) && options.sourceEffectId!=='toxicspikes' && !options.synchronized) {
    const reflected=setStatus(options.source,status,rng,{...options,source:pokemon,ignoreAbility:false,synchronized:true});
    // Report only a confirmed reflected status that survived an immediate berry.
    // The Battle observer attributes the event without guessing from final states.
    if(reflected && options.source.status===status)
      notifyEffectEvent(pokemon,{kind:'status-reflect',ability:'synchronize',target:options.source,status});
  }
  statusBerry(pokemon);
  return true;
}
export function addConfusion(pokemon, rng, {ignoreAbility=false,causeMoveId=null}={}) {
  if (pokemon.hp <= 0 || !ignoreAbility && pokemon.abilityId==='owntempo' || pokemon.volatiles?.confusion) return false;
  pokemon.volatiles ??= {};
  pokemon.volatiles.confusion = {time:2 + rng.integer(4)};
  notifyEffectEvent(pokemon,{id:'confusion',...(causeMoveId?{causeMoveId}:{})});
  statusBerry(pokemon);
  return true;
}
export function addFlinch(pokemon, {ignoreAbility = false} = {}) {
  if (pokemon.hp <= 0 || !ignoreAbility && pokemon.abilityId === 'innerfocus') return false;
  pokemon.volatiles ??= {}; pokemon.volatiles.flinch = true;
  return true;
}

// Called before PP deduction. Every counter and random draw is server-owned.
// Result exposes an event to the engine; it must log damage/faint and check victory.
export function beforeMoveStatus(pokemon, move, rng, {attack = pokemon.stats.atk, defense = pokemon.stats.def, restriction = () => '', onStatusEnd = () => {}} = {}) {
  if(pokemon.volatiles?.glaiverush)delete pokemon.volatiles.glaiverush;
  statusBerry(pokemon);
  if(pokemon.volatiles?.recharge){delete pokemon.volatiles.recharge;return {canMove:false,reason:'recharge'};}
  const state = pokemon.statusState ??= {};
  if (pokemon.status === 'slp') {
    state.time -= pokemon.abilityId === 'earlybird' ? 2 : 1;
    if (state.time <= 0 && clearStatus(pokemon)) onStatusEnd('slp');
    else if (!move.sleepUsable) return {canMove:false, reason:'slp'};
  }
  if (pokemon.status === 'frz') {
    if (move.flags?.defrost) { if (clearStatus(pokemon)) onStatusEnd('frz'); }
    else {
      state.time--;
      if (state.time <= 0 || rng.integer(4) === 0) { if (clearStatus(pokemon)) onStatusEnd('frz'); }
      else return {canMove:false, reason:'frz'};
    }
  }
  if (pokemon.volatiles?.flinch) {
    delete pokemon.volatiles.flinch;
      // Flinch dispatches once when it prevents the action, after sleep/freeze,
      // rather than each time a multi-hit attack applies the volatile.
      if(pokemon.abilityId==='steadfast')changeBoosts(pokemon,{spe:1},{effect:'steadfast'});
    return {canMove:false, reason:'flinch'};
  }
  const blocked=restriction();
  if (blocked)return {canMove:false,reason:blocked};
  const confusion = pokemon.volatiles?.confusion;
  if (confusion) {
    if (--confusion.time <= 0) endCondition(pokemon,'confusion');
    else if (rng.integer(100) < 33) {
      const beforeHP=pokemon.hp;
      const damage = Math.min(pokemon.hp, elementalDamage({attack, defense, power:40, roll:rng.integer(16)}));
      pokemon.hp -= damage;
      // Held berries update after self-inflicted damage, before the next action.
      consumeHealingBerry(pokemon);
      return {canMove:false, reason:'confusion', damage, beforeHP};
    }
  }
  if(pokemon.volatiles?.attract && rng.integer(2)===0)return {canMove:false,reason:'attract'};
  if (pokemon.status === 'par' && rng.integer(8) === 0) return {canMove:false, reason:'par'};
  return {canMove:true};
}
export function statusSpeed(speed, pokemon) {
  return pokemon.status === 'par' ? Math.floor(speed / 2) : speed;
}
export function residualStatus(pokemon) {
  if (pokemon.hp <= 0) return 0;
  let amount = 0;
  if (pokemon.status === 'brn') amount = Math.max(1, Math.floor(pokemon.maxHP / 16));
  if (pokemon.status === 'psn') amount = Math.max(1, Math.floor(pokemon.maxHP / 8));
  if (pokemon.status === 'tox') {
    pokemon.statusState ??= {stage:0};
    pokemon.statusState.stage = Math.min(15, (pokemon.statusState.stage ?? 0) + 1);
    amount = Math.floor(Math.max(1, pokemon.maxHP / 16) * pokemon.statusState.stage);
  }
  if(pokemon.abilityId==='magicguard')return 0;
  if(pokemon.status==='brn' && pokemon.abilityId==='heatproof')amount=Math.max(1,Math.floor(amount/2));
  const lost = Math.min(pokemon.hp, amount);
  pokemon.hp -= lost;
  return lost;
}
export function switchOutStatus(pokemon) {
  if (pokemon.abilityId === 'naturalcure' && pokemon.hp > 0) clearStatus(pokemon);
  pokemon.volatiles = {};
  if (pokemon.status === 'tox') pokemon.statusState.stage = 0;
}
export function thawFromHit(pokemon, move) {
  return pokemon.status === 'frz' && (move.thawsTarget || move.type === 'Fire' && move.category !== 'Status') ? clearStatus(pokemon) : false;
}

export function rest(pokemon, rng, options = {}) {
  if (pokemon.hp <= 0 || pokemon.hp === pokemon.maxHP || pokemon.status === 'slp' || pokemon.volatiles?.healblock) return false;
  // Rest replaces an existing non-sleep status. Do not erase it if sleep is blocked.
  if (!canSetStatus({...pokemon,status:''}, 'slp', {...options,source:null})) return false;
  clearStatus(pokemon);
  setStatus(pokemon,'slp',rng,options);
  if (pokemon.status === 'slp') pokemon.statusState.time = 3;
  pokemon.hp = pokemon.maxHP;
  return true;
}
