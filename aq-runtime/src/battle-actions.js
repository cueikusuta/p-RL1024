import { requireRule } from './draft.js';
import { changeBoosts } from './battle-moves.js';
import { consumeHeldItem } from './battle-mechanics.js';
import { notifyEffectEvent, endCondition } from './battle-messages.js';

const CHOICE_ITEMS=['choicescarf'];
const HERB_CONDITIONS=['attract','taunt','encore','torment','disable','healblock'];
export function mentalHerb(pokemon) {
  if (!pokemon.hp || pokemon.itemId!=='mentalherb' || !HERB_CONDITIONS.some(id=>pokemon.volatiles?.[id]))return false;
  consumeHeldItem(pokemon);
  for(const id of HERB_CONDITIONS)endCondition(pokemon,id);
  return true;
}
export function moveRestriction(pokemon,id,definition,{selecting=false}={}) {
  if (id==='struggle')return '';
  const v=pokemon.volatiles ?? {};
  // Forced continuations remain selectable even if Disable aborts execution at BeforeMove.
  if (selecting && (v.lockedmove?.move===id || v.twoturnmove?.move===id))return '';
  if (v.disable?.move===id)return 'disable';
  if (selecting && definition.flags?.cantusetwice && pokemon.lastMove===id)return 'cantusetwice';
  if (v.lockedmove && v.lockedmove.move!==id)return 'lockedmove';
  if (v.healblock && definition.flags?.heal)return 'healblock';
  if (v.throatchop && definition.flags?.sound)return 'throatchop';
  const statusMove=definition.category==='Status' || definition.category===undefined && !definition.power;
  if (v.taunt && statusMove)return 'taunt';
  if (v.choicelock && CHOICE_ITEMS.includes(pokemon.itemId) && v.choicelock.move!==id)return 'choicelock';
  if (selecting && v.encore && v.encore.move!==id)return 'encore';
  if (selecting && v.twoturnmove && v.twoturnmove.move!==id)return 'twoturnmove';
  if (selecting && ['fakeout','firstimpression'].includes(id) && pokemon.activeMoveActions)return 'firstturn';
  return '';
}
export function legalMoveSlots(pokemon,definitions,{hasRevivalTarget=true}={}) {
  return pokemon.moves.map((slot,index)=>{
    const definition=definitions[slot.id];requireRule(definition,'Unknown battle move');
    const charging=pokemon.volatiles?.twoturnmove?.move===slot.id || pokemon.volatiles?.lockedmove?.move===slot.id;
    const reason=slot.pp<=0 && !charging ? 'nopp' : moveRestriction(pokemon,slot.id,definition,{selecting:true});
    const revivalTargetUnavailable=slot.id==='revivalblessing'&&!hasRevivalTarget;
    return {index,id:slot.id,disabled:Boolean(reason)||revivalTargetUnavailable,reason:reason||(revivalTargetUnavailable?'revivaltarget':'')};
  });
}
export function canStruggle(pokemon,definitions,options) {
  return legalMoveSlots(pokemon,definitions,options).every(slot=>slot.disabled);
}
export function registerMoveUsed(pokemon,id) {
  pokemon.lastMove=id;
  if (id!=='struggle' && CHOICE_ITEMS.includes(pokemon.itemId)) {
    pokemon.volatiles ??= {};
    pokemon.volatiles.choicelock ??= {move:id};
  }
}
export function applyTaunt(pokemon,{willMove=true,causeMoveId=null}={}) {
  pokemon.volatiles ??= {};
  if (!pokemon.hp || pokemon.volatiles.taunt)return false;
  pokemon.volatiles.taunt={duration:3+(pokemon.activeTurns && !willMove ? 1 : 0)};
  notifyEffectEvent(pokemon,{id:'taunt',...(causeMoveId==='taunt'?{causeMoveId}: {})});
  mentalHerb(pokemon);return true;
}
export function applyEncore(pokemon,definitions,{willMove=true,causeMoveId=null}={}) {
  pokemon.volatiles ??= {};
  const id=pokemon.lastMove,definition=definitions[id],slot=pokemon.moves.find(s=>s.id===id);
  if (!pokemon.hp || pokemon.volatiles.encore || !definition || definition.flags?.failencore || !slot || slot.pp<=0)return {applied:false};
  pokemon.volatiles.encore={move:id,duration:willMove ? 3 : 4};
  notifyEffectEvent(pokemon,{id:'encore',...(causeMoveId==='encore'?{causeMoveId}: {})});
  const herb=mentalHerb(pokemon);
  // Champions changes an already queued action immediately, instead of using
  // base Encore's OverrideAction event. The battle owner must update its queue.
  return {applied:true,override:willMove && !herb ? id : null};
}
export function applyDisable(pokemon,{willMove=true,activeMove=false,causeMoveId=null,causeAbilityId=null,causeSource=null}={}) {
  pokemon.volatiles ??= {};
  const slot=pokemon.moves.find(s=>s.id===pokemon.lastMove);
  if (!pokemon.hp || pokemon.volatiles.disable || !slot || !slot.pp || slot.id==='struggle')return false;
  pokemon.volatiles.disable={move:slot.id,duration:willMove || activeMove ? 4 : 5};
  notifyEffectEvent(pokemon,{id:'disable',...(causeMoveId==='disable'?{causeMoveId}: {}),...(causeAbilityId==='cursedbody'?{causeAbilityId,...(causeSource?{causeSource}:{})}: {})});
  mentalHerb(pokemon);return true;
}
export function applyThroatChop(pokemon,{causeMoveId=null}={}) {
  pokemon.volatiles ??= {};
  if (!pokemon.hp || pokemon.volatiles.throatchop)return false;
  pokemon.volatiles.throatchop={duration:2};notifyEffectEvent(pokemon,{id:'throatchop',...(causeMoveId==='throatchop'?{causeMoveId}:{})});return true;
}
export function applyHealBlock(pokemon) {
  pokemon.volatiles ??= {};
  if (!pokemon.hp || pokemon.volatiles.healblock)return false;
  pokemon.volatiles.healblock={duration:2};notifyEffectEvent(pokemon,{id:'healblock'});mentalHerb(pokemon);return true;
}
export function residualActionRestrictions(pokemon) {
  const v=pokemon.volatiles ?? {};
  for(const id of ['taunt','encore','disable','healblock','throatchop']) {
    if (v[id] && --v[id].duration<=0)endCondition(pokemon,id);
  }
  if (v.encore && !pokemon.moves.some(slot=>slot.id===v.encore.move && slot.pp>0))endCondition(pokemon,'encore');
  if (v.choicelock && (!CHOICE_ITEMS.includes(pokemon.itemId) || !pokemon.moves.some(slot=>slot.id===v.choicelock.move)))delete v.choicelock;
}
export function sleepTalkMove(pokemon,definitions,rng) {
  if (pokemon.status!=='slp')return null;
  const candidates=pokemon.moves.filter(slot=>{
    const move=definitions[slot.id];requireRule(move,'Unknown battle move');
    return !move.flags?.nosleeptalk && !move.flags?.charge;
  });
  // Called moves need not have PP: only Sleep Talk spends its own PP.
  return candidates.length ? candidates[rng.integer(candidates.length)].id : null;
}
export function solarBeamCharge(pokemon,weather) {
  pokemon.volatiles ??= {};
  if (pokemon.volatiles.twoturnmove?.move==='solarbeam') {
    delete pokemon.volatiles.twoturnmove;return {attack:true,spendPP:false};
  }
  if (weather==='sunnyday')return {attack:true,spendPP:true};
  pokemon.volatiles.twoturnmove={move:'solarbeam'};
  return {attack:false,spendPP:true};
}
export function electroShotCharge(pokemon,weather) {
  pokemon.volatiles ??={};
  if(pokemon.volatiles.twoturnmove?.move==='electroshot') {
    delete pokemon.volatiles.twoturnmove;return {attack:true,spendPP:false};
  }
  changeBoosts(pokemon,{spa:1},{effect:'electroshot'});
  if(weather==='raindance')return {attack:true,spendPP:true};
  pokemon.volatiles.twoturnmove={move:'electroshot'};
  return {attack:false,spendPP:true};
}
