// Fixed Champions item hooks: AfterMoveSecondary, once after all hits.
// Champions explicitly retains this event under Sheer Force; only move
// secondaries (not held-item switch hooks) are suppressed.
// These hooks run after move recoil, before move-user held-item hooks.
// Champions Eject Button preserves the attacking Pokemon's pivot request.
export function resolveSwitchItem({move,actor,target,result,targetHasBench,actorHasBench,
  actorActive=true,actorForceSwitch=false,targetForceSwitch=false,targetReturning=false,
  anyPlainSwitch=false,canDragActor=true,useItem=()=>true}) {
  if (!move || move.category==='Status' || actor===target ||
      !target.hp || !result?.hits || result.hits===(result.substituteHits ?? 0))return null;
  if (target.itemId==='ejectbutton') {
    if (move.flags?.futuremove || !targetHasBench || targetForceSwitch || targetReturning ||
        target.volatiles?.commanding || target.volatiles?.commanded || anyPlainSwitch)return null;
    if (!useItem(target))return null;
    consumeHeldItem(target,{notify:false});
    return {item:'ejectbutton',kind:'selected',pokemon:target,preserveAttackerPivot:true};
  }
  if (target.itemId==='redcard') {
    if (!actor.hp || !actorActive || !actorHasBench || actorForceSwitch || targetForceSwitch)return null;
    if (!useItem(target,actor))return null;
    consumeHeldItem(target,{notify:false});
    return {item:'redcard',kind:canDragActor ? 'forced' : 'consumed',pokemon:actor};
  }
  return null;
}
import { consumeHeldItem } from './battle-mechanics.js';
