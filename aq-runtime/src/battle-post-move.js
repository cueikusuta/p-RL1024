import { consumeHealingBerry, transferHeldItem, announceHeldItemEvent } from './battle-mechanics.js';
import { changeBoosts } from './battle-moves.js';
import { resolveSwitchItem } from './battle-switch-items.js';

// Applies post-hit secondary consequences in the existing callback order.
// Battle keeps turn ownership; the caller persists switchItem between callbacks.
export function resolvePostMoveEffects({
  id,effective,hits,actor,target,seat,teams,delayed,sourceActive,
  targetHpBefore,getActive,message,previousSwitchItem=null,
}) {
  let switchItem=previousSwitchItem;
  if (target.abilityId === 'berserk' && target.hp > 0 && targetHpBefore > target.maxHP/2 && target.hp <= target.maxHP/2)
    changeBoosts(target, {spa:1}, {effect:'berserk'});
  if (target.volatiles?.berserkItemPending) {
    delete target.volatiles.berserkItemPending;
    consumeHealingBerry(target);
  }
  const canTake = (holder, owner) => holder.itemId && holder.itemId !== holder.megaStoneId &&
    (!holder.hp || owner.abilityId === 'moldbreaker' || holder.abilityId !== 'stickyhold');
  // The reference sets source.switchFlag before AfterMoveSecondary for successful self-switch moves.
  const actorHasSwitchFlag = ['uturn','voltswitch','flipturn'].includes(id) && hits.hits > 0 && actor.hp > 0 &&
    actor === getActive(seat) && !actor.volatiles?.commanded;
  const targetForced = Boolean(effective.forceSwitch && hits.hits > hits.substituteHits && actor.hp && target.hp &&
    !target.volatiles?.ingrain && !['suctioncups','guarddog'].includes(target.abilityId) &&
    teams[1-seat].some(member => member !== target && member.hp > 0));
  if (hits.hits > hits.substituteHits && actor.hp && target.hp && !actorHasSwitchFlag && !targetForced &&
    !target.itemId && canTake(actor,target) && target.abilityId === 'pickpocket' && effective.flags?.contact) {
    transferHeldItem(actor, target);
  }
  const nextSwitchItem = switchItem ?? resolveSwitchItem({move:effective,actor,target,result:hits,
    actorActive:sourceActive,
    actorHasBench:teams[seat].some(member => member !== actor && member.hp > 0),
    targetHasBench:teams[1-seat].some(member => member !== target && member.hp > 0),
    targetForceSwitch:targetForced,
    canDragActor:!actor.volatiles?.ingrain && !['suctioncups','guarddog'].includes(actor.abilityId)});
  // The item has already been consumed. Publish once through the same
  // confirmed item event used by other activations, so the displayed message
  // and the opponent's observed-item history cannot disagree.
  if(nextSwitchItem && !switchItem)
    announceHeldItemEvent(target,{item:nextSwitchItem.item,kind:'activated'});
  switchItem = nextSwitchItem;
  // Target AfterMoveSecondary (including Red Card consumption) precedes
  // the source's AfterMoveSecondarySelf registration and item theft.
  if (!delayed && sourceActive && hits.hits > 0 && actor.hp && !actorHasSwitchFlag && !actor.itemId && !actor.volatiles?.gem &&
    canTake(target,actor) && actor.abilityId === 'magician') {
    transferHeldItem(target, actor);
  }
  return {switchItem,actorForcedSwitch:switchItem?.kind === 'forced'};
}
