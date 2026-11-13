import { setStatus } from './battle-status.js';
import { grounded } from './battle-grounding.js';
import { damagingHitAbility, updateUnnerve } from './battle-abilities.js';
import { updateWeatherSuppression } from './battle-field.js';
import { applyConfirmedPostHitEffects } from './battle-post-hit.js';
import { resetBoosts } from './battle-moves.js';
import { updateDynamicForms, updateDisguise } from './battle-forms.js';
import { abilityLabels } from '../data/catalog/labels-ja.js';
import { snapshotConditions, notifyConditionChanges } from './battle-messages.js';
import { createAttackHitPresentation } from './battle-attack-presentation.js';
import { resolvePostMoveEffects } from './battle-post-move.js';

// Keep condition-change announcements at the original hit callbacks' completion
// points, after the confirmed mutations of both participants.
function beginConditionNotices(actor,target,{causeMoveId=null}={}) {
  const before=new Map([actor,target].map(p=>[p,snapshotConditions(p)]));
  return () => {
    for(const [pokemon,previous] of before)notifyConditionChanges(pokemon,previous,{causeMoveId});
  };
}

export function createProductionAttackEvents(context) {
  const {
    id, move, actor, target, eventActor, field, seat, teams, totalFaintedBySide, data, rng,
    delayed, sourceActive, targetHpBefore,
    getActive, getFrames, impact, damageEvent, breakIllusion, message,
    publishAbilityActivation, publishAbilityLifecycle, publishEntryAbility, updateItems,
    flushSuppression,headlessPresentation=false
  } = context;
  // Only presentation bookkeeping is elided; the effect callbacks below
  // continue to run in exactly the same order and with the same RNG.
  const presentation=headlessPresentation
    ? {onImpact(){},onDamageMessagesComplete(){},afterHit(){}}
    : createAttackHitPresentation({seat,id,target,getFrames,impact});
  let switchItem = null;

  return {
    callbacks: {
      onImpact: presentation.onImpact,
      onDamageMessagesComplete: presentation.onDamageMessagesComplete,
      onHitComplete: event => {
        const active = [getActive(0), getActive(1)];
        // Per-hit Update runs before faint processing. Keep a holder's effects
        // until the move's hit loop completes, even at 0 HP.
        updateWeatherSuppression(field, active,{includeFainted:true});
        updateUnnerve(active,{includeFainted:true});
        // Disguise's Update changes form, then applies its one-eighth HP cost
        // before the same Update cycle evaluates held items.
        if(event?.disguise) {
          for(const pokemon of active)updateDisguise(pokemon,data);
          event.applyDisguiseDamage?.();
        }
        updateItems();
        updateDynamicForms(active, field, data);
        if(!event?.disguise)for(const pokemon of active)updateDisguise(pokemon,data);
        presentation.afterHit();
      },
      onAfterHitLoop: () => flushSuppression(),
      onAfterRecoil: () => updateItems(),
      onDamage: event => damageEvent({...event, futureMove:Boolean(move.flags?.futuremove)}),
      onTryHit: effective => {
        if (['brickbreak', 'ragingbull', 'psychicfangs'].includes(id))
          for (const effect of ['reflect', 'lightscreen', 'auroraveil']) delete field.sides[1-seat][effect];
        if (effective.breaksProtect) {
          const protections = ['protect', 'kingsshield', 'spikyshield'];
          const broke = protections.some(effect => target.volatiles[effect]);
          for (const effect of protections) delete target.volatiles[effect];
          if (broke) delete target.volatiles.stall;
        }
      },
      onBeforeDamage: (effective, event) => {
        if (target.abilityId === 'berserk' && !effective.multihit) {
          target.volatiles ??= {};
          target.volatiles.berserkItemPending = true;
        }
      },
      onAfterHit: (effective, event) => {
        const finishConditionNotices=beginConditionNotices(actor,target,{
          causeMoveId:id==='jawlock'?'jawlock':null,
        });
        if (id === 'clearsmog' && !event.substitute && target.hp) {
          resetBoosts(target,{notify:true});
        }
        if (id === 'jawlock' && !event.substitute && actor.hp && target.hp) {
          actor.volatiles.jawlock = {sourceId:target.instanceId};
          target.volatiles.jawlock = {sourceId:actor.instanceId};
        }
        finishConditionNotices();
      },
      onAfterDamagingHit: (effective, event) => {
        const finishConditionNotices=beginConditionNotices(actor,target);
        applyConfirmedPostHitEffects({id,effective,event,actor,target,field,seat,getActive});
        finishConditionNotices();
      },
      onDamagingHit: (effective, event) => {
        // Pinned spreadMoveHit includes numeric zero, but not Substitute hits.
        if (!event.substitute && typeof event.damage==='number') target.publicIdentityKnown=true;
        if (target.illusionInstanceId && !event.substitute && typeof event.damage==='number') breakIllusion(target);
        if (!event.substitute && target.volatiles?.beakblast && effective.flags?.contact &&
          actor.itemId !== 'protectivepads' && actor.hp) {
          setStatus(actor, 'brn', rng, {chart:data.typeChart,source:target,
            sunny:actor.abilityId==='megasol' || !field.weatherSuppressed && field.weather==='sunnyday',
            mistyTerrain:field.terrain==='mistyterrain',grounded:grounded(actor)});
        }
        const abilityBefore=target.abilityId;
        const actorAbilityBefore=eventActor.abilityId;
        const abilityActivated=damagingHitAbility(effective, eventActor, target, {rng,chart:data.typeChart,field,
          active:[getActive(0),getActive(1)],targetSeat:1-seat,abilityFlags:data.abilityFlags,
          damage:event.damage,substitute:event.substitute,teams,totalFaintedBySide,
          weather:field.weatherSuppressed?'':field.weather,terrain:field.terrain,grounded:pokemon=>grounded(pokemon),
          message,publishAbilityLifecycle,publishEntryAbility});
        // Mummy and Wandering Spirit announce their own canonical lifecycle
        // messages at the pinned point; do not re-log the pre-change ability.
        if(abilityActivated && eventActor.abilityId===actorAbilityBefore) {
          // The same pre-change holder/ability that used to produce the public
          // line now drives both the message and the observed ability history.
          if(!publishAbilityActivation?.(target,abilityBefore))
            message(`${target.speciesName} の ${abilityLabels[abilityBefore] ?? 'とくせい'}！`);
        }
        // Ending Illusion through a supported contact ability (Mummy) also
        // reveals the active user; ordinary switch cleanup remains silent.
        if(actor.illusionInstanceId && actor.abilityId!=='illusion')breakIllusion(actor);
      },
      onAfterMoveSecondary: (effective, hits) => {
        const result=resolvePostMoveEffects({
          id,effective,hits,actor,target,seat,teams,delayed,sourceActive,
          targetHpBefore,getActive,message,previousSwitchItem:switchItem,
        });
        switchItem=result.switchItem;
        return {actorForcedSwitch:result.actorForcedSwitch};
      }
    },
    getSwitchItem: () => switchItem
  };
}
