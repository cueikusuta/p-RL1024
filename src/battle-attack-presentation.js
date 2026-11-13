import { projectConfirmedHitDamageFrame } from './battle-hit-frame-projection.js';

// An attack's public frames are a projection of already-resolved mechanics.
// The caller invokes each method at the existing hit callbacks; this module
// neither calculates damage nor advances the RNG or battle state.
export function createAttackHitPresentation({seat,id,target,getFrames,impact}) {
  let before=getFrames();
  let impactEvent;
  return {
    onImpact(effectiveness,hit) {
      impactEvent=impact(seat,id,effectiveness,hit,before);
    },
    onDamageMessagesComplete(event) {
      // Item resolution has already run. Freeze the loss before buffered
      // recovery messages are flushed, without changing battle-state HP.
      const hp=Math.max(0,event.beforeHP-event.damage);
      impactEvent.damageFrame=getFrames();
      projectConfirmedHitDamageFrame(impactEvent.damageFrame,{
        seat,targetPublicInstanceId:target.publicInstanceId,targetMaxHP:target.maxHP,hp,
      });
    },
    afterHit() {
      impactEvent.after=getFrames();
      before=impactEvent.after;
    },
  };
}
