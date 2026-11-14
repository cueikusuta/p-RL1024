// Presentation-only projection for an already-resolved hit. The caller owns
// damage calculation, frame capture timing, and the presentation sequence.
// Only the attack target's public appearance receives an opponent HP update.
export function projectConfirmedHitDamageFrame(frames,{seat,targetPublicInstanceId,targetMaxHP,hp}) {
  for(const [viewer,frame] of frames.entries()) {
    if(viewer===1-seat)frame.self.team[frame.self.active].hp=hp;
    else {
      frame.opponent.hpPercent=hp===0?0:Math.max(1,Math.ceil(hp/targetMaxHP*100));
      for(const record of frame.revealed)if(record.publicInstanceId===targetPublicInstanceId)
        record.hpPercent=frame.opponent.hpPercent;
    }
  }
}
