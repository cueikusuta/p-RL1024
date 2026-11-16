// Opt-in, seat-local anti-loop deployment policy, NOT a changed RNAD model.
// This accepts only the authority-accepted history of one's OWN actions;
// never consults the opponent's private team or creates new legal actions.
// Training remains on the original, uncensored RNAD policy unless explicitly
// redesigned as a separately versioned policy-gradient objective.
export const RNAD_SWITCH_GUARD='anti-loop-v1';
const validAction=a=>a&&typeof a==='object'&&!Array.isArray(a)&&
  (a.kind==='switch'||a.kind==='move');
export class RnadSwitchLoopGuard {
  constructor(policy){
    if(!policy||typeof policy.choose!=='function')
      throw new TypeError('Switch-loop guard needs a synchronous RNAD policy');
    this.policy=policy;
    this.last=null;
    this.pending=null;
  }
  reset(){
    this.last=null;
    this.pending=null;
    this.policy.reset?.();
  }
  choose({observation,legalActions,seat,round,rng}){
    if(this.pending)throw new Error('Previous switch-loop proposal not accepted');
    if(![0,1].includes(seat)||!Number.isSafeInteger(round)||round<1||
      !observation||!['draft','selection','battle'].includes(observation.phase)||
      !Array.isArray(legalActions)||legalActions.length<1)
      throw new TypeError('Invalid guarded seat-local decision');
    const ordinary=observation.phase==='battle'&&observation.requestType==='turn';
    const active=observation.self?.active;
    let allowed=legalActions;
    // Don't interfere with mandatory replacement, pivot, no-move (including
    // forced-only) and single legal option. The authoritative engine decides.
    const hasMove=legalActions.some(a=>a.kind==='move');
    const previous=this.last;
    if(ordinary&&hasMove&&legalActions.length>1&&
       Number.isInteger(active)&&previous?.round===round&&
       previous?.to===active){
      const filtered=legalActions.filter(action=>{
        if(action.kind!=='switch')return true;
        // After two consecutive ordinary switches, the next ordinary
        // decision must attack or use a legal status move, if available.
        if(previous.consecutive>=2)return false;
        // If an attacker is available, don't immediately undo the previous
        // voluntary switch. Alternative nonreversing retreat remains legal.
        return action.index!==previous.from;
      });
      if(filtered.length)allowed=filtered;
    }
    const proposed=this.policy.choose({
      observation,legalActions:allowed,seat,round,rng
    });
    if(!proposed||!validAction(proposed.action)&&
       observation.phase==='battle'||!allowed.some(a=>
         JSON.stringify(a)===JSON.stringify(proposed.action)))
      throw new TypeError('RNAD returned a nonlegal guarded proposal');
    this.pending={
      seat,round,phase:observation.phase,
      requestType:observation.requestType??null,
      from:active,to:proposed.action?.kind==='switch'?
        proposed.action.index:null,
      turn:observation.turn??null,action:structuredClone(proposed.action)
    };
    return proposed;
  }
  // Invoked after collector.commit() validates the one-use authority receipt.
  accepted({seat,round,phase,action}){
    const p=this.pending;
    if(!p||p.seat!==seat||p.round!==round||p.phase!==phase||
       JSON.stringify(p.action)!==JSON.stringify(action))
      throw new Error('Unaccepted or crossed-seat switch action');
    this.pending=null;
    if(phase!=='battle')return;
    if(p.requestType==='turn'&&action.kind==='switch'){
      const old=this.last;
      this.last={round,from:p.from,to:p.to,
        consecutive:old?.round===round&&old?.to===p.from?
          Math.min(3,old.consecutive+1):1};
    }else{
      // A voluntary move or a mandatory replacement breaks the loop chain.
      this.last=null;
    }
  }
}
