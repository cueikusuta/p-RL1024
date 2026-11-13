// Issue #856: opt-in diagnostic flags for plausible human-visible AI mistakes.
// Flags are not proof of irrational play; the authoritative Battle chooses
// legality and actual outcome. Never export raw actions, sets or opponent data.
const RECOVERY=new Set(['recover','roost','slackoff','softboiled','milkdrink',
  'synthesis','moonlight','morningsun','rest','strengthsap']);
const HAZARD=new Set(['stealthrock','spikes','toxicspikes','stickyweb']);
const SCREEN=new Set(['reflect','lightscreen','auroraveil']);
const STATUS=new Set(['yawn','toxic','willowisp','thunderwave',
  'spore','sleeppowder','glare']);
const SETUP=new Set(['swordsdance','nastyplot','dragondance','quiverdance',
  'calmmind','bulkup','shellsmash','agility','irondefense']);

const zero=()=>({
  observedTurns:0,moveDecisions:0,voluntarySwitches:0,
  possibleSwitchCycles:0,recoveryAtHighHp:0,
  repeatedScreen:0,hazardAgainstLastOpponent:0,
  statusAgainstAlreadyStatused:0,setupAtHighBoost:0,
});
export function createHumanQualityObserver(){
  return {summary:zero(),privateSwitchHistory:[]};
}
export function observeHumanQualityDecision(observer,{view,action,moveId}){
  if(!observer?.summary||!Array.isArray(observer.privateSwitchHistory)||
     !view?.self||!view?.opponent||!action)
    throw new TypeError('Invalid human quality observation inputs');
  if(view.requestType!=='turn')return;
  const counts=observer.summary,actor=view.self.team?.[view.self.active];
  if(!actor)return;
  counts.observedTurns++;
  if(action.kind==='switch'){
    counts.voluntarySwitches++;
    const history=observer.privateSwitchHistory;
    const previous=history.at(-1);
    // Only count a contiguous run of one voluntary switch per *new turn*.
    // Any attack, utility choice, interrupted turn or repeated same-turn
    // signal resets the run. This is a conservative suspicious-cycle flag.
    if(previous&&view.turn!==previous.turn+1)
      history.length=0;
    history.push({turn:view.turn,index:action.index});
    if(history.length>4)history.shift();
    if(history.length===4&&
       history[0].index===history[2].index&&
       history[1].index===history[3].index&&
       history[0].index!==history[1].index)
      counts.possibleSwitchCycles++;
    return;
  }
  if(action.kind!=='move')return;
  observer.privateSwitchHistory.length=0;
  counts.moveDecisions++;
  if(typeof moveId!=='string')return;
  const hp=actor.hp/Math.max(1,actor.maxHP);
  if(RECOVERY.has(moveId)&&hp>=0.9)counts.recoveryAtHighHp++;
  // Choice only: even repeated existing hazards count as a suspicious
  // last-enemy choice, never as "newly installed" or a failed move.
  if(HAZARD.has(moveId)&&view.opponent.remaining<=1)
    counts.hazardAgainstLastOpponent++;
  if(STATUS.has(moveId)&&view.opponent.status)
    counts.statusAgainstAlreadyStatused++;
  if(SCREEN.has(moveId)&&(view.field?.sideEffects?.[0]??[]).some(e=>
    e.id===moveId&&(e.duration??1)>0))
    counts.repeatedScreen++;
  if(SETUP.has(moveId)&&
     Math.max(...Object.values(actor.boosts??{}).map(Number),0)>=3)
    counts.setupAtHighBoost++;
}
export function summarizeHumanQualityObservations(observer) {
  if(!observer?.summary)throw new TypeError('Missing human quality summary');
  return {...observer.summary,
    caution:'Only suspicious action indicators; a flag is NOT proof of a bad decision. No private action IDs or opponents are emitted.',
  };
}
