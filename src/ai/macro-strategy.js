// I737 P4: conservative macro adjustments to P2 ranked legal actions.
// This layer never generates a new action, guesses hidden opponent sets, or
// treats P1 matchup utility as a calibrated probability.
import {rankStrategyActions} from './action-evaluator.js';
import {rankTacticalStrategyActions} from './tactical-plans.js';

const finite=value=>typeof value==='number'&&Number.isFinite(value);
const hpPercent=p=>100*p.hp/Math.max(1,p.maxHP);
const signal=(model,key)=>model?.modelVersion==='matchup-p1-v1'&&
  model[key]?.status==='evaluated'&&finite(model[key].score)
  ?Math.max(-1,Math.min(1,model[key].score)):null;
const actionKey=(action,view)=>action.kind==='switch'
  ?'switch:'+action.index
  :action.kind==='move'
    ?'move:'+(typeof action.index==='number'
      ?view.self.team[view.self.active].moves?.[action.index]?.id
      :action.index)
    :'unknown';
const accuracy=(m)=>m?.accuracy===true?1:
  finite(m?.accuracy)?Math.max(0,Math.min(1,m.accuracy/100)):1;
const protectedKO=(view,action,matchup,data)=>{
  if(action.kind!=='move'||typeof action.index!=='number'||
     matchup?.modelVersion!=='matchup-p1-v1')return false;
  const active=view.self.team[view.self.active];
  const id=active.moves?.[action.index]?.id;
  // Explosion on our last living Pokemon forfeits the round under SPEC.md.
  // Do not reintroduce it via the P4 "guaranteed KO" bonus.
  if(id==='explosion'&&view.self.team.filter(p=>p.hp>0).length===1)
    return false;
  const forecast=matchup.moveEstimates?.find(row=>row.id===id);
  const move=data.moves?.[id];
  return !!move&&accuracy(move)>=1&&
    finite(forecast?.minPercent)&&
    finite(forecast?.immuneCandidateWeight)&&
    forecast.immuneCandidateWeight===0&&
    forecast.minPercent>=view.opponent.hpPercent*1.12;
};

function updateMemory(view,memory,action) {
  const current=actionKey(action,view);
  const recent=Array.isArray(memory.macroRecent)?memory.macroRecent:[];
  memory.macroRecent=[...recent.slice(-5),current];
  if(action.kind==='switch')memory.lastSwitchTurn=view.turn;
  else if(action.kind==='move'){
    memory.lastMoveId=typeof action.index==='number'
      ?view.self.team[view.self.active].moves[action.index]?.id:'struggle';
    memory.lastMoveTurn=view.turn;
  }
}

// ownMatchups is indexed by OWN battle slot and must be computed with the
// existing P1 public-belief evaluator. Values are not opponent true sets.
export function rankMacroStrategyActions(view,{data,memory={},matchup=null,
  switchMatchups=null,ownMatchups=null,roundWins=null}={}) {
  const ranked=rankStrategyActions(view,{data,memory,matchup,switchMatchups});
  if(!ranked.length)return ranked;
  const self=view.self,active=self.team[self.active];
  const healthy=self.team.map((p,i)=>({p,i})).filter(x=>x.p.hp>0);
  // The carryover reward is identically zero if winning this battle could
  // secure the fourth match win. A null match-score adds no carry bonus.
  const carryEligible=Number.isInteger(roundWins)&&roundWins>=0&&roundWins<3;
  const bySlot=Array.isArray(ownMatchups)&&ownMatchups.length===self.team.length
    ?ownMatchups:null;
  const liveAnswers=bySlot?healthy.filter(({i})=>{
    const risk=signal(bySlot[i],'breakthroughRisk');
    const offense=signal(bySlot[i],'attack');
    return risk!==null&&risk<0.25&&offense!==null&&offense>-0.2;
  }):[];
  const activeSoleAnswer=liveAnswers.length===1&&
    liveAnswers[0].i===self.active;
  const ownEffects=view.field?.sideEffects?.[0]??[];
  const screened=ownEffects.some(e=>
    ['reflect','lightscreen','auroraveil'].includes(e.id)&&
      (!finite(e.duration)||e.duration>1));
  const activeHP=hpPercent(active);
  const foeRemaining=view.opponent.remaining??3;
  const recent=Array.isArray(memory.macroRecent)?memory.macroRecent:[];
  const scored=ranked.map(row=>{
    const action=row.action;
    let delta=0;
    if(action.kind==='switch'){
      const reserve=self.team[action.index];
      const projection=bySlot?.[action.index]??switchMatchups?.[action.index];
      const reserveAttack=signal(projection,'attack');
      const safe=signal(projection,'switchIn');
      // Do not force an ace onto hazards or into an unmodelled counterattack.
      if(screened&&safe!==null&&safe>0.3&&reserveAttack!==null&&
         reserveAttack>0.1&&foeRemaining>1)delta+=16;
      // Protect a unique answer only when switching can save it AND the
      // reserve is a known non-suicidal destination.
      if(activeSoleAnswer&&activeHP<45&&safe!==null&&safe>-0.15)
        delta+=30;
      // Carryover value is a small, finite tie-break, never worth the round.
      if(carryEligible&&activeHP<30&&hpPercent(reserve)>55&&
         safe!==null&&safe>0)delta+=3;
    }
    if(action.kind==='move'){
      if(protectedKO(view,action,matchup,data)) {
        // Secure a near-term knockout rather than spend a turn setting up.
        // Damage is supported across public type hypotheses; turn order and
        // opponent priority still make this a tactical estimate, not certainty.
        delta+=foeRemaining===1?60:24;
      }
      if(activeSoleAnswer&&activeHP<35&&row.category==='setup')
        delta-=18;
    }
    // Detect the A-B-A-B sequence without banning legal repeated attacks.
    const key=actionKey(action,view);
    if(recent.length>=4&&recent.at(-4)===recent.at(-2)&&
       recent.at(-3)===recent.at(-1)&&recent.at(-2)===key&&
       (row.category==='switch'||['setup','screen','recovery'].includes(row.category)))
      delta-=38;
    return {...row,macroDelta:delta,score:row.score+delta};
  });
  // Stable sort preserves P2's deterministic tie breaks when macro
  // adjustments are equal; the macro layer does not choose new actions.
  scored.sort((a,b)=>b.score-a.score);
  return scored;
}

export function chooseMacroStrategyAction(view,options={}) {
  const actions=rankMacroStrategyActions(view,options);
  const action=actions[0]?.action??null;
  if(action)updateMemory(view,options.memory??{},action);
  return action;
}

// Opt-in P4+P5 composition. Keep the proven P2 legal-action set as the only
// authority for actions; use one P5 plan bonus on top of conservative P4
// adjustments. No new opponent information or hidden set access is allowed.
export function rankIntegratedStrategyActions(view,options={}) {
  const macro=rankMacroStrategyActions(view,options);
  if(!macro.length)return {rows:macro,plan:null,plans:[]};
  const tactical=rankTacticalStrategyActions(view,options);
  const plan=tactical.plan;
  const rows=macro.map(row=>{
    const applies=plan&&row.action.kind===plan.candidate.kind&&
      row.action.index===plan.candidate.index;
    const planBonus=applies?plan.bonus:0;
    return {...row,planId:applies?plan.id:null,planBonus,
      score:row.score+planBonus};
  }).sort((a,b)=>b.score-a.score);
  return {rows,plan,plans:tactical.plans};
}

// The selected action, not a rejected plan recommendation, updates memory.
export function chooseIntegratedStrategyAction(view,options={}) {
  const {rows,plan,plans}=rankIntegratedStrategyActions(view,options);
  const action=rows[0]?.action??null;
  if(!action)return null;
  const memory=options.memory??{};
  updateMemory(view,memory,action);
  const followed=plan&&action.kind===plan.candidate.kind&&
    action.index===plan.candidate.index;
  memory.tacticalPlan=followed
    ?{id:plan.id,turn:view.turn,state:plan.state}:null;
  // Optional Simulator-only observer: public planner states, never opponent truth.
  // It runs after action selection and cannot affect the ranked legal choice.
  if(typeof options.onTacticalChoice==='function')
    options.onTacticalChoice({
      proposed:plan?{id:plan.id,state:plan.state}:null,
      followed:!!followed,
      planStates:plans.map(p=>({id:p.id,state:p.state,feasible:!!p.candidate})),
    });
  return action;
}
