// I737 P5: four reusable, publicly observable multi-turn tactical plans.
// This is an opt-in rank adjustment over P2's legal actions, not a fixed
// move script or an additional damage evaluator. Never read enemy true sets.
import {rankStrategyActions} from './action-evaluator.js';

export const STRATEGY_PLAN_IDS=Object.freeze([
  'hazard-yawn-phaze','screens-ace','boost-baton','trickroom-slow',
]);
const HAZARD=['stealthrock'];
const PHASE=['whirlwind','roar','dragontail','circlethrow'];
const SCREENS=['reflect','lightscreen','auroraveil'];
const BOOST=['swordsdance','nastyplot','dragondance','agility','quiverdance',
  'bulkup','calmmind','shellsmash','cosmicpower','irondefense'];
const moveAt=(view,action)=>action.kind==='move'&&typeof action.index==='number'
  ?view.self.team[view.self.active].moves?.[action.index]?.id:null;
const has=(side,id)=>side.some(e=>e.id===id&&
  (e.duration===undefined||e.duration>0));
const aliveReserves=view=>view.self.team.map((p,i)=>({p,i}))
  .filter(({p,i})=>i!==view.self.active&&p.hp>0);
const active=view=>view.self.team[view.self.active];
const isAttack=(id,data)=>!!data.moves?.[id]&&data.moves[id].category!=='Status';
const health=p=>100*p.hp/Math.max(1,p.maxHP);
const positive=p=>Object.values(p.boosts??{}).some(n=>n>0);
const choice=(rows,view,ids)=>rows.filter(r=>ids.includes(moveAt(view,r.action)));
const byMove=(rows,view,ids)=>choice(rows,view,ids).sort((a,b)=>b.score-a.score)[0];
const pick=(rows,category,condition=()=>true)=>rows.filter(r=>
  r.category===category&&condition(r)).sort((a,b)=>b.score-a.score)[0];
const result=(id,state,candidate=null,bonus=0,reason='')=>({
  id,state,candidate:candidate?.action??null,bonus,reason,
});
const validTurn=view=>view.requestType==='turn'&&
  !view.self.confirmed&&active(view)?.hp>0;
function planHazard(view,rows,data,memory) {
  const id='hazard-yawn-phaze';
  const own=active(view),enemyEffects=view.field?.sideEffects?.[1]??[];
  const threat=view.opponent;
  const capable=own.moves.some(m=>HAZARD.includes(m.id))&&
    own.moves.some(m=>m.id==='yawn')&&
    own.moves.some(m=>PHASE.includes(m.id));
  if(!capable)return result(id,'unavailable',null,0,'missing-participant-moves');
  if(health(own)<30||threat.remaining<=1)
    return result(id,'interrupted',null,0,'no-future-switch-pressure');
  const hazard=byMove(rows,view,HAZARD);
  const yawn=byMove(rows,view,['yawn']);
  const phase=byMove(rows,view,PHASE);
  const hasHazard=has(enemyEffects,'stealthrock');
  if(!hasHazard&&hazard&&hazard.score>-80)
    return result(id,'start',hazard,17,'add-switch-pressure');
  const recentlyYawned=memory.lastMoveId==='yawn'&&
    view.turn-(memory.lastMoveTurn??-99)<=2;
  if(hasHazard&&yawn&&yawn.score>-80&&!threat.status&&
     !recentlyYawned)return result(id,'continue',yawn,14,'force-response');
  if(hasHazard&&phase&&phase.score>-80&&
     (recentlyYawned||threat.status))
    return result(id,'continue',phase,14,'force-hazard-cycle');
  return result(id,hasHazard?'completed':'interrupted',null,0,
    'pressure-exhausted-or-action-unavailable');
}
function planScreens(view,rows,data) {
  const id='screens-ace',self=active(view);
  const ownEffects=view.field?.sideEffects?.[0]??[];
  const living=aliveReserves(view);
  const receivers=living.filter(({p})=>p.moves?.some(m=>isAttack(m.id,data)));
  const screenMoves=self.moves.filter(m=>SCREENS.includes(m.id)).map(m=>m.id);
  const anyScreen=ownEffects.some(e=>SCREENS.includes(e.id)&&
    (e.duration===undefined||e.duration>1));
  if(!screenMoves.length&&!anyScreen)
    return result(id,'unavailable',null,0,'no-screen-capability');
  if(!receivers.length)
    return result(id,'interrupted',null,0,'no-living-ace');
  if(health(self)<22)
    return result(id,'interrupted',null,0,'setter-at-risk');
  if(!anyScreen){
    const screen=byMove(rows,view,screenMoves);
    if(screen&&screen.score>-80)return result(id,'start',screen,17,'shield-ace');
    return result(id,'interrupted',null,0,'cannot-raise-screen');
  }
  // Do not feed an ace into unknown/lethal entry risk blindly: use P2's
  // currently legal and hazard-adjusted switch scores as the candidate set.
  const switching=rows.filter(r=>r.action.kind==='switch'&&
    receivers.some(x=>x.i===r.action.index));
  const best=switching.sort((a,b)=>b.score-a.score)[0];
  const duration=Math.max(...ownEffects.filter(e=>SCREENS.includes(e.id))
    .map(e=>e.duration??1),0);
  if(duration>1&&best&&best.score>0)
    return result(id,'continue',best,12,'enter-under-active-screen');
  return result(id,'interrupted',null,0,'screen-expiring-or-unsafe-switch');
}
function planBaton(view,rows,data) {
  const id='boost-baton',self=active(view);
  if(!self.moves.some(m=>m.id==='batonpass'))
    return result(id,'unavailable',null,0,'no-baton-pass');
  const receivers=aliveReserves(view).filter(({p})=>
    p.moves?.some(m=>isAttack(m.id,data)));
  if(!receivers.length)
    return result(id,'interrupted',null,0,'recipient-unavailable');
  if(view.requestType==='pivot'){
    const switches=rows.filter(r=>r.action.kind==='switch'&&
      receivers.some(x=>x.i===r.action.index));
    const best=switches.sort((a,b)=>b.score-a.score)[0];
    return result(id,best?'continue':'interrupted',best,best?24:0,
      best?'select-receiver':'no-legal-recipient');
  }
  const baton=byMove(rows,view,['batonpass']);
  if(positive(self)&&baton&&baton.score>-80)
    return result(id,'continue',baton,21,'pass-existing-boost');
  const boost=byMove(rows,view,BOOST);
  if(!positive(self)&&boost&&health(self)>65&&boost.score>0)
    return result(id,'start',boost,12,'prepare-baton');
  // Speed Boost can supply a useful boost without an explicit setup move.
  if(!positive(self)&&self.abilityId==='speedboost'&&
     health(self)>55&&byMove(rows,view,['protect']))
    return result(id,'start',byMove(rows,view,['protect']),8,
      'wait-for-speed-boost');
  return result(id,'interrupted',null,0,'no-safe-baton-window');
}
function planTrickRoom(view,rows,data) {
  const id='trickroom-slow',self=active(view);
  const living=aliveReserves(view).filter(({p})=>
    p.moves?.some(m=>isAttack(m.id,data))&&
    Number.isFinite(p.stats?.spe)&&Number.isFinite(self.stats?.spe)&&
    p.stats.spe<self.stats.spe);
  const hasRoom=(view.field?.trickRoomTurns??0)>0;
  if(!self.moves.some(m=>m.id==='trickroom')&&!hasRoom)
    return result(id,'unavailable',null,0,'no-trickroom-setter');
  if(!living.length)return result(id,'interrupted',null,0,'no-slow-recipient');
  if(!hasRoom){
    const trick=byMove(rows,view,['trickroom']);
    return result(id,trick&&trick.score>0&&health(self)>50?'start':'interrupted',
      trick&&trick.score>0&&health(self)>50?trick:null,
      trick&&trick.score>0&&health(self)>50?14:0,'turn-investment');
  }
  const remaining=view.field.trickRoomTurns;
  if(remaining<3)return result(id,'interrupted',null,0,'room-expiring');
  const best=rows.filter(r=>r.action.kind==='switch'&&
    living.some(({i})=>i===r.action.index)&&r.score>0)
    .sort((a,b)=>b.score-a.score)[0];
  return result(id,best?'continue':'interrupted',best,best?14:0,
    best?'enter-slow-attacker':'unsafe-or-no-switch');
}
const PLAN_EVALUATORS=[planHazard,planScreens,planBaton,planTrickRoom];

// Re-evaluate every turn. No two plans ever stack bonuses on one action.
// P2's high-confidence lethal attack has priority over a plan recommendation.
export function rankTacticalStrategyActions(view,options={}) {
  const {data,memory={},matchup=null,switchMatchups=null}=options;
  const rows=rankStrategyActions(view,{data,memory,matchup,switchMatchups});
  if(!rows.length)return {rows,plan:null,plans:[]};
  if(!validTurn(view)&&view.requestType!=='pivot')
    return {rows,plan:null,plans:[]};
  const plans=PLAN_EVALUATORS.map(fn=>fn(view,rows,data,memory));
  const feasible=plans.filter(p=>p.candidate&&p.bonus>0);
  const ranked=feasible.map(p=>{
    const target=rows.find(r=>r.action.kind===p.candidate.kind&&
      r.action.index===p.candidate.index);
    return {plan:p,adjusted:(target?.score??-Infinity)+p.bonus};
  }).sort((a,b)=>b.adjusted-a.adjusted||
    STRATEGY_PLAN_IDS.indexOf(a.plan.id)-
    STRATEGY_PLAN_IDS.indexOf(b.plan.id));
  let chosen=ranked[0]?.plan??null;
  const currentId=memory.tacticalPlan?.id;
  const sticky=ranked.find(row=>row.plan.id===currentId);
  if(sticky&&ranked[0]&&sticky.adjusted>=ranked[0].adjusted-4)
    chosen=sticky.plan;
  const lethal=rows.find(row=>row.category==='attack'&&
    row.action.kind==='move'&&typeof row.action.index==='number'&&
    matchup?.modelVersion==='matchup-p1-v1'&&
    matchup.moveEstimates?.some(m=>
      m.id===moveAt(view,row.action)&&
      Number.isFinite(m.minPercent)&&
      m.minPercent>=view.opponent.hpPercent*1.12&&
      m.immuneCandidateWeight===0&&
      (data.moves?.[m.id]?.accuracy===true||
       data.moves?.[m.id]?.accuracy===100))&&row.score>=40);
  if(lethal&&chosen&&lethal.score>=
    rows.find(r=>r.action.kind===chosen.candidate.kind&&
      r.action.index===chosen.candidate.index).score)
    chosen=null;
  const rescored=rows.map(row=>{
    const applies=chosen&&row.action.kind===chosen.candidate.kind&&
      row.action.index===chosen.candidate.index;
    return {...row,planId:applies?chosen.id:null,
      planBonus:applies?chosen.bonus:0,
      score:row.score+(applies?chosen.bonus:0)};
  }).sort((a,b)=>b.score-a.score);
  return {rows:rescored,plan:chosen,plans};
}
export function chooseTacticalStrategyAction(view,options={}) {
  const {rows,plan}=rankTacticalStrategyActions(view,options);
  const chosen=rows[0]?.action??null;
  if(!chosen)return null;
  const memory=options.memory??{};
  const id=moveAt(view,chosen);
  if(chosen.kind==='switch')memory.lastSwitchTurn=view.turn;
  else{memory.lastMoveId=id??'struggle';memory.lastMoveTurn=view.turn;}
  if(plan&&chosen.kind===plan.candidate.kind&&
     chosen.index===plan.candidate.index) {
    memory.tacticalPlan={id:plan.id,turn:view.turn,state:plan.state};
  } else memory.tacticalPlan=null;
  return chosen;
}
